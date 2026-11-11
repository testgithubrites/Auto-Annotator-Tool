"""
Auto-annotate images with Grounding DINO (zero-shot) and export YOLO labels.
Supports one or several classes, e.g. prompt "helmet" or "helmet, person".

Install:
    pip install -r requirements.txt

Run the web UI:
    python auto_annotate.py

Run from the command line:
    python auto_annotate.py --zip images.zip --prompt "helmet, person"
    python auto_annotate.py --zip frames.zip --min-change 0.03   # skip near-identical frames
    python auto_annotate.py --zip frames.zip --min-change 0      # keep every frame

Extra features:
  * Smart frame sampling: near-identical frames (e.g. from a static CCTV camera) are skipped
    before detection, so you do not annotate the same scene hundreds of times.
  * Interactive box editor (web UI, "Review & edit" tab): fix false or inaccurate detections
    by dragging, resizing, deleting, adding or relabelling boxes. Edits are written back to the
    YOLO labels, and annotations.zip can be rebuilt with your corrections.
"""
import argparse
import csv
import html
import inspect
import json
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # use "-base" for better accuracy
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Box colors by class id. The same colors are used in preview images and in the UI legend.
COLORS = ["#22C55E", "#EF4444", "#06B6D4", "#EAB308", "#D946EF", "#F97316", "#94A3B8", "#EC4899"]
GALLERY_LIMIT = 36

# Smart frame sampling
SAMPLE_MIN_CHANGE = 0.02   # a frame is kept only if >= 2% of the picture changed vs. recent kept frames
SAMPLE_WINDOW = 30         # how many recently kept frames each new frame is compared with
SAMPLE_PIXEL_THR = 0.05    # per-pixel brightness change (0-1, about 13 grey levels) that counts as "changed"
SIG_SIZE = 64              # frames are compared as 64x64 grayscale thumbnails

# Report statuses
REVIEW_STATUSES = {"no_detection", "low_confidence", "unreadable"}

_processor = None
_model = None


# --------------------------------------------------------------------------- model

def load_model():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return _processor, _model


def parse_classes(prompt):
    """'helmet, person' -> ['helmet', 'person'] (lowercase, de-duplicated, order kept)."""
    seen, out = set(), []
    for c in prompt.replace(".", ",").split(","):
        c = c.strip().lower()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@torch.no_grad()
def detect_one(image, phrase, box_thr, text_thr):
    """Run Grounding DINO for ONE phrase. Returns (boxes Tensor[N,4], scores Tensor[N])."""
    processor, model = load_model()
    inputs = processor(images=image, text=phrase + ".", return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    kwargs = dict(text_threshold=text_thr, target_sizes=[image.size[::-1]])
    try:  # newer transformers
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, **kwargs)[0]
    except TypeError:  # older transformers
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_thr, **kwargs)[0]
    return res["boxes"].cpu(), res["scores"].cpu()


def detect(image, classes, box_thr, text_thr, nms_iou, max_area_frac):
    """Detect every class separately (so labels never get mixed up).
    Returns list of (x1, y1, x2, y2, score, class_id)."""
    w, h = image.size
    all_boxes, all_scores, all_ids = [], [], []
    for cid, phrase in enumerate(classes):
        boxes, scores = detect_one(image, phrase, box_thr, text_thr)
        for b, s in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
            if (x2 - x1) * (y2 - y1) > max_area_frac * w * h:
                continue  # drop boxes covering almost the whole image
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(s)
            all_ids.append(cid)
    if not all_boxes:
        return []

    boxes_t = torch.tensor(all_boxes)
    keep = batched_nms(boxes_t, torch.tensor(all_scores), torch.tensor(all_ids), nms_iou)  # per-class NMS
    return [(*all_boxes[i], all_scores[i], all_ids[i]) for i in keep.tolist()]


def to_yolo(det, w, h):
    x1, y1, x2, y2, _, cid = det
    return f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


# ------------------------------------------------------------------ frame sampling

def natural_key(path):
    """Sort frame_2 before frame_10 (plain sorting would put frame_10 first)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def frame_signature(path):
    """Small grayscale thumbnail used to compare frames. The mean is removed so a pure
    brightness change (auto-exposure, flicker) does not count as motion."""
    with Image.open(path) as im:
        im.draft("L", (SIG_SIZE * 2, SIG_SIZE * 2))  # fast JPEG decode; no-op for other formats
        g = im.convert("L").resize((SIG_SIZE, SIG_SIZE), Image.BOX)
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a - a.mean()


def select_distinct_frames(files, min_change=SAMPLE_MIN_CHANGE, window=SAMPLE_WINDOW,
                           pixel_thr=SAMPLE_PIXEL_THR, iterator=None):
    """Drop frames that look almost the same as a recently kept frame.

    A frame is skipped when less than `min_change` (fraction of the picture, 0-1) differs from
    at least one of the last `window` kept frames. Comparing with a window (not only the previous
    frame) also catches a scene that returns to an earlier state, and comparing with *kept* frames
    (not the previous frame) stops slow drift from sneaking duplicates through.
    Returns (kept, skipped, unreadable); skipped is a list of (path, similar_to_path)."""
    kept, skipped, unreadable = [], [], []
    recent_sigs, recent_paths = [], []
    for p in (iterator if iterator is not None else files):
        try:
            sig = frame_signature(p)
        except Exception:
            unreadable.append(p)
            continue
        if recent_sigs:
            changed = (np.abs(np.stack(recent_sigs) - sig) > pixel_thr).mean(axis=(1, 2))
            j = int(changed.argmin())
            if changed[j] < min_change:
                skipped.append((p, recent_paths[j]))
                continue
        kept.append(p)
        recent_sigs.append(sig)
        recent_paths.append(p)
        if len(recent_sigs) > window:
            recent_sigs.pop(0)
            recent_paths.pop(0)
    return kept, skipped, unreadable


# ------------------------------------------------------------------------ previews

def render_preview(img, dets, names):
    """Copy of the image with boxes drawn. dets: (x1, y1, x2, y2, score_or_None, class_id)."""
    prev = img.copy()
    draw = ImageDraw.Draw(prev)
    for x1, y1, x2, y2, s, cid in dets:
        color = COLORS[cid % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 3, y1 + 3), names[cid] if s is None else f"{names[cid]} {s:.2f}", fill=color)
    return prev


def write_report(info):
    with open(info["out"] / "report.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "num_boxes", "min_score", "status", "note"])
        wr.writerows(info["rows"])


# ----------------------------------------------------------------------- pipeline

def write_data_yaml(out_dir, names, has_val):
    """Write the data.yaml that Ultralytics YOLO expects.
    `path` is left out on purpose: YOLO then uses the folder that contains data.yaml,
    so the dataset works wherever you unzip it."""
    lines = [
        "# YOLO dataset config. Paths are relative to the folder containing this file.",
        "train: images/train",
        "val: images/val" if has_val else "val: images/train  # no validation split was made; reusing train",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def run_pipeline(zip_path, prompt="helmet", class_name="", box_thr=0.35,
                 text_thr=0.25, nms_iou=0.5, max_area_frac=0.9,
                 review_below=0.5, val_split=0.2, min_change=SAMPLE_MIN_CHANGE,
                 sample_window=SAMPLE_WINDOW, progress=None):
    """Returns (path_to_annotations_zip, summary_text, info_dict).
    min_change: smart frame sampling cutoff (0 = keep every frame)."""
    classes = parse_classes(prompt)
    if not classes:
        raise ValueError("Enter at least one class to find, for example: helmet")
    names = list(classes)
    class_name = (class_name or "").strip()
    if len(classes) == 1 and class_name:
        names = [class_name]  # optional rename, single-class only

    work = Path(tempfile.mkdtemp())
    src, out = work / "src", work / "dataset"
    (out / "preview").mkdir(parents=True)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out / kind / split).mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(src)
    all_files = sorted((p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and "__MACOSX" not in p.parts),
                       key=natural_key)
    if not all_files:
        raise ValueError("No images found in the zip. Supported formats: jpg, jpeg, png, bmp, webp.")

    # Smart frame sampling: skip near-identical frames before spending time on detection.
    rows, skipped_rows = [], []
    files = all_files
    if min_change > 0 and len(all_files) > 1:
        it = progress.tqdm(all_files, desc="Skipping similar frames") if progress else tqdm(all_files, desc="Sampling")
        files, skipped, unreadable = select_distinct_frames(all_files, min_change, sample_window, iterator=it)
        for p in unreadable:
            rows.append([p.name, 0, 0.0, "unreadable", ""])
        for p, ref in skipped:
            skipped_rows.append([str(p.relative_to(src)), 0, 0.0, "skipped_similar",
                                 f"similar to {ref.relative_to(src)}"])
        if not files:
            raise ValueError("Every image was unreadable.")

    (out / "classes.txt").write_text("\n".join(names) + "\n")

    # Train/val split (fixed seed, so re-runs give the same split)
    n_val = 0
    if val_split > 0 and len(files) >= 2:
        n_val = max(1, min(len(files) - 1, round(len(files) * val_split)))
    val_idx = set(random.Random(0).sample(range(len(files)), n_val))
    write_data_yaml(out, names, has_val=n_val > 0)
    used_names = set()
    split_of, order_rank = {}, {}
    class_counts = [0] * len(names)
    iterator = progress.tqdm(files, desc="Annotating") if progress else tqdm(files)

    for i, p in enumerate(iterator):
        split = "val" if i in val_idx else "train"
        name = p.stem
        while name in used_names:  # avoid collisions from subfolders
            name += "_"
        used_names.add(name)

        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            rows.append([p.name, 0, 0.0, "unreadable", ""])
            continue

        dets = detect(img, classes, box_thr, text_thr, nms_iou, max_area_frac)
        w, h = img.size
        for d in dets:
            class_counts[d[5]] += 1

        img.save(out / "images" / split / f"{name}.jpg", quality=95)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(to_yolo(d, w, h) for d in dets))

        render_preview(img, dets, names).save(out / "preview" / f"{name}.jpg", quality=85)

        min_score = min((d[4] for d in dets), default=0.0)
        if not dets:
            status = "no_detection"
        elif min_score < review_below:
            status = "low_confidence"
        else:
            status = "ok"
        split_of[f"{name}.jpg"] = split
        order_rank[f"{name}.jpg"] = {"no_detection": 0, "low_confidence": 1}.get(status, 2)
        rows.append([f"{name}.jpg", len(dets), round(min_score, 3), status, ""])

    rows.extend(skipped_rows)  # skipped frames are listed in report.csv but get no labels
    info = {
        "names": names,
        "class_counts": class_counts,
        "n_boxes": sum(class_counts),
        "rows": rows,
        "preview_dir": out / "preview",
        "work": work,
        "out": out,
        "split_of": split_of,
        # review-first order used by the box editor (stable: edits do not reshuffle it)
        "edit_order": sorted(split_of, key=lambda n: (order_rank[n], natural_key(n))),
        "n_skipped": len(skipped_rows),
        "n_total": len(all_files),
        "n_edited": 0,
        "zip_stale": False,
    }
    refresh_counts(info)
    write_report(info)

    zip_out = shutil.make_archive(str(work / "annotations"), "zip", out)
    info["zip_path"] = zip_out
    n_kept = len(split_of)
    summary = (f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(names))}. "
               f"Annotated {n_kept} images ({n_kept - n_val} train, {n_val} val)"
               + (f"; skipped {len(skipped_rows)} near-identical frames" if skipped_rows else "")
               + f". {info['n_review']} need manual review (see report.csv).")
    return zip_out, summary, info


# ------------------------------------------------------------------------ box editor
# These functions work on the dataset folder that run_pipeline() built, so edits go straight
# into the YOLO label files, the preview image and report.csv.

EDIT_FILTERS = ("Needs review", "All images", "Edited")
STATUS_TEXT = {"no_detection": "nothing found", "low_confidence": "low confidence",
               "edited": "edited", "ok": "ok"}
EDIT_TOLERANCE_PX = 1.5  # the editor rounds boxes to whole pixels; ignore differences below this


def refresh_counts(info):
    info["row_of"] = {r[0]: r for r in info["rows"]}
    info["n_images"] = len(info["split_of"])
    info["n_boxes"] = sum(info["class_counts"])
    info["n_review"] = sum(r[3] in REVIEW_STATUSES for r in info["rows"])


def _paths(info, image_name):
    split = info["split_of"][image_name]
    stem = Path(image_name).stem
    return info["out"] / "images" / split / image_name, info["out"] / "labels" / split / f"{stem}.txt"


def read_label_boxes(label_path, w, h, n_classes):
    """YOLO txt -> [(class_id, x1, y1, x2, y2)] in pixels."""
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid, (xc, yc, bw, bh) = int(parts[0]), (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if 0 <= cid < n_classes:
                out.append((cid, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def load_for_editor(info, image_name):
    """Value for the box-editor component: the image plus its current boxes."""
    img_path, label_path = _paths(info, image_name)
    with Image.open(img_path) as im:
        w, h = im.size
    boxes = [{"xmin": round(x1), "ymin": round(y1), "xmax": round(x2), "ymax": round(y2),
              "label": info["names"][cid]}
             for cid, x1, y1, x2, y2 in read_label_boxes(label_path, w, h, len(info["names"]))]
    return {"image": str(img_path), "boxes": boxes}


def _same_boxes(old, new, tol=EDIT_TOLERANCE_PX):
    if len(old) != len(new):
        return False
    free = list(old)
    for cid, *c in new:
        hit = next((o for o in free if o[0] == cid and all(abs(a - b) <= tol for a, b in zip(o[1:], c))), None)
        if hit is None:
            return False
        free.remove(hit)
    return True


def save_editor_boxes(info, image_name, boxes):
    """Write the editor's boxes back to labels/, preview/ and report.csv.
    Returns (changed, message)."""
    names = info["names"]
    img_path, label_path = _paths(info, image_name)
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    w, h = img.size

    new, dropped = [], 0
    for b in boxes or []:
        label = str(b.get("label", "")).strip().lower()
        if label not in names:
            dropped += 1
            continue
        x1, x2 = sorted((max(0.0, min(w, b["xmin"])), max(0.0, min(w, b["xmax"]))))
        y1, y2 = sorted((max(0.0, min(h, b["ymin"])), max(0.0, min(h, b["ymax"]))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            dropped += 1
            continue
        new.append((names.index(label), x1, y1, x2, y2))
    warn = f" {dropped} box(es) had no valid class or were too small and were not saved." if dropped else ""

    old = read_label_boxes(label_path, w, h, len(names))
    if _same_boxes(old, new):
        return False, f"No changes to {image_name}." + warn

    label_path.write_text("\n".join(to_yolo((x1, y1, x2, y2, None, cid), w, h)
                                    for cid, x1, y1, x2, y2 in new))
    render_preview(img, [(x1, y1, x2, y2, None, cid) for cid, x1, y1, x2, y2 in new], names) \
        .save(info["preview_dir"] / image_name, quality=85)

    for cid, *_ in old:
        info["class_counts"][cid] -= 1
    for cid, *_ in new:
        info["class_counts"][cid] += 1
    row = info["row_of"][image_name]
    if row[3] != "edited":
        info["n_edited"] += 1
    row[1:5] = [len(new), "", "edited", "corrected in box editor"]
    refresh_counts(info)
    write_report(info)
    info["zip_stale"] = True
    return True, f"Saved {len(new)} box(es) for {image_name}." + warn


def _passes(info, name, filt):
    status = info["row_of"][name][3]
    if filt == "Edited":
        return status == "edited"
    if filt == "Needs review":
        return status in ("no_detection", "low_confidence")
    return True


def edit_choices(info, filt, current=None):
    """Dropdown entries (label, image_name) for the chosen filter. The current image is always listed."""
    out = []
    for name in info["edit_order"]:
        if name == current or _passes(info, name, filt):
            r = info["row_of"][name]
            out.append((f"{name}  -  {STATUS_TEXT.get(r[3], r[3])}, {r[1]} box(es)", name))
    return out


def neighbor(info, filt, current, step):
    """Next (step=1) or previous (step=-1) image that matches the filter, or None."""
    order = info["edit_order"]
    i = order.index(current) if current in order else (-1 if step > 0 else len(order))
    i += step
    while 0 <= i < len(order):
        if _passes(info, order[i], filt):
            return order[i]
        i += step
    return None


def rebuild_zip(info):
    """Re-create annotations.zip so it contains the edited labels."""
    info["zip_path"] = shutil.make_archive(str(info["work"] / "annotations"), "zip", info["out"])
    info["zip_stale"] = False
    return info["zip_path"]


# ----------------------------------------------------------------------------- UI

HERO_HTML = """
<div class="hero">
  <div class="hero-text">
    <h1>Label a dataset with one sentence</h1>
    <p>Upload a zip of images and say what to find. You get YOLO labels, preview images,
    and a short list of the pictures worth a second look. Detection runs on this computer.</p>
  </div>
  <div class="vf" aria-hidden="true">
    <div class="body"></div><div class="head"></div><div class="hat"></div>
    <div class="bx b-person"><span>person 0.91</span></div>
    <div class="bx b-hat"><span>helmet 0.94</span></div>
  </div>
</div>
"""

EMPTY_HTML = """
<div class="empty">
  <strong>Results will appear here</strong>
  Upload a zip, choose what to find, then press Annotate images.
</div>
"""

FOOT_HTML = """
<div class="foot">
  The first run downloads the Grounding DINO model once from Hugging Face. After that it works offline,
  and your images never leave this computer. Without a GPU, expect a few seconds per image for each class.
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&display=swap');
:root{
  --ink:#101A2B; --muted:#5B6779; --line:#D6DCE5; --surface:#FFFFFF; --accent:#2B59FF; --vf:#E3E8F0;
  --display:"Bricolage Grotesque","IBM Plex Sans",system-ui,sans-serif;
}
.dark{ --ink:#E8EDF7; --muted:#9AA7BD; --line:#26324A; --surface:#161E2E; --accent:#6F8DFF; --vf:#1C2740; }


.gradio-container{ max-width:1120px !important; margin:0 auto !important; }
footer{ display:none !important; }

.hero{ display:flex; gap:32px; align-items:center; justify-content:space-between; padding:28px 4px 12px; }
.hero h1{ font-family:var(--display); font-weight:700; font-size:clamp(2rem,4.2vw,3rem); line-height:1.05;
  letter-spacing:-0.02em; margin:0 0 14px; color:var(--ink); max-width:15ch; }
.hero p{ color:var(--muted); max-width:54ch; font-size:1.05rem; line-height:1.55; margin:0; }

.vf{ position:relative; width:260px; height:206px; flex:none; border-radius:12px; background:var(--vf);
  border:1px solid var(--line); overflow:hidden; }
.vf .body{ position:absolute; left:98px; top:92px; width:64px; height:86px; border-radius:26px 26px 6px 6px; background:#8895AB; }
.vf .head{ position:absolute; left:114px; top:56px; width:32px; height:32px; border-radius:50%; background:#A7B2C5; }
.vf .hat{ position:absolute; left:110px; top:52px; width:40px; height:22px; border-radius:40px 40px 6px 6px; background:#FFC93C; }
.vf .bx{ position:absolute; border:2px solid; border-radius:3px; }
.vf .bx span{ position:absolute; left:-2px; font:600 11px/16px system-ui,sans-serif; padding:0 6px; white-space:nowrap; }
.vf .b-hat{ left:105px; top:49px; width:50px; height:30px; border-color:#22C55E; }
.vf .b-hat span{ bottom:100%; background:#22C55E; color:#06210F; }
.vf .b-person{ left:70px; top:14px; width:120px; height:170px; border-color:#EF4444; }
.vf .b-person span{ top:100%; background:#EF4444; color:#fff; }

.chips{ gap:8px !important; flex-wrap:wrap !important; }
.chips button{ flex:0 0 auto !important; min-width:0 !important; border-radius:999px !important; font-size:.85rem !important; }

.empty{ border:1.5px dashed var(--line); border-radius:14px; padding:28px; color:var(--muted); line-height:1.5; }
.empty strong{ display:block; color:var(--ink); font:600 1.15rem var(--display); margin-bottom:6px; }

.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; }
.stat{ border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:var(--surface); }
.stat .num{ font:700 1.9rem/1.1 var(--display); color:var(--ink); }
.stat .lbl{ color:var(--muted); font-size:.85rem; margin-top:2px; }
.stat.flag{ border-color:#F59E0B; }
.legend{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.tag{ display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border:1px solid var(--line);
  border-radius:999px; font-size:.88rem; color:var(--ink); background:var(--surface); }
.tag i{ width:10px; height:10px; border-radius:2px; display:inline-block; }

.foot{ color:var(--muted); font-size:.85rem; line-height:1.5; padding:14px 4px 24px; max-width:80ch; }

@media (max-width:760px){ .hero{ flex-direction:column; align-items:flex-start; } }
"""


def stats_html(info):
    flag = " flag" if info["n_review"] else ""
    tags = "".join(
        f'<span class="tag"><i style="background:{COLORS[i % len(COLORS)]}"></i>'
        f'{html.escape(n)}: {c}</span>'
        for i, (n, c) in enumerate(zip(info["names"], info["class_counts"])))
    return (
        '<div class="stats">'
        f'<div class="stat"><div class="num">{info["n_images"]}</div><div class="lbl">images processed</div></div>'
        f'<div class="stat"><div class="num">{info["n_boxes"]}</div><div class="lbl">boxes found</div></div>'
        f'<div class="stat{flag}"><div class="num">{info["n_review"]}</div><div class="lbl">to review first</div></div>'
        + (f'<div class="stat"><div class="num">{info["n_skipped"]}</div>'
           f'<div class="lbl">near-identical frames skipped</div></div>' if info["n_skipped"] else "")
        + (f'<div class="stat"><div class="num">{info["n_edited"]}</div>'
           f'<div class="lbl">images corrected by you</div></div>' if info["n_edited"] else "")
        + '</div>'
        f'<div class="legend">{tags}</div>'
    )


def gallery_items(info):
    """Previews as (image, caption); images that need review come first."""
    order = {"no_detection": 0, "low_confidence": 1, "edited": 2, "ok": 3}
    labels = {"no_detection": "nothing found", "low_confidence": "low confidence", "edited": "edited"}
    rows = sorted((r for r in info["rows"] if r[3] in order), key=lambda r: order[r[3]])
    items = []
    for image, n_boxes, _, status, _note in rows[:GALLERY_LIMIT]:
        path = info["preview_dir"] / image
        if not path.exists():
            continue
        im = Image.open(path)
        im.thumbnail((720, 720))
        caption = f"{image}: {n_boxes} boxes" + (f" ({labels[status]})" if status in labels else "")
        items.append((im, caption))
    return items


EDITOR_KWARGS = dict(height=560, show_clear_button=False, show_remove_button=False,
                     show_download_button=False, show_share_button=False, sources=[],
                     boxes_alpha=0.15, box_thickness=3, box_selected_thickness=5)


def build_ui():
    import gradio as gr
    try:
        from gradio_image_annotation import image_annotator
    except ImportError:
        raise SystemExit("The box editor needs one more package:  pip install -r requirements.txt\n"
                         "(or: pip install gradio_image_annotation)")

    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#EDF0F4", body_background_fill_dark="#0E1420",
        block_background_fill="#FFFFFF", block_background_fill_dark="#161E2E",
        block_border_color="#D6DCE5", block_border_color_dark="#26324A",
        block_border_width="1px", block_shadow="none", block_shadow_dark="none",
        button_primary_background_fill="#2B59FF", button_primary_background_fill_hover="#1F45D6",
        button_primary_background_fill_dark="#4D74FF", button_primary_background_fill_hover_dark="#6F8DFF",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    )

    def editor_component(info, name):
        """A fresh editor component: class list, class colors and the first image to show."""
        names = info["names"]
        return image_annotator(
            value=load_for_editor(info, name) if name else None,
            label_list=names, label_colors=[COLORS[i % len(COLORS)] for i in range(len(names))],
            use_default_label=True, image_type="filepath", **EDITOR_KWARGS)

    def annotate(zip_file, prompt, rename, box_thr, text_thr, review_below, val_split,
                 min_change, progress=gr.Progress()):
        if not zip_file:
            raise gr.Error("Upload a zip of images first.")
        try:
            zip_out, _, info = run_pipeline(zip_file, prompt, rename, box_thr, text_thr,
                                            review_below=review_below, val_split=val_split,
                                            min_change=min_change, progress=progress)
        except zipfile.BadZipFile:
            raise gr.Error("That file is not a valid zip. Re-create the zip and upload it again.")
        except ValueError as e:
            raise gr.Error(str(e))
        filt = "Needs review" if any(_passes(info, n, "Needs review") for n in info["edit_order"]) else "All images"
        choices = edit_choices(info, filt)
        first = choices[0][1] if choices else None
        msg = ("Drag a box to move it, drag its handles to resize, press Delete to remove it, "
               "or drag on empty space to draw a new one. Changes save automatically when you change image.")
        return (stats_html(info), gallery_items(info), zip_out, info,
                gr.update(value=filt), gr.update(choices=choices, value=first), first,
                editor_component(info, first), msg, None)

    # ---- box editor handlers
    def _autosave(info, cur, ed):
        if info is None or not cur or not ed:
            return ""
        return save_editor_boxes(info, cur, ed.get("boxes"))[1]

    def _stale_note(info):
        return "  \nannotations.zip is out of date. Press **Update annotations.zip** to include your edits." \
            if info and info["zip_stale"] else ""

    def _show(info, filt, name, note):
        choices = edit_choices(info, filt, name)
        return (gr.update(choices=choices, value=name), name,
                gr.update(value=load_for_editor(info, name) if name else None),
                (note + _stale_note(info)).strip(), stats_html(info))

    def go_neighbor(step):
        def fn(info, filt, cur, ed):
            if info is None:
                raise gr.Error("Run the annotator first.")
            note = _autosave(info, cur, ed)
            target = neighbor(info, filt, cur, step)
            if target is None:
                target, note = cur, note + "  \nNo more images in this list."
            return _show(info, filt, target, note)
        return fn

    def go_pick(picked, info, filt, cur, ed):
        if info is None or not picked:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        note = _autosave(info, cur, ed)
        return _show(info, filt, picked, note)

    def go_filter(info, filt, cur, ed):
        if info is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        note = _autosave(info, cur, ed)
        return _show(info, filt, cur, note)

    def save_now(info, filt, cur, ed):
        if info is None:
            raise gr.Error("Run the annotator first.")
        note = _autosave(info, cur, ed)
        return _show(info, filt, cur, note)

    def update_zip(info, filt, cur, ed):
        if info is None:
            raise gr.Error("Run the annotator first.")
        note = _autosave(info, cur, ed)
        path = rebuild_zip(info)
        return (*_show(info, filt, cur, note + "  \nannotations.zip updated with your edits."), path, path)

    def refresh_gallery(info):
        return gallery_items(info) if info else gr.skip()

    def chip_value(v):
        return lambda: v

    blocks_kwargs, launch_kwargs = {"title": "Auto-Annotator"}, {}
    if "css" in inspect.signature(gr.Blocks.launch).parameters:  # Gradio 6+
        launch_kwargs.update(theme=theme, css=CSS)
    else:
        blocks_kwargs.update(theme=theme, css=CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(HERO_HTML)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                zip_in = gr.File(label="Image zip", file_types=[".zip"], type="filepath")
                classes = gr.Textbox(label="What should it find?", value="helmet",
                                     info="Separate several classes with commas.")
                with gr.Row(elem_classes="chips"):
                    for example in ("helmet", "helmet, person", "person, car"):
                        btn = gr.Button(example, size="sm")
                        btn.click(chip_value(example), None, classes)
                with gr.Accordion("Fine-tune detection", open=False):
                    rename = gr.Textbox(label="Save the class under another name",
                                        info="Only for a single class. Prompt 'hard hat', save it as 'helmet'.")
                    box_thr = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Box confidence cutoff",
                                        info="Raise it to remove false boxes. Lower it if objects are missed.")
                    text_thr = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Text match cutoff",
                                         info="How closely a box must match the words you typed.")
                    review_below = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Flag images below this score",
                                             info="Flagged images are listed first in the previews.")
                    val_split = gr.Slider(0, 0.5, value=0.2, step=0.05, label="Validation share",
                                          info="Share of images held out for validation. Set 0 to skip the split.")
                    min_change = gr.Slider(0, 0.2, value=SAMPLE_MIN_CHANGE, step=0.005,
                                           label="Skip near-identical frames",
                                           info="A frame is skipped when less than this share of the picture "
                                                "changed versus recent frames. 0 keeps every frame. Lower it if "
                                                "small objects far from the camera are being skipped.")
                run_btn = gr.Button("Annotate images", variant="primary", size="lg")
            with gr.Column(scale=7):
                stats = gr.HTML(EMPTY_HTML)
                with gr.Tabs():
                    with gr.Tab("Results") as results_tab:
                        gallery = gr.Gallery(label="Previews", columns=3, height=420, object_fit="contain")
                        out_file = gr.File(label="annotations.zip (labels, images, previews, report)")
                    with gr.Tab("Review & edit"):
                        with gr.Row():
                            filt = gr.Radio(list(EDIT_FILTERS), value="Needs review", label="Show")
                            img_pick = gr.Dropdown(label="Image", choices=[], interactive=True, filterable=True)
                        editor = image_annotator(label="Boxes", label_list=[], image_type="filepath",
                                                 **EDITOR_KWARGS)
                        with gr.Row():
                            prev_btn = gr.Button("Previous")
                            save_btn = gr.Button("Save")
                            next_btn = gr.Button("Save & next", variant="primary")
                        edit_msg = gr.Markdown("Run the annotator, then fix boxes here.")
                        zip_btn = gr.Button("Update annotations.zip")
                        edit_zip = gr.File(label="annotations.zip with your edits")
        gr.HTML(FOOT_HTML)

        info_state, cur_state = gr.State(None), gr.State(None)
        run_btn.click(annotate,
                      [zip_in, classes, rename, box_thr, text_thr, review_below, val_split, min_change],
                      [stats, gallery, out_file, info_state, filt, img_pick, cur_state, editor, edit_msg,
                       edit_zip])
        view_out = [img_pick, cur_state, editor, edit_msg, stats]
        ed_in = [info_state, filt, cur_state, editor]
        next_btn.click(go_neighbor(1), ed_in, view_out)
        prev_btn.click(go_neighbor(-1), ed_in, view_out)
        save_btn.click(save_now, ed_in, view_out)
        img_pick.input(go_pick, [img_pick, info_state, filt, cur_state, editor], view_out)
        filt.input(go_filter, ed_in, view_out)
        zip_btn.click(update_zip, ed_in, view_out + [out_file, edit_zip])
        results_tab.select(refresh_gallery, info_state, gallery)
    return demo, launch_kwargs


def launch_ui():
    demo, launch_kwargs = build_ui()
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--prompt", default="helmet", help='e.g. "helmet" or "helmet, person"')
    ap.add_argument("--class-name", default="", help="rename class (single-class only)")
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--val-split", type=float, default=0.2, help="share of images for validation (0 = none)")
    ap.add_argument("--min-change", type=float, default=SAMPLE_MIN_CHANGE,
                    help="skip frames where less than this share of the picture changed vs. recent frames "
                         "(0 = keep every frame)")
    ap.add_argument("--sample-window", type=int, default=SAMPLE_WINDOW,
                    help="how many recently kept frames each frame is compared with")
    args = ap.parse_args()

    if args.zip:
        result, msg, _ = run_pipeline(args.zip, args.prompt, args.class_name, args.box_thr, args.text_thr,
                                     val_split=args.val_split, min_change=args.min_change,
                                     sample_window=args.sample_window)
        print(msg, "\nOutput:", result)
    else:
        launch_ui()
