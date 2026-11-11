# Auto-Annotator (Grounding DINO)

Zero-shot image auto-annotation tool. Upload a zip of images, tell it what to find (e.g. `helmet`), and get back YOLO-format labels, preview images, and a report of which images need manual review. No training and no manual labelling are required for the first pass.

## Features

| Feature | What it does |
|---|---|
| **Smart frame sampling** | Skips near-identical frames (typical for CCTV or video exports) before detection, so you do not annotate the same scene hundreds of times. Skipped frames are listed in `report.csv` with the frame they matched |
| **Interactive box editor** | The **Review & edit** tab lets you correct false or inaccurate detections: move, resize, delete, add and relabel boxes. Edits go straight into the YOLO label files, previews and report |
| Zero-shot detection | Grounding DINO finds objects from a text prompt, no training needed |
| Review queue | Images with no detections or low-confidence boxes are listed first |

## How it works

```
images.zip  ->  extract  ->  skip near-identical frames  ->  Grounding DINO (text prompt: "helmet.")
            ->  NMS + filtering  ->  YOLO labels + previews + report.csv  ->  (optional) fix boxes in the editor
            ->  annotations.zip
```

1. The zip is extracted to a temporary folder. Images are sorted in natural order (`frame_2` before `frame_10`).
2. **Smart frame sampling** drops frames that barely differ from a recently kept frame (see below).
3. Each remaining image is passed to Grounding DINO together with your text prompt.
4. Detections are filtered by confidence, de-duplicated with Non-Maximum Suppression (NMS), and boxes covering more than 90% of the image are dropped.
5. Results are written in YOLO format and zipped for download.
6. In the **Review & edit** tab you can fix boxes, then press **Update annotations.zip** to download the corrected dataset.

## Is it calling an API, or running on my PC?

**The model runs locally on your PC. No API is called for inference, and no API key is needed.**

| Step | Where it happens |
|---|---|
| Model weights download | **One time only**, from the Hugging Face Hub (`IDEA-Research/grounding-dino-tiny`) on the first run |
| Detection / inference | **Locally** on your GPU (CUDA) or CPU |
| Your images | **Never uploaded anywhere**; they stay on your machine |
| Web UI (Gradio) | Runs a **local server** at `http://127.0.0.1:7860`. It is not shared publicly (no `share=True`) |

- The model is downloaded automatically by `from_pretrained()` and cached at `~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`).
- Approximate download size: about 0.7 GB for `grounding-dino-tiny`, about 1 GB for `grounding-dino-base`.
- After the first download you can run **fully offline**:
  ```bash
  export HF_HUB_OFFLINE=1        # Windows PowerShell: $env:HF_HUB_OFFLINE=1
  python auto_annotate.py
  ```
- Internet is only needed for `pip install` and the first model download.

## Everything used

| Component | Purpose |
|---|---|
| **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`, via Hugging Face `transformers`) | Open-vocabulary object detector: finds objects from a text prompt |
| **PyTorch** | Runs the model (GPU if available, otherwise CPU) |
| **torchvision** (`ops.nms`) | Non-Maximum Suppression to remove overlapping duplicate boxes |
| **Pillow** | Image loading, saving, and drawing preview boxes |
| **Gradio** | Local web UI for uploading the zip and downloading results |
| **tqdm** | Progress bar |
| **gradio_image_annotation** | Box-editor component used in the Review & edit tab (draw, move, resize, delete, relabel boxes) |
| **NumPy** | Frame comparison for smart sampling |
| Python standard library (`zipfile`, `shutil`, `csv`, `tempfile`, `argparse`, `pathlib`) | Zip handling, report writing, CLI |

## Requirements

- Python 3.9+
- Gradio 6.14 or newer (required by the box-editor component; `pip install -r requirements.txt` handles it)
- A CUDA GPU is recommended (the script uses it automatically). CPU works but is much slower (several seconds per image).
- About 2 GB of free disk space for the model and cache.

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For GPU support, install the CUDA build of PyTorch first from https://pytorch.org/get-started/locally/.

## Usage

### Web UI

```bash
python auto_annotate.py
```

Open the local URL printed in the terminal, upload your zip, enter the classes (default `helmet`; use commas for several, e.g. `helmet, person`), and press **Annotate images**. The page then shows a summary, per-class box counts, and a gallery of previews (images that need review are listed first). Download `annotations.zip` from the same page.

### Command line

```bash
python auto_annotate.py --zip images.zip --prompt "helmet"
# several classes:
python auto_annotate.py --zip images.zip --prompt "helmet, person"
```

Optional flags: `--box-thr 0.35`, `--text-thr 0.25`, `--val-split 0.2`, `--min-change 0.02` (frame sampling, `0` keeps every frame) and `--sample-window 30`.

The box editor is part of the web UI only; the command line writes the same dataset, which you can open in any labelling tool.

## Smart frame sampling

Video exports and CCTV clips contain long runs of almost identical frames. Annotating all of them wastes time and makes a training set that is mostly duplicates (and leaks near-copies between train and val).

Each image is shrunk to a 64x64 grayscale thumbnail and compared with the last 30 frames that were **kept**. A frame is skipped when less than **2%** of the picture changed compared with one of them (default; change it with the *Skip near-identical frames* slider or `--min-change`). Details:

- Pure brightness changes (auto-exposure, flicker) and sensor noise do not count as change.
- A person walking into an empty scene does count, so those frames are kept.
- Comparing with *kept* frames (not just the previous one) stops slow drift from letting duplicates through, and also catches a scene that returns to an earlier state.
- Skipped frames are not copied to the dataset. They appear in `report.csv` as `skipped_similar` with the kept frame they matched.
- If small objects far from the camera are being skipped, lower the value (for example `0.005`). If too many near-copies remain, raise it (for example `0.05`). `0` turns sampling off.

## Box editor

After annotating, open the **Review & edit** tab.

1. Pick what to show: **Needs review** (no detection or low confidence), **All images**, or **Edited**. Choose an image from the list, or use **Previous / Save & next**.
2. Fix the boxes: drag a box to move it, drag its handles to resize, press `Delete` to remove the selected box, or drag on empty space to draw a new one. The class label can be changed with the label tool in the toolbar; new boxes get the first class by default.
3. Changes are **saved automatically** when you move to another image, change the filter, or press **Save**. Saving rewrites that image's YOLO label file and preview and marks it `edited` in `report.csv`.
4. Press **Update annotations.zip** to rebuild the download with your corrections. The editor shows a reminder while the zip is out of date.

Notes: only the class names you typed can be used as labels. Boxes with no valid class or smaller than 2 pixels are not saved. Edited boxes have no confidence score. Edits live in the temporary working folder of the running app, so download the updated zip before closing it.

## Output structure

```
annotations.zip
├── data.yaml      # dataset config that YOLO (Ultralytics) expects
├── images/
│   ├── train/     # training images (.jpg)
│   └── val/       # validation images (.jpg)
├── labels/
│   ├── train/     # one YOLO .txt per image: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
│   └── val/
├── preview/       # images with detected boxes and scores drawn on them (for checking, not for training)
├── classes.txt    # class names (one per line)
└── report.csv     # per-image status: ok / low_confidence / no_detection / edited / skipped_similar / unreadable
```

`data.yaml` looks like this (class order matches the class IDs in the label files):

```yaml
train: images/train
val: images/val

nc: 2
names:
  0: "helmet"
  1: "person"
```

`path` is left out on purpose, so YOLO uses the folder that contains `data.yaml`. You can unzip the dataset anywhere and it still works. The train/val split is random but fixed (same seed every run), 80/20 by default. If you set the validation share to 0, `val` points at the train images instead.

Train directly on the result:

```bash
pip install ultralytics
yolo detect train data=/path/to/unzipped/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

An empty `.txt` label file means no helmet was detected in that image.

`report.csv` columns: `image, num_boxes, min_score, status, note`. `note` says which frame a skipped one matched (`skipped_similar`) or that you corrected the image (`edited`).

## Settings

| Setting | Default | Effect |
|---|---|---|
| Classes (prompt) | `helmet` | What to detect. Separate multiple classes with commas, e.g. `helmet, person`. Class IDs follow the order you type them (`helmet`=0, `person`=1) and are saved in `classes.txt` |
| Rename class | empty | Optional. With a single class, lets you prompt `hard hat` but save it as `helmet` |
| Validation share | `0.2` | Fraction of images held out for validation. `0` skips the split. CLI: `--val-split 0.2` |
| Box threshold | `0.35` | Higher = fewer but more confident boxes. Lower = more detections, more false positives |
| Text threshold | `0.25` | How strongly a box must match the prompt text |
| Review threshold | `0.5` | Images with any box scoring below this are flagged `low_confidence` |
| `MODEL_ID` (in script) | `grounding-dino-tiny` | Switch to `IDEA-Research/grounding-dino-base` for better accuracy (slower, larger) |
| NMS IoU | `0.5` | Overlap above which duplicate boxes are merged |
| Max box area | `0.9` | Boxes covering more than 90% of the image are discarded |
| Skip near-identical frames | `0.02` | Minimum share of the picture that must change versus recent kept frames. `0` keeps every frame. CLI: `--min-change` |

## Recommended workflow

1. Run the auto-annotator on your full dataset.
2. Open `report.csv` and review only the `low_confidence` and `no_detection` images. Check `preview/` to spot-check the rest.
3. Fix mistakes in the **Review & edit** tab (or in Label Studio, CVAT or Roboflow, which can import YOLO format), then press **Update annotations.zip**.
4. Train a YOLO model on the cleaned dataset.

## Limitations

- Zero-shot models are not perfect: expect some misses, false positives (e.g. caps, bike helmets, hair), and loose boxes. Always review a sample before training.
- Results depend heavily on the prompt wording and thresholds.
- Very small or heavily occluded helmets may be missed.
- Multiple classes are supported (comma-separated), but each class is detected in its own pass, so processing time grows with the number of classes.
- All output images are re-saved as JPEG.

## Troubleshooting

| Problem | Fix |
|---|---|
| `TypeError` about `box_threshold` / `threshold` | Handled automatically for old and new `transformers` versions. If it persists, run `pip install -U transformers` |
| Very slow | You are probably on CPU. Check with `python -c "import torch; print(torch.cuda.is_available())"` |
| CUDA out of memory | Use `grounding-dino-tiny`, or process a smaller batch of images |
| Model download fails | Check your internet connection or proxy. After one successful download it is cached |
| `ImportError` / "box editor needs one more package" | `pip install -r requirements.txt` (needs Gradio 6.14+ and `gradio_image_annotation`). Careful: the similarly named `gradio_image_annotator` on PyPI is a different, empty package |
| Frames I wanted are missing | Lower *Skip near-identical frames* (or `--min-change`), or set it to `0` |
| No images found in the zip | Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |

## Licenses

- Grounding DINO model weights on Hugging Face are released under the Apache 2.0 license. Check the model card at https://huggingface.co/IDEA-Research/grounding-dino-tiny for current terms.
- Make sure you have the right to use the images you annotate.
