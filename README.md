# Helmet Detector

Train and run a helmet detector with the Ultralytics YOLO Python API. The model in this repository is a PyTorch `.pt` file exported from Ultralytics:

`modele/helmet-detection-v1.pt`

The bundled dataset export is:

`dataset/Helmet Detection Model Nd.ndjson`

It contains two classes:

| ID | Class |
| --- | --- |
| 0 | `no_helmet` |
| 1 | `helmet` |

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A GPU is optional. CPU works, but training will be slower.

Install dependencies:

```bash
uv sync
```

If you do not already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project Layout

```text
.
├── dataset/
│   └── Helmet Detection Model Nd.ndjson
├── modele/
│   └── helmet-detection-v1.pt
├── src/
│   └── helmet_detector/
├── pyproject.toml
└── README.md
```

## Predict From an Image Path or URL

Run the existing PyTorch model on one image:

```bash
uv run helmet-predict path/to/image.jpg
```

Example using the bundled sample image:

```bash
uv run helmet-predict "dataset/example/Biker avec casque.jpeg" --device cpu
```

Run prediction from an image URL:

```bash
uv run helmet-predict "https://example.com/image.jpg"
```

Show the annotated result in a window while predicting:

```bash
uv run helmet-predict path/to/image.jpg --show
```

Close the preview window when you are done. You can also press `q` or `Esc`.

Use a different model checkpoint:

```bash
uv run helmet-predict path/to/image.jpg --model runs/train/helmet-detector/weights/best.pt
```

Save YOLO text predictions with confidence scores:

```bash
uv run helmet-predict path/to/image.jpg --save-txt
```

Prediction images are saved by default under:

```text
runs/predict/helmet-detector/
```

Useful options:

```bash
uv run helmet-predict path/to/image.jpg --conf 0.4 --imgsz 640 --device cpu
uv run helmet-predict path/to/folder --conf 0.25
```

On Apple Silicon, you can try:

```bash
uv run helmet-predict path/to/image.jpg --device mps
```

## Prepare the Ultralytics NDJSON Dataset

The repository includes an Ultralytics Platform NDJSON export. Convert it to the standard YOLO folder structure before training:

```bash
uv run helmet-prepare-ndjson
```

This creates:

```text
data/helmet-yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

The NDJSON file stores signed image URLs. If downloads fail, re-export the dataset from Ultralytics Platform because signed URLs expire. You can also use local images that you downloaded manually:

```bash
uv run helmet-prepare-ndjson \
  --local-images-dir path/to/downloaded/images \
  --no-download
```

## Train With the Ultralytics Dataset

After preparing the dataset:

```bash
uv run helmet-train --data data/helmet-yolo/data.yaml --epochs 50 --imgsz 640
```

Train from the existing helmet checkpoint:

```bash
uv run helmet-train \
  --model modele/helmet-detection-v1.pt \
  --data data/helmet-yolo/data.yaml \
  --epochs 50
```

Training output is saved under:

```text
runs/train/helmet-detector/
```

The best checkpoint is usually:

```text
runs/train/helmet-detector/weights/best.pt
```

## Train With Your Own Pictures

You can also train from local pictures. For object detection, images alone are not enough to teach the model where helmets are; you need YOLO label files for real supervised training.

Expected label format for each image:

```text
class_id x_center y_center width height
```

All coordinates must be normalized from `0` to `1`.

Example:

```text
1 0.5125 0.2400 0.1800 0.2100
0 0.7100 0.2600 0.1200 0.1800
```

Prepare a local image dataset with labels:

```bash
uv run helmet-prepare-images \
  --images path/to/images \
  --labels path/to/yolo-labels \
  --output data/my-helmet-yolo
```

Then train:

```bash
uv run helmet-train --data data/my-helmet-yolo/data.yaml --epochs 50
```

If you pass images without labels, the script creates empty label files. That is useful for background images or dataset bootstrapping, but it will not teach the detector new helmet boxes.

```bash
uv run helmet-prepare-images \
  --images path/to/background-images \
  --output data/background-yolo
```

## Common Commands

```bash
# Install dependencies
uv sync

# Convert the bundled Ultralytics NDJSON dataset
uv run helmet-prepare-ndjson

# Train/fine-tune
uv run helmet-train --data data/helmet-yolo/data.yaml --epochs 50

# Predict one image
uv run helmet-predict path/to/image.jpg
```

## Notes

- The Ultralytics Platform model page is: <https://platform.ultralytics.com/nitish-platform/helmet-detection-train/helmet-detection-v1>
- Ultralytics supports YOLO datasets with `images/`, `labels/`, and `data.yaml`.
- The NDJSON export format contains dataset metadata followed by one JSON object per image.
