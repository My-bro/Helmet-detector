from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import yaml
from ultralytics import YOLO


DEFAULT_MODEL = Path("modele/helmet-detection-v1.pt")
DEFAULT_NDJSON = Path("dataset/Helmet Detection Model Nd.ndjson")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _prediction_source(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value

    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist and source is not a valid URL: {value}")
    return str(path)


def _read_ndjson(path: Path) -> tuple[dict, list[dict]]:
    metadata: dict | None = None
    images: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_type = item.get("type")
            if item_type == "dataset":
                metadata = item
            elif item_type == "image":
                images.append(item)
            else:
                raise ValueError(f"Unsupported NDJSON row type on line {line_number}: {item_type}")

    if metadata is None:
        raise ValueError(f"No dataset metadata row found in {path}")

    return metadata, images


def _class_names(metadata: dict) -> dict[int, str]:
    class_names = metadata.get("class_names") or {}
    if not class_names:
        raise ValueError("NDJSON metadata does not contain class_names")
    return {int(key): str(value) for key, value in class_names.items()}


def _write_dataset_yaml(output_dir: Path, names: dict[int, str], has_test: bool) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if has_test:
        data["test"] = "images/test"

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return yaml_path


def _download_image(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, target)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"Could not download {url}. Ultralytics NDJSON URLs are signed and can expire; "
            "re-export the dataset if this happens."
        ) from exc


def _write_yolo_label(target: Path, boxes: list[list[float]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for box in boxes:
        if len(box) != 5:
            raise ValueError(f"Expected YOLO box with 5 values, got: {box}")
        class_id, x_center, y_center, width, height = box
        rows.append(f"{int(class_id)} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}")
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def prepare_ndjson_dataset(
    ndjson_path: Path,
    output_dir: Path,
    local_images_dir: Path | None,
    download: bool,
) -> Path:
    metadata, images = _read_ndjson(ndjson_path)
    names = _class_names(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_test = False
    for image in images:
        split = image.get("split") or "train"
        if split not in {"train", "val", "test"}:
            split = "train"
        has_test = has_test or split == "test"

        filename = image["file"]
        image_target = output_dir / "images" / split / filename
        label_target = output_dir / "labels" / split / f"{Path(filename).stem}.txt"

        source_image = local_images_dir / filename if local_images_dir else None
        if source_image and source_image.exists():
            image_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_target)
        elif download:
            _download_image(image["url"], image_target)

        boxes = image.get("annotations", {}).get("boxes", [])
        _write_yolo_label(label_target, boxes)

    yaml_path = _write_dataset_yaml(output_dir, names, has_test=has_test)
    print(f"Prepared YOLO dataset at {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def prepare_images_dataset(
    images_dir: Path,
    output_dir: Path,
    labels_dir: Path | None,
    val_ratio: float,
    names: dict[int, str],
) -> Path:
    image_paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    val_count = int(len(image_paths) * val_ratio)
    val_names = {path.name for path in image_paths[-val_count:]} if val_count else set()

    for image_path in image_paths:
        split = "val" if image_path.name in val_names else "train"
        image_target = output_dir / "images" / split / image_path.name
        label_target = output_dir / "labels" / split / f"{image_path.stem}.txt"

        image_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)

        source_label = labels_dir / f"{image_path.stem}.txt" if labels_dir else None
        if source_label and source_label.exists():
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, label_target)
        else:
            _write_yolo_label(label_target, [])

    yaml_path = _write_dataset_yaml(output_dir, names, has_test=False)
    print(f"Prepared local image dataset at {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def train_model(args: argparse.Namespace) -> None:
    model = YOLO(str(args.model))
    project = args.project.resolve()
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        pretrained=args.pretrained,
    )
    print(f"Training complete. Results saved to: {results.save_dir}")


def _show_prediction_windows(results: list) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("The --show flag requires OpenCV. Install dependencies with `uv sync`.") from exc

    window_name = "Helmet detector prediction"
    for result in results:
        image = result.plot()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        print("Close the prediction window, or press q / Esc, to continue.")

        while True:
            key = cv2.waitKey(100)
            if key in {27, ord("q"), ord("Q")}:
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(window_name)


def predict(args: argparse.Namespace) -> None:
    model = YOLO(str(args.model))
    project = args.project.resolve()
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        save=args.save,
        show=False,
        save_txt=args.save_txt,
        save_conf=args.save_txt,
    )

    for result in results:
        print(f"\nImage: {result.path}")
        if result.boxes is None or len(result.boxes) == 0:
            print("No detections.")
            continue

        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            print(f"- {names[class_id]} {confidence:.3f} bbox={xyxy}")

    if args.save:
        save_dir = results[0].save_dir if results else project / args.name
        print(f"\nPrediction images saved under: {save_dir}")

    if args.show:
        _show_prediction_windows(results)


def _parse_names(value: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        key, name = part.split(":", maxsplit=1)
        names[int(key.strip())] = name.strip()
    if not names:
        raise argparse.ArgumentTypeError("names must look like '0:no_helmet,1:helmet'")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helmet-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ndjson = subparsers.add_parser("prepare-ndjson", help="Convert an Ultralytics NDJSON export to YOLO format.")
    ndjson.add_argument("--ndjson", type=_existing_path, default=DEFAULT_NDJSON)
    ndjson.add_argument("--output", type=Path, default=Path("data/helmet-yolo"))
    ndjson.add_argument("--local-images-dir", type=_existing_path)
    ndjson.add_argument("--no-download", action="store_true", help="Only write labels/data.yaml; do not fetch image URLs.")
    ndjson.set_defaults(func=lambda args: prepare_ndjson_dataset(args.ndjson, args.output, args.local_images_dir, not args.no_download))

    images = subparsers.add_parser("prepare-images", help="Build a YOLO dataset from local images and optional YOLO labels.")
    images.add_argument("--images", type=_existing_path, required=True)
    images.add_argument("--labels", type=_existing_path)
    images.add_argument("--output", type=Path, default=Path("data/local-images-yolo"))
    images.add_argument("--val-ratio", type=float, default=0.2)
    images.add_argument("--names", type=_parse_names, default="0:no_helmet,1:helmet")
    images.set_defaults(func=lambda args: prepare_images_dataset(args.images, args.output, args.labels, args.val_ratio, args.names))

    train = subparsers.add_parser("train", help="Train or fine-tune the helmet detector.")
    train.add_argument("--data", type=_existing_path, default=Path("data/helmet-yolo/data.yaml"))
    train.add_argument("--model", type=_existing_path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=_positive_int, default=50)
    train.add_argument("--imgsz", type=_positive_int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default=None, help="Example: cpu, 0, 0,1, or mps.")
    train.add_argument("--project", type=Path, default=Path("runs/train"))
    train.add_argument("--name", default="helmet-detector")
    train.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(func=train_model)

    predict_parser = subparsers.add_parser("predict", help="Run prediction on an image, URL, folder, video, or stream path.")
    predict_parser.add_argument("source", type=_prediction_source, help="Path or http(s) URL to predict.")
    predict_parser.add_argument("--model", type=_existing_path, default=DEFAULT_MODEL)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--imgsz", type=_positive_int, default=640)
    predict_parser.add_argument("--device", default=None, help="Example: cpu, 0, 0,1, or mps.")
    predict_parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    predict_parser.add_argument("--name", default="helmet-detector")
    predict_parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    predict_parser.add_argument("--show", action="store_true", help="Display predictions and close cleanly when the window is closed.")
    predict_parser.add_argument("--save-txt", action="store_true")
    predict_parser.set_defaults(func=predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def prepare_ndjson_main() -> None:
    main(["prepare-ndjson", *sys.argv[1:]])


def prepare_images_main() -> None:
    main(["prepare-images", *sys.argv[1:]])


def train_main() -> None:
    main(["train", *sys.argv[1:]])


def predict_main() -> None:
    main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    main()
