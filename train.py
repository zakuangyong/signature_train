from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _resolve_device(device: str) -> str:
    if device.lower() not in {"auto", "cpu"} and not device.strip():
        return "auto"
    if device.lower() != "auto":
        return device

    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/signature/signature.yaml")
    parser.add_argument("--weights", default="models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="signature_yolov8n")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _create_fallback_model(YOLO) -> object:
    for spec in ("yolov8n.yaml", "yolov8.yaml"):
        try:
            return YOLO(spec)
        except Exception:
            pass

    try:
        import ultralytics

        root = Path(ultralytics.__file__).resolve().parent
        for rel in (
            "cfg/models/v8/yolov8.yaml",
            "models/v8/yolov8.yaml",
            "yolo/cfg/models/v8/yolov8.yaml",
        ):
            path = root / rel
            if path.exists():
                return YOLO(str(path))
    except Exception:
        pass

    return YOLO("yolov8n.yaml")


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(__file__).resolve().parent

    data_path = _resolve_path(args.data, root)
    weights_path = _resolve_path(args.weights, root)
    project_path = _resolve_path(args.project, root)
    device = _resolve_device(args.device)

    if not data_path.exists():
        raise FileNotFoundError(f"data yaml not found: {data_path}")

    from ultralytics import YOLO

    try:
        model_spec = str(weights_path) if weights_path.exists() else args.weights
        model = YOLO(model_spec)
    except ModuleNotFoundError:
        model = _create_fallback_model(YOLO)
    except (FileNotFoundError, ConnectionError, OSError):
        model = _create_fallback_model(YOLO)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
