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


def _default_weights(root: Path) -> Path:
    best = (root / "runs/detect/signature_yolov8n/weights/best.pt").resolve()
    if best.exists():
        return best
    return (root / "yolov8n.pt").resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="")
    parser.add_argument("--source", default="datasets/signature/images/val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="signature_predict")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(__file__).resolve().parent
    device = _resolve_device(args.device)

    if args.weights:
        weights_spec: str | Path = _resolve_path(args.weights, root)
        if not Path(weights_spec).exists():
            raise FileNotFoundError(f"weights not found: {weights_spec}")
    else:
        default_weight_path = _default_weights(root)
        weights_spec = str(default_weight_path) if default_weight_path.exists() else "yolov8n.pt"

    source_path = _resolve_path(args.source, root)
    project_path = _resolve_path(args.project, root)

    if not source_path.exists():
        raise FileNotFoundError(f"source not found: {source_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_spec))
    model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
