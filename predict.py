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
    return (root / "models/yolov8n.pt").resolve()


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

    weights_path = _resolve_path(args.weights, root) if args.weights else _default_weights(root)
    source_path = _resolve_path(args.source, root)
    project_path = _resolve_path(args.project, root)

    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"source not found: {source_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
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
