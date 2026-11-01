"""Command-line interface for the FloodNet reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodnet",
        description="Semi-supervised classification and segmentation on FloodNet UAV imagery.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and extract FloodNet Track 1.")
    download.add_argument("--dest", type=Path, default=Path("data/raw"))
    download.add_argument("--zip", type=Path, default=None, help="Use an existing local zip.")
    download.add_argument("--force", action="store_true", help="Force re-download.")
    download.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the SHA256 check of the HuggingFace archive.",
    )

    prepare = subparsers.add_parser("prepare", help="Validate data and build processed caches.")
    prepare.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split cap for quick experiments.",
    )

    cls = subparsers.add_parser("train-classifier", help="Train the ResNet18 classifier.")
    cls.add_argument("--config", type=Path, default=Path("configs/classification.yaml"))
    cls.add_argument("--epochs", type=int, default=None)
    cls.add_argument("--batch-size", type=int, default=None)
    cls.add_argument("--device", type=str, default=None)
    cls.add_argument("--smoke", action="store_true", help="Use one epoch and small batches.")
    cls.add_argument("--no-pretrained", action="store_true")

    seg = subparsers.add_parser("train-segmentor", help="Train DeepLabV3+ on FloodNet.")
    seg.add_argument("--config", type=Path, default=Path("configs/segmentation.yaml"))
    seg.add_argument(
        "--stage",
        choices=("supervised", "semi-supervised", "both"),
        default="both",
    )
    seg.add_argument("--supervised-epochs", type=int, default=None)
    seg.add_argument("--semi-supervised-epochs", type=int, default=None)
    seg.add_argument("--batch-size", type=int, default=None)
    seg.add_argument("--device", type=str, default=None)
    seg.add_argument("--resume", type=Path, default=None)
    seg.add_argument("--smoke", action="store_true", help="Use one epoch and small batches.")
    seg.add_argument("--no-pretrained", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate or predict a split.")
    evaluate.add_argument("--task", choices=("classification", "segmentation"), required=True)
    evaluate.add_argument(
        "--split",
        default="val",
        choices=("val", "test", "train", "official_val", "official_test"),
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=None)
    evaluate.add_argument("--output", type=Path, default=None)
    evaluate.add_argument("--device", type=str, default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument(
        "--legacy-invert",
        action="store_true",
        help="Apply the notebook's pred ^ 1 inversion to classification output.",
    )

    predict = subparsers.add_parser("predict", help="Run inference on an image or folder.")
    predict.add_argument("--task", choices=("classification", "segmentation"), required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--config", type=Path, default=None)
    predict.add_argument("--device", type=str, default=None)
    predict.add_argument(
        "--legacy-invert",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply the notebook's pred ^ 1 inversion for classification output. "
            "Defaults to configs/base.yaml legacy.classification_output_inversion."
        ),
    )
    predict.add_argument(
        "--output-size",
        type=str,
        default=None,
        help="Segmentation output size as WxH, default original image size.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        from floodnet.data.download import download_dataset

        download_dataset(
            dest=args.dest,
            zip_path=args.zip,
            force=args.force,
            verify=not args.no_verify,
        )
        return 0

    if args.command == "prepare":
        from floodnet.data.prepare import prepare_dataset

        prepare_dataset(
            config_path=args.config,
            dry_run=args.dry_run,
            force=args.force,
            limit=args.limit,
        )
        return 0

    if args.command == "train-classifier":
        from floodnet.engine.train_classifier import train_classifier_cli

        return train_classifier_cli(args)

    if args.command == "train-segmentor":
        from floodnet.engine.train_segmentor import train_segmentor_cli

        return train_segmentor_cli(args)

    if args.command == "evaluate":
        from floodnet.engine.evaluate import evaluate_cli

        return evaluate_cli(args)

    if args.command == "predict":
        from floodnet.engine.predict import predict_cli

        return predict_cli(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
