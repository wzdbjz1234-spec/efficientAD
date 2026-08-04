from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

from .artifacts import DEFAULT_SOURCE_DIR, ModelArtifacts
from .evaluation import EvaluationResult, evaluate_dataset
from .images import discover_images
from .predictor import EfficientADPredictor
from .visualization import save_feature_visualization, save_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_tools",
        description="Inference, visualization and evaluation for EfficientAD outputs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Resolve model artifacts")
    _add_model_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)

    infer_parser = subparsers.add_parser("infer", help="Infer one image or a directory")
    _add_model_arguments(infer_parser)
    infer_parser.add_argument(
        "--input", "--input-dir", dest="input", required=True,
        help="Image or directory of cropped ROI images",
    )
    infer_parser.add_argument("--output-dir", default=None, help="Save overlays and scores.csv")
    infer_parser.add_argument("--threshold", type=float, default=None)
    infer_parser.add_argument("--recursive", action="store_true")
    infer_parser.add_argument("--alpha", type=float, default=0.4)
    infer_parser.set_defaults(handler=_infer)

    visualize_parser = subparsers.add_parser(
        "visualize", help="Draw feature channels and anomaly maps"
    )
    _add_model_arguments(visualize_parser)
    visualize_parser.add_argument("--input", required=True, help="Image or directory")
    visualize_parser.add_argument("--output-dir", required=True)
    visualize_parser.add_argument("--top-k", type=int, default=8)
    visualize_parser.add_argument("--limit", type=int, default=None)
    visualize_parser.add_argument("--recursive", action="store_true")
    visualize_parser.set_defaults(handler=_visualize)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an MVTec-style test directory"
    )
    _add_model_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--data-dir",
        required=True,
        help="Product dataset directory or its test directory",
    )
    evaluate_parser.add_argument("--normal-class", default="good")
    evaluate_parser.add_argument("--output", default=None, help="Evaluation JSON path")
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold; defaults to the Youden threshold from this evaluation",
    )
    evaluate_parser.add_argument(
        "--misclassified-dir",
        default=None,
        help="Directory for false-positive/false-negative diagnostics",
    )
    evaluate_parser.add_argument(
        "--misclassified-top-k",
        type=int,
        default=8,
        help="Feature channels shown for each misclassified image",
    )
    evaluate_parser.add_argument(
        "--no-misclassified-visuals",
        action="store_true",
        help="Do not generate feature diagnostics for misclassified images",
    )
    evaluate_parser.set_defaults(handler=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="output run id (for example 12) or model artifact directory",
    )
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--product", default="my_product")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Training images used only when norm_params.json is missing",
    )
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--student", default=None)
    parser.add_argument("--autoencoder", default=None)
    parser.add_argument("--norm-cache", default=None)


def _artifacts(args: argparse.Namespace) -> ModelArtifacts:
    return ModelArtifacts.from_output(
        args.model,
        dataset=args.dataset,
        product=args.product,
        source_dir=args.source_dir,
    ).with_overrides(
        teacher=args.teacher,
        student=args.student,
        autoencoder=args.autoencoder,
        norm_params=args.norm_cache,
    )


def _predictor(args: argparse.Namespace) -> EfficientADPredictor:
    return EfficientADPredictor.load(
        _artifacts(args),
        device=args.device,
        train_dir=args.train_dir,
    )


def _inspect(args: argparse.Namespace) -> int:
    artifacts = _artifacts(args)
    artifacts.validate(require_norm=args.train_dir is None)
    print(json.dumps(artifacts.as_dict(), indent=2, ensure_ascii=False))
    return 0


def _infer(args: argparse.Namespace) -> int:
    images = discover_images(args.input, recursive=args.recursive)
    if not images:
        raise ValueError(f"No images found: {args.input}")
    predictor = _predictor(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'filename':<42} {'score':>12}  judgment")
    print("-" * 70)
    for image_path in tqdm(images, desc="Infer", unit="img"):
        prediction = predictor.predict(image_path)
        judgment = ""
        if args.threshold is not None:
            judgment = "ANOMALY" if prediction.is_anomaly(args.threshold) else "NORMAL"
        print(f"{image_path.name:<42} {prediction.score:>12.6f}  {judgment}")
        rows.append(
            {
                "path": str(image_path),
                "score": prediction.score,
                "student_teacher_score": prediction.student_teacher_score,
                "autoencoder_score": prediction.autoencoder_score,
                "judgment": judgment,
            }
        )
        if output_dir:
            save_overlay(
                image_path,
                prediction,
                _result_path(input_path, image_path, output_dir),
                alpha=args.alpha,
            )

    scores = [float(row["score"]) for row in rows]
    print("\nScore summary")
    print(f"  count:  {len(scores)}")
    print(f"  min:    {min(scores):.6f}")
    print(f"  max:    {max(scores):.6f}")
    print(f"  mean:   {statistics.fmean(scores):.6f}")
    print(f"  median: {statistics.median(scores):.6f}")
    if args.threshold is not None:
        anomaly_count = sum(score >= args.threshold for score in scores)
        print(f"  anomaly: {anomaly_count}/{len(scores)}")

    if output_dir:
        with (output_dir / "scores.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved overlays and scores to {output_dir}")
    return 0


def _visualize(args: argparse.Namespace) -> int:
    images = discover_images(args.input, recursive=args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No images found: {args.input}")

    predictor = _predictor(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(images, desc="Visualize", unit="img"):
        prediction = predictor.predict(image_path, include_features=True)
        save_feature_visualization(
            image_path,
            prediction,
            _result_path(input_path, image_path, output_dir),
            top_k=args.top_k,
        )
    print(f"Saved {len(images)} visualizations to {output_dir}")
    return 0


def _result_path(input_path: Path, image_path: Path, output_dir: Path) -> Path:
    if input_path.is_dir():
        relative = image_path.relative_to(input_path)
        return (output_dir / relative).with_suffix(".png")
    return output_dir / f"{image_path.stem}.png"


def _evaluate(args: argparse.Namespace) -> int:
    if args.train_dir is None:
        data_dir = Path(args.data_dir)
        candidate = data_dir / "train" / args.normal_class
        if candidate.is_dir():
            args.train_dir = str(candidate)
    predictor = _predictor(args)
    result = evaluate_dataset(
        predictor,
        args.data_dir,
        normal_class=args.normal_class,
        threshold=args.threshold,
    )
    metrics = result.metrics
    print(f"Samples:   {metrics['sample_count']}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"Youden threshold:  {metrics['best_threshold_youden']:.6f}")
    print(f"Decision threshold: {metrics['decision_threshold']:.6f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Confusion: {metrics['confusion_matrix']}")
    if not args.no_misclassified_visuals:
        visualization_dir = _misclassified_output_dir(args)
        result = _visualize_misclassifications(
            predictor,
            result,
            args.data_dir,
            visualization_dir,
            top_k=args.misclassified_top_k,
        )
    if args.output:
        output = result.save(args.output)
        print(f"Saved evaluation report to {output.resolve()}")
    return 0


def _misclassified_output_dir(args: argparse.Namespace) -> Path:
    if args.misclassified_dir:
        return Path(args.misclassified_dir).expanduser().resolve()
    if args.output:
        report = Path(args.output).expanduser().resolve()
        return report.parent / f"{report.stem}_misclassified"
    return Path("evaluation_misclassified").resolve()


def _visualize_misclassifications(
    predictor: EfficientADPredictor,
    result: EvaluationResult,
    data_dir: str | Path,
    output_dir: Path,
    *,
    top_k: int,
) -> EvaluationResult:
    if top_k < 1:
        raise ValueError("--misclassified-top-k must be at least 1")

    test_root = Path(data_dir).expanduser().resolve()
    if (test_root / "test").is_dir():
        test_root = test_root / "test"

    misclassified = [
        record
        for record in result.records
        if record.outcome in {"false_positive", "false_negative"}
    ]
    if not misclassified:
        print("Misclassified: 0; no diagnostic images generated")
        return result

    replacements = {}
    for record in tqdm(
        misclassified,
        desc="Visualize misclassified",
        unit="img",
    ):
        image_path = Path(record.path)
        try:
            relative = image_path.resolve().relative_to(test_root)
        except ValueError:
            relative = Path(record.defect_class) / image_path.name
        target = (
            output_dir / str(record.outcome) / relative
        ).with_suffix(".png")

        prediction = predictor.predict(image_path, include_features=True)
        actual = "normal" if record.target == 0 else "anomaly"
        predicted = "normal" if record.predicted == 0 else "anomaly"
        title = (
            f"{str(record.outcome).replace('_', ' ').upper()} | "
            f"actual={actual}, predicted={predicted}, "
            f"score={record.score:.6f}, "
            f"threshold={result.metrics['decision_threshold']:.6f}"
        )
        saved = save_feature_visualization(
            image_path,
            prediction,
            target,
            top_k=top_k,
            title=title,
        )
        replacements[record.path] = str(saved.resolve())

    updated_records = [
        replace(
            record,
            visualization_path=replacements.get(record.path),
        )
        for record in result.records
    ]
    fp_count = sum(record.outcome == "false_positive" for record in misclassified)
    fn_count = sum(record.outcome == "false_negative" for record in misclassified)
    print(
        f"Misclassified diagnostics: {len(misclassified)} "
        f"(FP={fp_count}, FN={fn_count}) -> {output_dir}"
    )
    return replace(result, records=updated_records)
