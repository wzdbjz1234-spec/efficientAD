"""Benchmark EfficientAD models from output/<id> on my_product4.

For each model: evaluate accuracy on the test set with the Youden
(optimal) threshold, measure per-image inference latency, and save
heatmap diagnostics for every misclassified image under
results/model<id>_my_product4/misclassified/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from efficientad_tools.artifacts import ModelArtifacts
from efficientad_tools.evaluation import EvaluationResult, evaluate_dataset
from efficientad_tools.images import discover_images
from efficientad_tools.predictor import EfficientADPredictor
from efficientad_tools.visualization import save_feature_visualization

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "mydataset" / "my_product4"
TEST_DIR = DATA_DIR / "test"
TRAIN_DIR = DATA_DIR / "train" / "good"
RESULTS_ROOT = PROJECT_ROOT / "results"

WARMUP_IMAGES = 10


@dataclass
class TimingSummary:
    count: int
    total_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def measure_latency(
    predictor: EfficientADPredictor, images: list[Path]
) -> list[float]:
    if not images:
        raise ValueError("No images to time")
    for _ in range(WARMUP_IMAGES):
        predictor.predict(images[0])
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for image_path in tqdm(images, desc="Measure latency", unit="img"):
        start = time.perf_counter()
        predictor.predict(image_path)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def summarize_latency(latencies: list[float]) -> TimingSummary:
    total = float(sum(latencies))
    return TimingSummary(
        count=len(latencies),
        total_ms=total,
        mean_ms=statistics.fmean(latencies),
        median_ms=statistics.median(latencies),
        std_ms=statistics.pstdev(latencies),
        min_ms=min(latencies),
        max_ms=max(latencies),
        fps=1000.0 / statistics.fmean(latencies),
    )


def visualize_misclassified(
    predictor: EfficientADPredictor,
    result: EvaluationResult,
    output_dir: Path,
    *,
    top_k: int,
) -> int:
    misclassified = [
        record
        for record in result.records
        if record.outcome in {"false_positive", "false_negative"}
    ]
    if not misclassified:
        return 0

    threshold = result.metrics["decision_threshold"]
    for record in tqdm(misclassified, desc="Save misclassified heatmaps", unit="img"):
        image_path = Path(record.path)
        relative = image_path.relative_to(TEST_DIR.resolve())
        target = (output_dir / str(record.outcome) / relative).with_suffix(".png")
        prediction = predictor.predict(image_path, include_features=True)
        actual = "normal" if record.target == 0 else "anomaly"
        predicted = "normal" if record.predicted == 0 else "anomaly"
        title = (
            f"{str(record.outcome).replace('_', ' ').upper()} | "
            f"actual={actual}, predicted={predicted}, "
            f"score={record.score:.6f}, threshold={threshold:.6f}"
        )
        save_feature_visualization(
            image_path,
            prediction,
            target,
            top_k=top_k,
            title=title,
        )
    return len(misclassified)


def write_summary(
    model_id: int,
    result: EvaluationResult,
    timing: TimingSummary,
    output_dir: Path,
) -> Path:
    misclassified = [
        record
        for record in result.records
        if record.outcome in {"false_positive", "false_negative"}
    ]
    summary = {
        "model": f"output/{model_id} (trainings/mvtec_ad/my_product4)",
        "dataset": str(TEST_DIR),
        "threshold_source": "Youden J from ROC curve",
        "decision_threshold": result.metrics["decision_threshold"],
        "metrics": result.metrics,
        "timing_ms": timing.as_dict(),
        "misclassified_count": len(misclassified),
        "misclassified_dir": str(output_dir / "misclassified"),
        "false_positive": [
            record.path
            for record in misclassified
            if record.outcome == "false_positive"
        ],
        "false_negative": [
            record.path
            for record in misclassified
            if record.outcome == "false_negative"
        ],
    }
    target = output_dir / "summary.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return target


def benchmark_model(model_id: int, device: str, top_k: int) -> None:
    model_dir = PROJECT_ROOT / "output" / str(model_id)
    artifacts = ModelArtifacts.from_output(model_dir, product="my_product4")
    predictor = EfficientADPredictor.load(
        artifacts,
        device=device,
        train_dir=str(TRAIN_DIR),
    )
    print(f"Device: {predictor.device}")

    images = discover_images(TEST_DIR, recursive=True)
    latencies = measure_latency(predictor, images)
    timing = summarize_latency(latencies)

    result = evaluate_dataset(predictor, TEST_DIR)
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
    print(
        f"Latency:   {timing.mean_ms:.1f} ms/image "
        f"(median {timing.median_ms:.1f}, std {timing.std_ms:.1f}, "
        f"min {timing.min_ms:.1f}, max {timing.max_ms:.1f}), "
        f"{timing.fps:.1f} fps"
    )

    output_dir = RESULTS_ROOT / f"model{model_id}_my_product4"
    result.save(output_dir / "results.json")
    summary_path = write_summary(model_id, result, timing, output_dir)
    count = visualize_misclassified(
        predictor, result, output_dir / "misclassified", top_k=top_k
    )
    print(f"Saved evaluation report to {summary_path.resolve()}")
    print(f"Misclassified heatmaps: {count} -> {output_dir / 'misclassified'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark accuracy, latency and misclassified heatmaps"
    )
    parser.add_argument("--model", nargs="+", type=int, default=[20, 21])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)
    for model_id in args.model:
        print("=" * 72)
        print(f"Model {model_id}")
        print("=" * 72)
        benchmark_model(model_id, args.device, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
