from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from .images import discover_images
from .predictor import EfficientADPredictor


@dataclass(frozen=True)
class EvaluationRecord:
    path: str
    defect_class: str
    target: int
    score: float
    student_teacher_score: float
    autoencoder_score: float
    predicted: int | None = None
    outcome: str | None = None
    visualization_path: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    score_summary: dict[str, Any]
    records: list[EvaluationRecord]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "score_summary": self.score_summary,
            "records": [asdict(record) for record in self.records],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, ensure_ascii=False)
        return target


def evaluate_dataset(
    predictor: EfficientADPredictor,
    test_dir: str | Path,
    *,
    normal_class: str = "good",
    threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate all class directories below an MVTec-style ``test`` directory."""

    root = Path(test_dir).expanduser().resolve()
    if (root / "test").is_dir():
        root = root / "test"
    if not root.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {root}")

    records: list[EvaluationRecord] = []
    for class_dir in class_dirs:
        target = 0 if class_dir.name == normal_class else 1
        paths = discover_images(class_dir, recursive=True)
        for path in tqdm(paths, desc=f"Evaluate {class_dir.name}", unit="img"):
            prediction = predictor.predict(path)
            records.append(
                EvaluationRecord(
                    path=str(path),
                    defect_class=class_dir.name,
                    target=target,
                    score=prediction.score,
                    student_teacher_score=prediction.student_teacher_score,
                    autoencoder_score=prediction.autoencoder_score,
                )
            )

    targets = np.asarray([record.target for record in records])
    scores = np.asarray([record.score for record in records])
    if len(np.unique(targets)) != 2:
        raise ValueError(
            f"Evaluation requires normal and anomalous samples; found targets "
            f"{sorted(np.unique(targets).tolist())}"
        )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(targets, scores)
    best_index = int(np.argmax(true_positive_rate - false_positive_rate))
    best_threshold = float(thresholds[best_index])
    decision_threshold = best_threshold if threshold is None else float(threshold)
    predictions = (scores >= decision_threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    records = [
        replace(
            record,
            predicted=int(predicted),
            outcome=_outcome(record.target, int(predicted)),
        )
        for record, predicted in zip(records, predictions)
    ]

    metrics = {
        "sample_count": int(len(records)),
        "normal_count": int(np.sum(targets == 0)),
        "anomaly_count": int(np.sum(targets == 1)),
        "auroc": float(roc_auc_score(targets, scores)),
        "best_threshold_youden": best_threshold,
        "decision_threshold": decision_threshold,
        "threshold_source": "youden" if threshold is None else "command_line",
        "true_positive_rate": float(tp / max(tp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    score_summary = {
        "normal": _summary(scores[targets == 0]),
        "anomaly": _summary(scores[targets == 1]),
    }
    return EvaluationResult(
        metrics=metrics,
        score_summary=score_summary,
        records=records,
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _outcome(target: int, predicted: int) -> str:
    if target == 0:
        return "true_negative" if predicted == 0 else "false_positive"
    return "false_negative" if predicted == 0 else "true_positive"
