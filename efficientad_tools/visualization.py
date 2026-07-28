from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .predictor import Prediction


def save_overlay(
    image_path: str | Path,
    prediction: Prediction,
    output_path: str | Path,
    *,
    alpha: float = 0.4,
) -> Path:
    """Save an OpenCV heatmap blended over the source image."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    anomaly_map = prediction.anomaly_map
    low = float(np.min(anomaly_map))
    high = float(np.max(anomaly_map))
    normalized = (anomaly_map - low) / max(high - low, 1e-12)
    heatmap = cv2.applyColorMap(
        np.clip(normalized * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), overlay):
        raise OSError(f"Failed to write visualization: {target}")
    return target


def save_feature_visualization(
    image_path: str | Path,
    prediction: Prediction,
    output_path: str | Path,
    *,
    top_k: int = 8,
    title: str | None = None,
) -> Path:
    """Save feature-channel diagnostics from ``predict(include_features=True)``."""

    if prediction.features is None:
        raise ValueError("Feature data is missing; call predict(include_features=True)")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    features = prediction.features
    st_strength = features.student_teacher_diff.mean(axis=(1, 2))
    ae_strength = features.autoencoder_diff.mean(axis=(1, 2))
    st_indices = np.argsort(st_strength)[::-1][:top_k]
    ae_indices = np.argsort(ae_strength)[::-1][:top_k]

    columns = top_k
    figure = plt.figure(figsize=(2.6 * columns, 24))
    grid = figure.add_gridspec(10, columns)
    if title:
        figure.suptitle(title, fontsize=14, fontweight="bold")

    input_axis = figure.add_subplot(grid[0, :])
    input_axis.imshow(rgb)
    input_axis.set_title(
        f"Input — combined score {prediction.score:.6f}",
        fontweight="bold",
    )
    input_axis.axis("off")

    _plot_channels(
        figure,
        grid,
        row=1,
        arrays=features.teacher,
        indices=st_indices,
        strengths=st_strength,
        label="Teacher",
        cmap="viridis",
    )
    _plot_channels(
        figure,
        grid,
        row=2,
        arrays=features.student_teacher,
        indices=st_indices,
        strengths=st_strength,
        label="Student-T",
        cmap="viridis",
    )
    _plot_channels(
        figure,
        grid,
        row=3,
        arrays=features.student_teacher_diff,
        indices=st_indices,
        strengths=st_strength,
        label="T/S diff",
        cmap="hot",
    )
    _plot_channels(
        figure,
        grid,
        row=4,
        arrays=features.autoencoder,
        indices=ae_indices,
        strengths=ae_strength,
        label="Autoencoder",
        cmap="viridis",
    )
    _plot_channels(
        figure,
        grid,
        row=5,
        arrays=features.student_autoencoder,
        indices=ae_indices,
        strengths=ae_strength,
        label="Student-AE",
        cmap="viridis",
    )
    _plot_channels(
        figure,
        grid,
        row=6,
        arrays=features.autoencoder_diff,
        indices=ae_indices,
        strengths=ae_strength,
        label="AE/S diff",
        cmap="hot",
    )

    _plot_anomaly_map(
        figure,
        grid,
        row=7,
        rgb=rgb,
        anomaly_map=prediction.student_teacher_map,
        title=f"Teacher / Student anomaly map — {prediction.student_teacher_score:.6f}",
    )
    _plot_anomaly_map(
        figure,
        grid,
        row=8,
        rgb=rgb,
        anomaly_map=prediction.autoencoder_map,
        title=f"Autoencoder / Student anomaly map — {prediction.autoencoder_score:.6f}",
    )
    _plot_anomaly_map(
        figure,
        grid,
        row=9,
        rgb=rgb,
        anomaly_map=prediction.anomaly_map,
        title=f"Combined anomaly map — {prediction.score:.6f}",
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return target


def _plot_anomaly_map(
    figure: plt.Figure,
    grid,
    *,
    row: int,
    rgb: np.ndarray,
    anomaly_map: np.ndarray,
    title: str,
) -> None:
    axis = figure.add_subplot(grid[row, :])
    axis.imshow(rgb)
    heat = axis.imshow(anomaly_map, cmap="hot", alpha=0.5)
    axis.set_title(title, fontweight="bold")
    axis.axis("off")
    figure.colorbar(heat, ax=axis, fraction=0.03, pad=0.02)


def _plot_channels(
    figure: plt.Figure,
    grid,
    *,
    row: int,
    arrays: np.ndarray,
    indices: np.ndarray,
    strengths: np.ndarray,
    label: str,
    cmap: str,
) -> None:
    for column, index in enumerate(indices):
        axis = figure.add_subplot(grid[row, column])
        axis.imshow(arrays[index], cmap=cmap)
        axis.set_title(f"{label} ch{index}\n{strengths[index]:.4g}", fontsize=8)
        axis.axis("off")
