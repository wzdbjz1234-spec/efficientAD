#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch EfficientAD inference with ROI annotation and difference heatmaps.

The public interface accepts either one image or a directory. Each image is
processed exactly once and produces an annotated image, a difference heatmap,
and a machine-readable record containing score, class, timings, ROI, and model
weight paths.

Python:
    detector = EfficientADDetector(
        model_dir="EfficientAD-main/output/verytiny-batch=4",
        roi=(1282, 284, 478, 565),
        threshold=0.014,
    )
    runner = BatchDetector(detector)
    summary = runner.process("raw_images", "batch_results", recursive=True)

CLI:
    python batch_detector.py --input raw_images --output-dir batch_results \
        --model-dir EfficientAD-main/output/verytiny-batch=4 \
        --roi 1282,284,478,565 --mask 0,0,80,120 \
        --save-roi-config roi-with-masks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from roi_mask import load_roi_config, save_roi_config, valid_mask_from_config

# ── Defaults ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "EfficientAD-main" / "output" / "verytiny-batch=4"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "train"
DEFAULT_THRESHOLD = 0.014  # optimal threshold from weight sweep
DEFAULT_AE_WEIGHT = 0.025  # paired with DEFAULT_THRESHOLD for verytiny-batch=4
IMAGE_SIZE = 256
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
}

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════
@dataclass
class DetectionResult:
    """Single-image detection result."""
    is_anomaly: bool
    score: float
    label: str                    # "ANOMALY" or "NORMAL"
    inference_time_ms: float
    processing_time_ms: float
    threshold: float
    st_map_weight: float
    ae_map_weight: float
    anomaly_map: np.ndarray       # [H, W] — resized to ROI size
    roi: Tuple[int, int, int, int]


@dataclass
class BatchDetectionRecord:
    """Serializable result for one input image."""

    input_path: str
    annotated_path: str = ""
    heatmap_path: str = ""
    label: str = "ERROR"
    is_anomaly: Optional[bool] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    st_map_weight: float = 0.0
    ae_map_weight: float = DEFAULT_AE_WEIGHT
    inference_time_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None
    masks_json: str = "[]"
    scoring_mode: str = "weighted_ae_difference"
    student_weight: str = ""
    autoencoder_weight: str = ""
    normalization: str = ""
    error: str = ""


@dataclass
class BatchSummary:
    """Aggregate result returned by :class:`BatchDetector`."""

    input_path: str
    output_dir: str
    records: list[BatchDetectionRecord]
    elapsed_ms: float

    @property
    def succeeded(self) -> int:
        return sum(not record.error for record in self.records)

    @property
    def failed(self) -> int:
        return sum(bool(record.error) for record in self.records)

    @property
    def anomaly_count(self) -> int:
        return sum(record.is_anomaly is True for record in self.records)

    @property
    def normal_count(self) -> int:
        return sum(record.is_anomaly is False for record in self.records)


# ═══════════════════════════════════════════════════════════════════════
class EfficientADDetector:
    """
    Teacher-free EfficientAD detector with ROI cropping.

    Only runs Student + Autoencoder forward — skips Teacher entirely
    for ~2.6x faster inference with identical accuracy.

    Parameters
    ----------
    model_dir : path to training output directory
    roi : (x, y, width, height) pixel coordinates for cropping.
          If None, reads from roi_config.
    roi_config : path to JSON file with ROI coordinates (see roi_mask.py).
    threshold : anomaly score decision threshold.
    train_dir : path to training images (only needed once for norm_params).
    device : 'auto', 'cuda', or 'cpu'.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        roi_config: Optional[str | Path] = None,
        masks: Optional[list[Tuple[int, int, int, int]]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        ae_weight: float = DEFAULT_AE_WEIGHT,
        train_dir: Optional[str | Path] = None,
        device: str = "auto",
        dataset: str = "mvtec_ad",
        product: str = "my_product",
        student_weight: Optional[str | Path] = None,
        autoencoder_weight: Optional[str | Path] = None,
        norm_cache: Optional[str | Path] = None,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.threshold = float(threshold)
        self.ae_weight = float(ae_weight)
        if self.ae_weight < 0:
            raise ValueError("ae_weight must be non-negative")
        self.roi = roi
        self.dataset = dataset
        self.product = product
        self._student_weight_override = student_weight
        self._autoencoder_weight_override = autoencoder_weight
        self._norm_cache_override = norm_cache
        self.valid_input_mask = None
        self._mask_config_explicit = masks is not None or roi_config is not None

        config_masks: list[Tuple[int, int, int, int]] = []
        if roi_config is not None:
            config_roi, config_masks = load_roi_config(roi_config)
            if self.roi is None:
                self.roi = config_roi
        self.masks = list(config_masks if masks is None else masks)
        if self.roi is not None:
            self._validate_roi(self.roi)
        if self.masks and self.roi is None:
            raise ValueError("Masks require --roi or --roi-config")
        if self.masks:
            valid_mask_from_config(self.roi, self.masks)

        # Device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load models (student + autoencoder only, no teacher)
        self._load_models(train_dir)

    @property
    def weight_paths(self) -> dict[str, str]:
        """Absolute artifact paths used by this detector."""
        return {
            "student": str(self.student_weight_path),
            "autoencoder": str(self.autoencoder_weight_path),
            "normalization": str(self.norm_cache_path),
        }

    def save_roi_configuration(self, path: str | Path) -> Path:
        """Save the active ROI and ROI-relative masks as JSON."""
        if self.roi is None:
            raise ValueError("Cannot save ROI config when no ROI is configured")
        target = Path(path).expanduser().resolve()
        save_roi_config(self.roi, self.masks, target)
        return target

    # ── Public API ─────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run detection on a BGR image (numpy array).

        Returns DetectionResult with score, label, anomaly_map, timing.
        """
        processing_start = time.perf_counter()
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a non-empty BGR array with 3 channels")

        # 1. Crop ROI
        if self.roi is not None:
            roi_img = self._crop_roi(image)
        else:
            roi_img = image
        roi_h, roi_w = roi_img.shape[:2]

        # 2. Preprocess
        rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        tensor = _DEFAULT_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.valid_input_mask is not None:
            tensor = tensor * self.valid_input_mask

        # 3. Teacher-free inference: Student + AE only
        self._synchronize_device()
        inference_start = time.perf_counter()
        with torch.no_grad():
            student_out = self.student(tensor)
            ae_out = self.autoencoder(tensor)

            # Student layout: first teacher_channels → mimic Teacher,
            # last ae_channels → mimic Autoencoder.  Take the AE-mimicking tail.
            ae_channels = ae_out.shape[1]
            student_ae = student_out[:, -ae_channels:]  # last N channels match AE
            diff_ae = (ae_out - student_ae) ** 2
            map_ae = torch.mean(diff_ae, dim=1, keepdim=True)

            # Normalize
            map_ae = 0.1 * (map_ae - self.q_ae_start) / (
                self.q_ae_end - self.q_ae_start + 1e-6)

            # Mask
            if self.valid_input_mask is not None:
                feature_mask = F.interpolate(
                    self.valid_input_mask,
                    size=map_ae.shape[-2:],
                    mode="nearest",
                ).bool()
                map_ae = map_ae * feature_mask
            map_ae = self.ae_weight * map_ae
        self._synchronize_device()
        inference_time_ms = (time.perf_counter() - inference_start) * 1000

        # 4. Resize anomaly map to ROI size
        padded = F.pad(map_ae, (4, 4, 4, 4))
        resized = F.interpolate(
            padded, size=(roi_h, roi_w), mode="bilinear",
            align_corners=False,
        )
        if self.valid_input_mask is not None:
            output_mask = F.interpolate(
                self.valid_input_mask, size=(roi_h, roi_w),
                mode="nearest",
            )[0, 0].cpu().numpy()
        else:
            output_mask = np.ones((roi_h, roi_w), dtype=np.float32)

        anomaly_map = resized[0, 0].cpu().numpy() * output_mask
        score = float(np.max(anomaly_map))
        is_anomaly = score >= self.threshold
        processing_time_ms = (time.perf_counter() - processing_start) * 1000

        return DetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            label="ANOMALY" if is_anomaly else "NORMAL",
            inference_time_ms=inference_time_ms,
            processing_time_ms=processing_time_ms,
            threshold=self.threshold,
            st_map_weight=0.0,
            ae_map_weight=self.ae_weight,
            anomaly_map=anomaly_map,
            roi=self.roi or (0, 0, roi_w, roi_h),
        )

    def detect_and_annotate(
        self,
        image: np.ndarray,
        *,
        draw_heatmap: bool = True,
        heatmap_alpha: float = 0.45,
    ) -> tuple[DetectionResult, np.ndarray]:
        """Run one inference and return both its result and annotated image."""
        result = self.detect(image)
        annotated = self.annotate(
            image,
            result,
            draw_heatmap=draw_heatmap,
            heatmap_alpha=heatmap_alpha,
        )
        return result, annotated

    def detect_and_draw(
        self,
        image: np.ndarray,
        *,
        draw_heatmap: bool = True,
        heatmap_alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Detect and annotate the image.

        Draws: ROI rectangle (green), anomaly heatmap overlay,
        label + score + timing text.

        Returns annotated BGR image (copy, original untouched).
        """
        result = self.detect(image)
        return self.annotate(
            image,
            result,
            draw_heatmap=draw_heatmap,
            heatmap_alpha=heatmap_alpha,
        )

    def annotate(
        self,
        image: np.ndarray,
        result: DetectionResult,
        *,
        draw_heatmap: bool = True,
        heatmap_alpha: float = 0.45,
    ) -> np.ndarray:
        """Draw a previously computed result without running inference again."""
        if not 0 <= heatmap_alpha <= 1:
            raise ValueError("heatmap_alpha must be between 0 and 1")
        output = image.copy()
        h, w = output.shape[:2]

        # ── ROI rectangle ──
        rx, ry, rw, rh = result.roi
        cv2.rectangle(output, (rx, ry), (rx + rw - 1, ry + rh - 1),
                       (0, 255, 0), 2)

        # ── Heatmap overlay ──
        if draw_heatmap and result.anomaly_map is not None:
            amap = result.anomaly_map
            low, high = float(amap.min()), float(max(amap.max(), 1e-12))
            normalized = (amap - low) / max(high - low, 1e-12)
            heatmap = cv2.applyColorMap(
                np.clip(normalized * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            # Place heatmap in ROI region
            roi_area = output[ry:ry + rh, rx:rx + rw]
            blended = cv2.addWeighted(
                roi_area, 1 - heatmap_alpha, heatmap, heatmap_alpha, 0)
            valid = self.output_valid_mask(rh, rw).astype(bool)
            roi_area[valid] = blended[valid]
            output[ry:ry + rh, rx:rx + rw] = roi_area

        # ── Excluded mask rectangles (coordinates are relative to ROI) ──
        for index, (mx, my, mw, mh) in enumerate(self.masks, start=1):
            left, top = rx + mx, ry + my
            right, bottom = left + mw - 1, top + mh - 1
            mask_overlay = output.copy()
            cv2.rectangle(
                mask_overlay, (left, top), (right, bottom),
                (80, 80, 80), -1,
            )
            cv2.addWeighted(mask_overlay, 0.55, output, 0.45, 0, output)
            cv2.rectangle(
                output, (left, top), (right, bottom),
                (0, 255, 255), 2,
            )
            cv2.line(output, (left, top), (right, bottom), (0, 255, 255), 1)
            cv2.line(output, (right, top), (left, bottom), (0, 255, 255), 1)
            cv2.putText(
                output, f"MASK {index}", (left + 4, min(bottom, top + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
            )

        # ── Info panel ──
        color = (0, 0, 255) if result.is_anomaly else (0, 200, 0)
        label_text = f"{result.label}"
        score_text = f"score={result.score:.5f}"
        time_text = (
            f"threshold={result.threshold:.4f} | "
            f"inference={result.inference_time_ms:.1f}ms | "
            f"total={result.processing_time_ms:.1f}ms"
        )
        weight_text = (
            f"score_weights=ST:0.000 + AE:{result.ae_map_weight:.3f} | "
            f"models={self.student_weight_path.name} + "
            f"{self.autoencoder_weight_path.name}"
        )

        # Background bar at top
        bar_h = min(92, h)
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        cv2.putText(output, label_text, (12, 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(output, score_text, (12, 56),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(output, time_text, (12, 76),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(output, weight_text, (12, 90),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

        # ── ROI label ──
        cv2.putText(output, f"ROI({rx},{ry},{rw}x{rh})",
                     (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX,
                     0.45, (0, 255, 0), 1)

        return output

    def colorize_heatmap(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Convert a numerical difference map to a JET BGR heatmap."""
        low = float(np.min(anomaly_map))
        high = float(np.max(anomaly_map))
        normalized = (anomaly_map - low) / max(high - low, 1e-12)
        heatmap = cv2.applyColorMap(
            np.clip(normalized * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        valid = self.output_valid_mask(
            anomaly_map.shape[0], anomaly_map.shape[1]
        ).astype(bool)
        heatmap[~valid] = 0
        return heatmap

    def output_valid_mask(self, height: int, width: int) -> np.ndarray:
        """Return a boolean mask resized to an output ROI shape."""
        if self.valid_input_mask is None:
            return np.ones((height, width), dtype=np.uint8)
        valid = self._valid_roi_mask
        return cv2.resize(
            valid,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ── Internals ──────────────────────────────────────────────────

    @staticmethod
    def _validate_roi(roi: Tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI width/height must be positive: {roi}")

    def _crop_roi(self, image: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        ih, iw = image.shape[:2]
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            raise ValueError(
                f"ROI ({x},{y},{w},{h}) outside image ({iw}x{ih})")
        return image[y:y + h, x:x + w].copy()

    def _load_models(self, train_dir: Optional[str | Path]) -> None:
        """Load student + autoencoder. Compute norm params if needed."""
        artifacts_dir = self._resolve_artifacts_dir()
        student_path = self._resolve_override(
            self._student_weight_override,
            artifacts_dir / "student_final.pth",
        )
        ae_path = self._resolve_override(
            self._autoencoder_weight_override,
            artifacts_dir / "autoencoder_final.pth",
        )
        norm_path = self._resolve_override(
            self._norm_cache_override,
            artifacts_dir / "norm_params.json",
        )
        mask_path = artifacts_dir / "mask_config.json"
        self.student_weight_path = student_path
        self.autoencoder_weight_path = ae_path
        self.norm_cache_path = norm_path

        for name, p in [("student", student_path), ("autoencoder", ae_path)]:
            if not p.is_file():
                raise FileNotFoundError(f"Missing {name}: {p}")

        print(f"[Detector] Loading student + autoencoder (teacher-free mode)")
        self.student = torch.load(
            student_path, map_location=self.device, weights_only=False)
        self.autoencoder = torch.load(
            ae_path, map_location=self.device, weights_only=False)

        for m in (self.student, self.autoencoder):
            m.to(self.device)
            m.eval()

        n_params = sum(p.numel() for p in self.student.parameters()) + \
                   sum(p.numel() for p in self.autoencoder.parameters())
        print(f"[Detector] Model params: {n_params / 1e6:.2f}M (2 models)")
        print(f"[Detector] Device: {self.device}")

        # Command-line/config masks override a mask saved with model artifacts.
        # An explicitly empty mask list disables the artifact mask.
        if self.masks:
            self.valid_input_mask = self._build_valid_input_mask(
                self.roi, self.masks
            )
        elif not self._mask_config_explicit and mask_path.is_file():
            artifact_roi, artifact_masks = load_roi_config(mask_path)
            self.masks = artifact_masks
            if artifact_masks:
                self.valid_input_mask = self._build_valid_input_mask(
                    artifact_roi, artifact_masks
                )

        # ── Normalization ──
        if norm_path.is_file():
            self._load_norm_params(norm_path)
            print(f"[Detector] Loaded norm params from {norm_path}")
        elif train_dir is not None:
            print("[Detector] Computing norm params from training images...")
            self._compute_norm_params(Path(train_dir))
            self._save_norm_params(norm_path)
        else:
            raise FileNotFoundError(
                f"norm_params.json not found at {norm_path}. "
                f"Pass train_dir to compute it on first run.")

    def _resolve_artifacts_dir(self) -> Path:
        if (self.model_dir / "student_final.pth").is_file():
            return self.model_dir
        nested = (
            self.model_dir / "trainings" / self.dataset / self.product
        )
        return nested

    @staticmethod
    def _resolve_override(
        override: Optional[str | Path],
        default: Path,
    ) -> Path:
        if override is None:
            return default.resolve()
        return Path(override).expanduser().resolve()

    def _load_norm_params(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        self.q_ae_start = torch.as_tensor(
            data["q_ae_start"], dtype=torch.float32, device=self.device)
        self.q_ae_end = torch.as_tensor(
            data["q_ae_end"], dtype=torch.float32, device=self.device)

    def _save_norm_params(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_ae_start": float(self.q_ae_start.cpu()),
            "q_ae_end": float(self.q_ae_end.cpu()),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _compute_norm_params(self, train_dir: Path) -> None:
        """Compute AE map quantiles from training images."""
        from torch.utils.data import DataLoader, Dataset

        class _ImageSet(Dataset):
            def __init__(self, paths):
                self.paths = paths
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                with Image.open(self.paths[idx]) as img:
                    return _DEFAULT_TRANSFORM(img.convert("RGB"))

        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths.extend(train_dir.rglob(ext))
        if not paths:
            raise ValueError(f"No images found in {train_dir}")

        loader = DataLoader(
            _ImageSet(paths), batch_size=1, shuffle=False, num_workers=0)

        maps_ae = []
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device)
                if self.valid_input_mask is not None:
                    images = images * self.valid_input_mask
                student_out = self.student(images)
                ae_out = self.autoencoder(images)
                ae_ch = ae_out.shape[1]
                diff_ae = (ae_out - student_out[:, -ae_ch:]) ** 2
                m = torch.mean(diff_ae, dim=1, keepdim=True)
                if self.valid_input_mask is None:
                    maps_ae.append(m.flatten().cpu())
                else:
                    feature_mask = F.interpolate(
                        self.valid_input_mask,
                        size=m.shape[-2:],
                        mode="nearest",
                    ).bool()
                    maps_ae.append(m.masked_select(feature_mask).cpu())

        all_ae = torch.cat(maps_ae)
        self.q_ae_start = torch.quantile(all_ae, 0.9).to(self.device)
        self.q_ae_end = torch.quantile(all_ae, 0.995).to(self.device)
        print(f"[Detector] q_ae: start={self.q_ae_start:.4f} "
              f"end={self.q_ae_end:.4f}")

    def _build_valid_input_mask(
        self,
        roi: Tuple[int, int, int, int],
        masks: list[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        valid = valid_mask_from_config(roi, masks)
        if not np.any(valid):
            raise ValueError("Masks exclude the entire ROI")
        self._valid_roi_mask = valid
        tensor = torch.from_numpy(valid).float()[None, None]
        tensor = F.interpolate(
            tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest"
        )
        return tensor.to(self.device)


# ═══════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════


class BatchDetector:
    """Process one image or a directory through an injected detector."""

    def __init__(self, detector: EfficientADDetector):
        self.detector = detector

    def process(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        recursive: bool = True,
        draw_heatmap: bool = True,
        save_heatmap: bool = True,
        heatmap_alpha: float = 0.45,
        fail_fast: bool = False,
    ) -> BatchSummary:
        """Process all supported images and write annotated outputs and reports."""
        source = Path(input_path).expanduser().resolve()
        output_root = Path(output_dir).expanduser().resolve()
        images = _discover_images(
            source,
            recursive=recursive,
            excluded_root=output_root,
        )
        if not images:
            raise ValueError(f"No supported images found under: {source}")

        output_root.mkdir(parents=True, exist_ok=True)
        records: list[BatchDetectionRecord] = []
        batch_start = time.perf_counter()

        for index, image_path in enumerate(images, start=1):
            item_start = time.perf_counter()
            relative = (
                Path(image_path.name)
                if source.is_file()
                else image_path.relative_to(source)
            )
            annotated_path = (
                output_root
                / "annotated"
                / relative.parent
                / f"{relative.stem}_result.png"
            )
            heatmap_path = (
                output_root
                / "heatmaps"
                / relative.parent
                / f"{relative.stem}_heatmap.png"
            )

            try:
                image = _read_image(image_path)
                result, annotated = self.detector.detect_and_annotate(
                    image,
                    draw_heatmap=draw_heatmap,
                    heatmap_alpha=heatmap_alpha,
                )
                _write_image(annotated_path, annotated)
                saved_heatmap = ""
                if save_heatmap:
                    heatmap = self.detector.colorize_heatmap(result.anomaly_map)
                    _write_image(heatmap_path, heatmap)
                    saved_heatmap = str(heatmap_path)

                rx, ry, rw, rh = result.roi
                weights = self.detector.weight_paths
                records.append(BatchDetectionRecord(
                    input_path=str(image_path),
                    annotated_path=str(annotated_path),
                    heatmap_path=saved_heatmap,
                    label=result.label,
                    is_anomaly=result.is_anomaly,
                    score=result.score,
                    threshold=result.threshold,
                    st_map_weight=result.st_map_weight,
                    ae_map_weight=result.ae_map_weight,
                    inference_time_ms=result.inference_time_ms,
                    processing_time_ms=result.processing_time_ms,
                    total_time_ms=(time.perf_counter() - item_start) * 1000,
                    roi_x=rx,
                    roi_y=ry,
                    roi_width=rw,
                    roi_height=rh,
                    masks_json=json.dumps(
                        getattr(self.detector, "masks", [])
                    ),
                    student_weight=weights["student"],
                    autoencoder_weight=weights["autoencoder"],
                    normalization=weights["normalization"],
                ))
                print(
                    f"[{index}/{len(images)}] {result.label:<7} "
                    f"score={result.score:.6f} "
                    f"inference={result.inference_time_ms:.1f}ms "
                    f"{image_path}"
                )
            except Exception as error:
                record = BatchDetectionRecord(
                    input_path=str(image_path),
                    annotated_path=str(annotated_path),
                    heatmap_path=str(heatmap_path) if save_heatmap else "",
                    ae_map_weight=float(
                        getattr(
                            self.detector,
                            "ae_weight",
                            DEFAULT_AE_WEIGHT,
                        )
                    ),
                    masks_json=json.dumps(
                        getattr(self.detector, "masks", [])
                    ),
                    total_time_ms=(time.perf_counter() - item_start) * 1000,
                    student_weight=self.detector.weight_paths["student"],
                    autoencoder_weight=self.detector.weight_paths["autoencoder"],
                    normalization=self.detector.weight_paths["normalization"],
                    error=f"{type(error).__name__}: {error}",
                )
                records.append(record)
                print(
                    f"[{index}/{len(images)}] ERROR   "
                    f"{image_path}: {record.error}",
                    file=sys.stderr,
                )
                if fail_fast:
                    summary = self._summary(
                        source, output_root, records, batch_start
                    )
                    self._write_reports(summary)
                    raise

        summary = self._summary(source, output_root, records, batch_start)
        self._write_reports(summary)
        return summary

    @staticmethod
    def _summary(
        source: Path,
        output_root: Path,
        records: list[BatchDetectionRecord],
        batch_start: float,
    ) -> BatchSummary:
        return BatchSummary(
            input_path=str(source),
            output_dir=str(output_root),
            records=records,
            elapsed_ms=(time.perf_counter() - batch_start) * 1000,
        )

    @staticmethod
    def _write_reports(summary: BatchSummary) -> None:
        output_root = Path(summary.output_dir)
        csv_path = output_root / "results.csv"
        json_path = output_root / "results.json"
        rows = [asdict(record) for record in summary.records]

        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(BatchDetectionRecord.__dataclass_fields__),
            )
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "input_path": summary.input_path,
            "output_dir": summary.output_dir,
            "elapsed_ms": summary.elapsed_ms,
            "total": len(summary.records),
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "anomaly_count": summary.anomaly_count,
            "normal_count": summary.normal_count,
            "records": rows,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def _discover_images(
    source: Path,
    *,
    recursive: bool,
    excluded_root: Optional[Path] = None,
) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []

    iterator = source.rglob("*") if recursive else source.iterdir()
    images = []
    for candidate in iterator:
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = candidate.resolve()
        if excluded_root is not None and _is_relative_to(resolved, excluded_root):
            continue
        images.append(resolved)
    return sorted(images, key=lambda path: str(path).lower())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the image")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"OpenCV could not encode output: {path}")
    encoded.tofile(path)


def _parse_roi(value: str) -> Tuple[int, int, int, int]:
    try:
        values = tuple(map(int, value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ROI must contain four integers: x,y,width,height"
        ) from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "ROI must contain four integers: x,y,width,height"
        )
    EfficientADDetector._validate_roi(values)
    return values


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch EfficientAD detector with ROI boxes, difference heatmaps, "
            "scores, class judgments, timings, and artifact metadata"
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Input image or directory",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output root for annotated images, heatmaps, CSV, and JSON",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="Model output directory")
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--product", default="my_product")
    parser.add_argument("--student-weight", default=None,
                        help="Override student_final.pth")
    parser.add_argument("--autoencoder-weight", default=None,
                        help="Override autoencoder_final.pth")
    parser.add_argument("--norm-cache", default=None,
                        help="Override norm_params.json")
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--roi", type=_parse_roi, default=None,
                        help="ROI as x,y,w,h (e.g. 100,50,400,300)")
    roi_group.add_argument("--roi-config", default=None,
                        help="ROI JSON config file path")
    parser.add_argument(
        "--mask",
        type=_parse_roi,
        action="append",
        default=None,
        metavar="X,Y,W,H",
        help=(
            "ROI-relative excluded rectangle; repeat --mask for multiple masks"
        ),
    )
    parser.add_argument(
        "--save-roi-config",
        default=None,
        help="Save the active ROI and masks to this JSON file",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Anomaly score threshold")
    parser.add_argument(
        "--ae-weight",
        type=float,
        default=DEFAULT_AE_WEIGHT,
        help=(
            "Weight applied to the normalized AE difference map "
            f"(default: {DEFAULT_AE_WEIGHT}, paired with threshold "
            f"{DEFAULT_THRESHOLD})"
        ),
    )
    parser.add_argument("--train-dir", default=None,
                        help="Training images for norm_params (first run only)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--heatmap-alpha", type=float, default=0.45)
    parser.add_argument("--non-recursive", action="store_true",
                        help="Only process images directly inside input")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Do not overlay the heatmap on annotated images")
    parser.add_argument("--no-heatmap-file", action="store_true",
                        help="Do not save standalone heatmap PNG files")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on the first unreadable or invalid image")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.heatmap_alpha <= 1:
        parser.error("--heatmap-alpha must be between 0 and 1")

    try:
        detector = EfficientADDetector(
            model_dir=args.model_dir,
            roi=args.roi,
            roi_config=args.roi_config,
            masks=args.mask,
            threshold=args.threshold,
            ae_weight=args.ae_weight,
            train_dir=args.train_dir,
            device=args.device,
            dataset=args.dataset,
            product=args.product,
            student_weight=args.student_weight,
            autoencoder_weight=args.autoencoder_weight,
            norm_cache=args.norm_cache,
        )
        if args.save_roi_config:
            saved_config = detector.save_roi_configuration(
                args.save_roi_config
            )
            print(f"[Detector] Saved ROI config: {saved_config}")
        summary = BatchDetector(detector).process(
            args.input,
            args.output_dir,
            recursive=not args.non_recursive,
            draw_heatmap=not args.no_overlay,
            save_heatmap=not args.no_heatmap_file,
            heatmap_alpha=args.heatmap_alpha,
            fail_fast=args.fail_fast,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("\nBatch summary")
    print(f"  total:      {len(summary.records)}")
    print(f"  succeeded:  {summary.succeeded}")
    print(f"  failed:     {summary.failed}")
    print(f"  anomaly:    {summary.anomaly_count}")
    print(f"  normal:     {summary.normal_count}")
    print(f"  elapsed:    {summary.elapsed_ms:.1f} ms")
    print(f"  reports:    {Path(summary.output_dir) / 'results.csv'}")
    print(f"              {Path(summary.output_dir) / 'results.json'}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
