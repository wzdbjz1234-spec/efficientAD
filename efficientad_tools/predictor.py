from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .artifacts import DEFAULT_SOURCE_DIR, ModelArtifacts
from .images import discover_images


DEFAULT_IMAGE_SIZE = 256

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass(frozen=True)
class FeatureAnalysis:
    teacher: np.ndarray
    student_teacher: np.ndarray
    autoencoder: np.ndarray
    student_autoencoder: np.ndarray
    student_teacher_diff: np.ndarray
    autoencoder_diff: np.ndarray


@dataclass(frozen=True)
class Prediction:
    score: float
    student_teacher_score: float
    autoencoder_score: float
    anomaly_map: np.ndarray
    student_teacher_map: np.ndarray
    autoencoder_map: np.ndarray
    features: FeatureAnalysis | None = None

    def is_anomaly(self, threshold: float) -> bool:
        return self.score >= threshold


@dataclass(frozen=True)
class NormalizationParams:
    teacher_mean: torch.Tensor
    teacher_std: torch.Tensor
    q_st_start: torch.Tensor
    q_st_end: torch.Tensor
    q_ae_start: torch.Tensor
    q_ae_end: torch.Tensor

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device
    ) -> "NormalizationParams | None":
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)

        required = ("teacher_mean", "teacher_std", "q_st_start", "q_st_end", "q_ae_start", "q_ae_end")
        if any(key not in data for key in required):
            return None

        def tensor(value: Any) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        return cls(
            teacher_mean=tensor(data["teacher_mean"]).reshape(1, -1, 1, 1),
            teacher_std=tensor(data["teacher_std"]).reshape(1, -1, 1, 1),
            q_st_start=tensor(data["q_st_start"]),
            q_st_end=tensor(data["q_st_end"]),
            q_ae_start=tensor(data["q_ae_start"]),
            q_ae_end=tensor(data["q_ae_end"]),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "teacher_mean": self.teacher_mean.detach().cpu().flatten().tolist(),
            "teacher_std": self.teacher_std.detach().cpu().flatten().tolist(),
            "q_st_start": float(self.q_st_start.detach().cpu()),
            "q_st_end": float(self.q_st_end.detach().cpu()),
            "q_ae_start": float(self.q_ae_start.detach().cpu()),
            "q_ae_end": float(self.q_ae_end.detach().cpu()),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return _DEFAULT_TRANSFORM(image.convert("RGB"))


class EfficientADPredictor:
    """Load one training output and expose a single-image prediction interface."""

    def __init__(
        self,
        *,
        artifacts: ModelArtifacts,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        autoencoder: torch.nn.Module,
        normalization: NormalizationParams,
        device: torch.device,
        valid_input_mask: torch.Tensor | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.normalization = normalization
        self.device = device
        self.valid_input_mask = valid_input_mask

    @classmethod
    def from_output(
        cls,
        model: str | Path,
        *,
        dataset: str = "mvtec_ad",
        product: str = "my_product",
        source_dir: str | Path = DEFAULT_SOURCE_DIR,
        device: str | torch.device = "auto",
        train_dir: str | Path | None = None,
    ) -> "EfficientADPredictor":
        artifacts = ModelArtifacts.from_output(
            model,
            dataset=dataset,
            product=product,
            source_dir=source_dir,
        )
        return cls.load(artifacts, device=device, train_dir=train_dir)

    @classmethod
    def load(
        cls,
        artifacts: ModelArtifacts,
        *,
        device: str | torch.device = "auto",
        train_dir: str | Path | None = None,
    ) -> "EfficientADPredictor":
        resolved_device = _resolve_device(device)
        artifacts.validate(require_norm=train_dir is None)

        source_text = str(artifacts.source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        teacher = torch.load(
            artifacts.teacher, map_location=resolved_device, weights_only=False
        )
        student = torch.load(
            artifacts.student, map_location=resolved_device, weights_only=False
        )
        autoencoder = torch.load(
            artifacts.autoencoder, map_location=resolved_device, weights_only=False
        )
        for model in (teacher, student, autoencoder):
            model.to(resolved_device)
            model.eval()

        valid_input_mask = _load_valid_input_mask(
            artifacts.model_dir / "mask_config.json", resolved_device
        )

        normalization = None
        if artifacts.norm_params.is_file():
            normalization = NormalizationParams.load(
                artifacts.norm_params, resolved_device
            )
        if normalization is None:
            if train_dir is not None:
                normalization = _compute_normalization(
                    teacher,
                    student,
                    autoencoder,
                    Path(train_dir),
                    resolved_device,
                    valid_input_mask,
                )
                normalization.save(artifacts.norm_params)
            else:
                raise FileNotFoundError(
                    f"Normalization cache not found: {artifacts.norm_params}. "
                    "Pass train_dir to compute it."
                )

        return cls(
            artifacts=artifacts,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            normalization=normalization,
            device=resolved_device,
            valid_input_mask=valid_input_mask,
        )

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        include_features: bool = False,
    ) -> Prediction:
        rgb, original_size = _read_rgb(image)
        tensor = _DEFAULT_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self.valid_input_mask is not None:
            tensor = tensor * self.valid_input_mask
        norm = self.normalization

        teacher_output = (
            self.teacher(tensor) - norm.teacher_mean
        ) / norm.teacher_std
        student_output = self.student(tensor)
        autoencoder_output = self.autoencoder(tensor)

        teacher_channels = teacher_output.shape[1]
        ae_channels = autoencoder_output.shape[1]
        if student_output.shape[1] < teacher_channels + ae_channels:
            raise ValueError(
                "Student output has fewer channels than teacher + autoencoder: "
                f"{student_output.shape[1]} < {teacher_channels + ae_channels}"
            )

        student_teacher = student_output[:, :teacher_channels]
        student_autoencoder = student_output[
            :, teacher_channels : teacher_channels + ae_channels
        ]
        diff_st = (teacher_output - student_teacher) ** 2
        diff_ae = (autoencoder_output - student_autoencoder) ** 2
        map_st = torch.mean(diff_st, dim=1, keepdim=True)
        map_ae = torch.mean(diff_ae, dim=1, keepdim=True)
        map_st = _normalize_map(map_st, norm.q_st_start, norm.q_st_end)
        map_ae = _normalize_map(map_ae, norm.q_ae_start, norm.q_ae_end)
        feature_mask = _feature_valid_mask(self.valid_input_mask, map_st)
        if feature_mask is not None:
            map_st = map_st * feature_mask
            map_ae = map_ae * feature_mask
        map_combined = 0 * map_st + map_ae

        height, width = original_size
        combined_np = _resize_map(map_combined, height, width)
        st_np = _resize_map(map_st, height, width)
        ae_np = _resize_map(map_ae, height, width)
        if self.valid_input_mask is not None:
            output_mask = F.interpolate(
                self.valid_input_mask,
                size=(height, width),
                mode="nearest",
            )[0, 0].detach().cpu().numpy()
            combined_np *= output_mask
            st_np *= output_mask
            ae_np *= output_mask

        features = None
        if include_features:
            features = FeatureAnalysis(
                teacher=_channels_to_numpy(teacher_output),
                student_teacher=_channels_to_numpy(student_teacher),
                autoencoder=_channels_to_numpy(autoencoder_output),
                student_autoencoder=_channels_to_numpy(student_autoencoder),
                student_teacher_diff=_channels_to_numpy(diff_st),
                autoencoder_diff=_channels_to_numpy(diff_ae),
            )

        return Prediction(
            score=float(np.max(combined_np)),
            student_teacher_score=float(np.max(st_np)),
            autoencoder_score=float(np.max(ae_np)),
            anomaly_map=combined_np,
            student_teacher_map=st_np,
            autoencoder_map=ae_np,
            features=features,
        )


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _read_rgb(
    image: str | Path | Image.Image | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            rgb = np.asarray(opened.convert("RGB"))
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy images must have shape [height, width, 3]")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    return rgb, rgb.shape[:2]


def _normalize_map(
    value: torch.Tensor, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    denominator = end - start
    if torch.isclose(denominator, torch.zeros_like(denominator)):
        return torch.zeros_like(value)
    return 0.1 * (value - start) / denominator


def _resize_map(value: torch.Tensor, height: int, width: int) -> np.ndarray:
    padded = F.pad(value, (4, 4, 4, 4))
    resized = F.interpolate(
        padded,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].detach().cpu().numpy()


def _channels_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value[0].detach().cpu().numpy()


def _load_valid_input_mask(
    path: Path, device: torch.device
) -> torch.Tensor | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    _, _, roi_width, roi_height = map(int, data["roi"])
    raw_masks = data.get("masks")
    if raw_masks is None:
        raw_mask = data.get("mask")
        raw_masks = [] if raw_mask is None else [raw_mask]
    if not raw_masks:
        return None

    valid = torch.ones((1, 1, roi_height, roi_width), dtype=torch.float32)
    for raw_mask in raw_masks:
        x, y, width, height = map(int, raw_mask)
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > roi_width
            or y + height > roi_height
        ):
            raise ValueError(
                f"Mask ({x},{y},{width},{height}) is outside "
                f"ROI {roi_width}x{roi_height}: {path}"
            )
        valid[:, :, y : y + height, x : x + width] = 0
    valid = F.interpolate(
        valid,
        size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        mode="nearest",
    )
    if not torch.any(valid):
        raise ValueError(f"Mask excludes the entire ROI: {path}")
    return valid.to(device)


def _feature_valid_mask(
    valid_input_mask: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor | None:
    if valid_input_mask is None:
        return None
    return F.interpolate(
        valid_input_mask,
        size=reference.shape[-2:],
        mode="nearest",
    ).bool()


@torch.no_grad()
def _compute_normalization(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    autoencoder: torch.nn.Module,
    train_dir: Path,
    device: torch.device,
    valid_input_mask: torch.Tensor | None = None,
) -> NormalizationParams:
    paths = discover_images(train_dir, recursive=True)
    if not paths:
        raise ValueError(f"No training images found under {train_dir}")
    loader = DataLoader(_ImageDataset(paths), batch_size=1, shuffle=False, num_workers=0)

    means = []
    for images in tqdm(loader, desc="Teacher mean"):
        images = images.to(device)
        if valid_input_mask is not None:
            images = images * valid_input_mask
        output = teacher(images)
        feature_mask = _feature_valid_mask(valid_input_mask, output)
        if feature_mask is None:
            mean = torch.mean(output, dim=(0, 2, 3))
        else:
            count = feature_mask.sum() * output.shape[0]
            mean = (output * feature_mask).sum(dim=(0, 2, 3)) / count
        means.append(mean.cpu())
    teacher_mean = torch.mean(torch.stack(means), dim=0).reshape(1, -1, 1, 1)
    teacher_mean = teacher_mean.to(device)

    variances = []
    for images in tqdm(loader, desc="Teacher std"):
        images = images.to(device)
        if valid_input_mask is not None:
            images = images * valid_input_mask
        output = teacher(images)
        distance = (output - teacher_mean) ** 2
        feature_mask = _feature_valid_mask(valid_input_mask, distance)
        if feature_mask is None:
            variance = torch.mean(distance, dim=(0, 2, 3))
        else:
            count = feature_mask.sum() * distance.shape[0]
            variance = (distance * feature_mask).sum(dim=(0, 2, 3)) / count
        variances.append(variance.cpu())
    teacher_std = torch.sqrt(torch.mean(torch.stack(variances), dim=0))
    teacher_std = teacher_std.reshape(1, -1, 1, 1).to(device)

    maps_st = []
    maps_ae = []
    for images in tqdm(loader, desc="Map quantiles"):
        images = images.to(device)
        if valid_input_mask is not None:
            images = images * valid_input_mask
        teacher_output = (teacher(images) - teacher_mean) / teacher_std
        student_output = student(images)
        autoencoder_output = autoencoder(images)
        teacher_channels = teacher_output.shape[1]
        ae_channels = autoencoder_output.shape[1]
        map_st = torch.mean(
            (teacher_output - student_output[:, :teacher_channels]) ** 2,
            dim=1,
            keepdim=True,
        )
        map_ae = torch.mean(
            (
                autoencoder_output
                - student_output[
                    :, teacher_channels : teacher_channels + ae_channels
                ]
            )
            ** 2,
            dim=1,
            keepdim=True,
        )
        feature_mask = _feature_valid_mask(valid_input_mask, map_st)
        if feature_mask is None:
            maps_st.append(map_st.flatten().cpu())
            maps_ae.append(map_ae.flatten().cpu())
        else:
            maps_st.append(map_st.masked_select(feature_mask).cpu())
            maps_ae.append(map_ae.masked_select(feature_mask).cpu())

    all_st = torch.cat(maps_st)
    all_ae = torch.cat(maps_ae)
    return NormalizationParams(
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_st_start=torch.quantile(all_st, 0.9).to(device),
        q_st_end=torch.quantile(all_st, 0.995).to(device),
        q_ae_start=torch.quantile(all_ae, 0.9).to(device),
        q_ae_end=torch.quantile(all_ae, 0.995).to(device),
    )
