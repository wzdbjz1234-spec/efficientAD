"""双ROI双模型推理流水线(模型30 + 模型31,teacher-free 批量化)。

ROI/模型对应关系(坐标写死在 SPECS 中):
    ROI-30: (1418, 564, 173, 196) 右侧ROI -> 模型30(mydataset/111)
    ROI-31: (150, 543, 156, 155)   左侧ROI -> 模型31(mydataset/222)

推理只使用 Student + Autoencoder(完全跳过 Teacher 前向)。当前模型的
combined 异常分数 = 0*ST图 + AE图,即完全由 AE 差异图决定,因此去掉
Teacher 后分数与原来完全一致,已有阈值可直接沿用。

阈值由人工设定(--threshold-30 / --threshold-31),默认:
    ROI-30: 0.2   (用户指定)
    ROI-31: 0.912657 (此前按 mydataset/222/train/good 正常分最高值+margin 计算)

批量化(--batch-size):每个批次一次性读入多张原图,每个ROI的裁剪图
合并成一个 tensor 做一次前向,大幅减少调用开销。

设备选择(--device):auto/cpu/cuda,决定模型加载与推理所在设备;
推理耗时按每张图片(含每个 ROI)记录到 results.json/csv 的
inference_ms 字段,JSON 顶部另存 device 与 avg/min/max 汇总。

保存策略(--save):
    all            保存所有叠加图;
    misclassified  只保存误判图(按输入路径中的 good/broken 判定真实类别);
                   无法判定真实类别的图放入 unknown/ 目录;
    未指定时在交互式终端询问用户。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from efficientad_tools import ModelArtifacts
from efficientad_tools.predictor import NormalizationParams
from efficientad_tools.images import discover_images
from roi_mask import apply_masks

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dual_detector"
DEFAULT_IMAGE_SIZE = 256
HEATMAP_ALPHA = 0.45
NORMAL_CLASS_NAMES = {"good"}
ANOMALY_CLASS_NAMES = {"broken"}

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass(frozen=True)
class ROISpec:
    name: str
    model: str
    product: str
    roi: tuple[int, int, int, int]
    masks: tuple[tuple[int, int, int, int], ...] = ()


SPECS = (
    ROISpec(
        name="ROI-30",
        model="30",
        product="111",
        roi=(1418, 564, 173, 196),
    ),
    ROISpec(
        name="ROI-31",
        model="31",
        product="222",
        roi=(150, 543, 156, 155),
    ),
)

DEFAULT_THRESHOLDS = {"ROI-30": 0.2, "ROI-31": 0.912657}


@dataclass
class _Unit:
    spec: ROISpec
    student: torch.nn.Module
    autoencoder: torch.nn.Module
    q_ae_start: torch.Tensor
    q_ae_end: torch.Tensor
    threshold: float
    roi: tuple[int, int, int, int]
    masks: list[tuple[int, int, int, int]]


@dataclass
class ROIInspection:
    name: str
    roi: tuple[int, int, int, int]
    score: float
    threshold: float
    is_anomaly: bool
    label: str
    anomaly_map: np.ndarray
    inference_ms: float = 0.0


@dataclass
class ImageInspection:
    input_path: str
    truth: Optional[int]
    truth_label: str
    predicted: int
    predicted_label: str
    is_misclassified: Optional[bool]
    rois: list[ROIInspection]
    annotated: np.ndarray
    inference_ms: float = 0.0

    def as_record(self, annotated_path: str = "") -> dict[str, Any]:
        row: dict[str, Any] = {
            "input_path": self.input_path,
            "truth_label": self.truth_label,
            "predicted_label": self.predicted_label,
            "is_misclassified": (
                "" if self.is_misclassified is None else bool(self.is_misclassified)
            ),
            "annotated_path": annotated_path,
            "inference_ms": round(self.inference_ms, 3),
        }
        for roi in self.rois:
            row[f"{roi.name}_label"] = roi.label
            row[f"{roi.name}_score"] = round(roi.score, 6)
            row[f"{roi.name}_threshold"] = round(roi.threshold, 6)
            row[f"{roi.name}_ms"] = round(roi.inference_ms, 3)
        return row


class DualROIDetector:
    """Teacher-free 双ROI检测器:只加载 Student + Autoencoder,支持批量前向。"""

    def __init__(
        self,
        *,
        device: str = "auto",
        batch_size: int = 16,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = _resolve_device(device)
        active = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            active.update({name: float(value) for name, value in thresholds.items() if value is not None})
        self.units: list[_Unit] = []
        for spec in SPECS:
            artifacts = ModelArtifacts.from_output(
                spec.model,
                dataset="mvtec_ad",
                product=spec.product,
                source_dir=PROJECT_ROOT,
            )
            if not artifacts.norm_params.is_file():
                raise FileNotFoundError(
                    f"缺少归一化参数缓存: {artifacts.norm_params}"
                )
            normalization = NormalizationParams.load(artifacts.norm_params, self.device)
            student = torch.load(
                artifacts.student, map_location=self.device, weights_only=False
            )
            autoencoder = torch.load(
                artifacts.autoencoder, map_location=self.device, weights_only=False
            )
            student.to(self.device).eval()
            autoencoder.to(self.device).eval()
            roi, masks = spec.roi, list(spec.masks)
            threshold = active[spec.name]
            self.units.append(
                _Unit(
                    spec=spec,
                    student=student,
                    autoencoder=autoencoder,
                    q_ae_start=normalization.q_ae_start,
                    q_ae_end=normalization.q_ae_end,
                    threshold=threshold,
                    roi=roi,
                    masks=masks,
                )
            )
            print(
                f"[{spec.name}] 模型{spec.model} (teacher-free: student+autoencoder) "
                f"阈值={threshold:.6f} ROI={roi}"
            )

    def inspect(self, image_path: str | Path) -> ImageInspection:
        results, failed = self.inspect_batch([image_path], batch_size=1)
        if not results:
            message = failed[0][1] if failed else "未知错误"
            raise ValueError(f"无法处理图片 {image_path}: {message}")
        return results[str(image_path)]

    def inspect_batch(
        self,
        paths: list[str | Path],
        *,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, ImageInspection], list[tuple[str, str]]]:
        """批量推理:返回 (path->ImageInspection, [(失败path, 错误信息)])。"""
        size = self.batch_size if batch_size is None else int(batch_size)
        results: dict[str, ImageInspection] = {}
        failed: list[tuple[str, str]] = []

        for chunk in _chunks(list(paths), size):
            valid: list[tuple[Path, np.ndarray]] = []
            for path in chunk:
                try:
                    image = _read_image(path)
                except ValueError as error:
                    failed.append((str(path), str(error)))
                    continue
                out_of_bounds = None
                for unit in self.units:
                    x, y, w, h = unit.roi
                    height, width = image.shape[:2]
                    if x < 0 or y < 0 or x + w > width or y + h > height:
                        out_of_bounds = (
                            unit.spec.name,
                            f"ROI {unit.roi} 超出图片 {width}x{height}",
                        )
                        break
                if out_of_bounds is None:
                    valid.append((path, image))
                else:
                    failed.append((str(path), out_of_bounds[1]))

            if not valid:
                continue

            pending: dict[str, list[ROIInspection]] = {
                str(path): [] for path, _ in valid
            }
            for unit in self.units:
                x, y, w, h = unit.roi
                entries: list[tuple[str, torch.Tensor]] = []
                for path, image in valid:
                    crop = apply_masks(image[y : y + h, x : x + w].copy(), unit.masks)
                    tensor = _DEFAULT_TRANSFORM(
                        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    )
                    entries.append((str(path), tensor))
                tensors = torch.stack([tensor for _, tensor in entries]).to(self.device)
                self._synchronize_device()
                forward_start = time.perf_counter()
                maps = self._forward(unit, tensors)
                self._synchronize_device()
                forward_ms = (time.perf_counter() - forward_start) * 1000
                per_image_ms = forward_ms / len(entries)

                for (path, _tensor), anomaly_map in zip(entries, maps):
                    score = float(np.max(anomaly_map))
                    is_anomaly = score >= unit.threshold
                    pending[path].append(
                        ROIInspection(
                            name=unit.spec.name,
                            roi=unit.roi,
                            score=score,
                            threshold=unit.threshold,
                            is_anomaly=is_anomaly,
                            label="ANOMALY" if is_anomaly else "NORMAL",
                            anomaly_map=anomaly_map,
                            inference_ms=per_image_ms,
                        )
                    )

            for path, image in valid:
                key = str(path)
                rois = pending[key]
                predicted = 1 if any(r.is_anomaly for r in rois) else 0
                truth = ground_truth_from_path(path)
                truth_label = (
                    "NORMAL" if truth == 0 else "ANOMALY" if truth == 1 else "UNKNOWN"
                )
                predicted_label = "ANOMALY" if predicted else "NORMAL"
                is_misclassified = None if truth is None else (predicted != truth)
                results[key] = ImageInspection(
                    input_path=key,
                    truth=truth,
                    truth_label=truth_label,
                    predicted=predicted,
                    predicted_label=predicted_label,
                    is_misclassified=is_misclassified,
                    rois=rois,
                    annotated=annotate(image, rois),
                    inference_ms=sum(r.inference_ms for r in rois),
                )

        return results, failed

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _forward(
        self,
        unit: _Unit,
        tensors: torch.Tensor,
    ) -> np.ndarray:
        """单ROI批量前向:Student+AE 差异 -> 归一化 AE 热力图,尺寸对齐ROI。"""
        student_out = unit.student(tensors)
        autoencoder_out = unit.autoencoder(tensors)
        ae_channels = autoencoder_out.shape[1]
        diff = (autoencoder_out - student_out[:, -ae_channels:]) ** 2
        map_ae = torch.mean(diff, dim=1, keepdim=True)
        denominator = unit.q_ae_end - unit.q_ae_start
        if torch.isclose(denominator, torch.zeros_like(denominator)):
            map_ae = torch.zeros_like(map_ae)
        else:
            map_ae = 0.1 * (map_ae - unit.q_ae_start) / denominator

        _x, _y, w, h = unit.roi
        padded = F.pad(map_ae, (4, 4, 4, 4))
        resized = F.interpolate(
            padded,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return resized[:, 0].detach().cpu().numpy()

    def threshold_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": unit.spec.name,
                "model": unit.spec.model,
                "product": unit.spec.product,
                "threshold": unit.threshold,
                "roi": list(map(int, unit.roi)),
                "masks": [list(map(int, m)) for m in unit.masks],
            }
            for unit in self.units
        ]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _chunks(sequence, size: int):
    for index in range(0, len(sequence), size):
        yield sequence[index : index + size]


def ground_truth_from_path(path: Path) -> Optional[int]:
    """按路径中的 good/broken 目录判定真实类别(0正常/1异常/None未知)。"""
    parts = {part.lower() for part in path.parts}
    normal = parts & NORMAL_CLASS_NAMES
    anomaly = parts & ANOMALY_CLASS_NAMES
    if normal and not anomaly:
        return 0
    if anomaly and not normal:
        return 1
    return None


def annotate(image: np.ndarray, inspections: list[ROIInspection]) -> np.ndarray:
    """叠加热力图、框出ROI并标记类别/得分/阈值,顶部显示整体判定。"""
    output = image.copy()
    is_anomaly = any(r.is_anomaly for r in inspections)

    for result in inspections:
        x, y, w, h = result.roi
        color = (0, 0, 255) if result.is_anomaly else (0, 200, 0)

        amap = result.anomaly_map
        if amap.shape != (h, w):
            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        normalized = np.clip(amap / max(result.threshold, 1e-12), 0.0, 1.0)
        heatmap = cv2.applyColorMap(
            np.clip(normalized * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        region = output[y : y + h, x : x + w]
        output[y : y + h, x : x + w] = cv2.addWeighted(
            region, 1 - HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0
        )

        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), color, 2)
        text = (
            f"{result.name} {result.label} "
            f"score={result.score:.4f} thr={result.threshold:.4f}"
        )
        _put_text_box(output, text, (x, y - 12), color)

    verdict = "ANOMALY" if is_anomaly else "NORMAL"
    summary = f"VERDICT {verdict} | " + " | ".join(
        f"{r.name} {r.score:.4f}" for r in inspections
    )
    _draw_top_bar(output, summary, (0, 0, 255) if is_anomaly else (0, 200, 0))
    return output


def _put_text_box(image, text, org, color):
    x, y = org
    y = max(y, 20)
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(image, (x, y - th - 6), (x + tw, y + 2), color, -1)
    cv2.putText(
        image, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def _draw_top_bar(image, text, color):
    height, width = image.shape[:2]
    bar_h = min(48, height)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.putText(
        image, text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
    )


def save_destination(mode: str, inspection: ImageInspection, relative: Path, output_dir: Path) -> Optional[Path]:
    """按保存策略决定叠加图的输出路径;None 表示不保存。"""
    if mode == "all":
        return output_dir / "annotated" / relative
    if inspection.is_misclassified is True:
        return output_dir / "misclassified" / relative
    if inspection.is_misclassified is None:
        return output_dir / "unknown" / relative
    return None


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV 无法解码图片: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"OpenCV 无法编码输出: {path}")
    encoded.tofile(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual_detector",
        description=(
            "双ROI双模型推理(teacher-free):模型30(右侧ROI)+模型31(左侧ROI) "
            "同时对整幅原图检测,输出叠加热力图与判定标注"
        ),
    )
    parser.add_argument("--input", required=True, help="原始图像文件或目录")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录(默认 {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--save",
        choices=("all", "misclassified"),
        default=None,
        help=(
            "保存策略:all=保存所有叠加图;misclassified=只保存误判图"
            "(无法判定真实类别的图保存到 unknown/);未指定时交互询问"
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="每个批次处理的图片数,ROI裁剪图合并为 batch 一次前向(默认 16)",
    )
    parser.add_argument(
        "--threshold-30",
        type=float,
        default=DEFAULT_THRESHOLDS["ROI-30"],
        help=f"ROI-30(模型30)判定阈值(默认 {DEFAULT_THRESHOLDS['ROI-30']})",
    )
    parser.add_argument(
        "--threshold-31",
        type=float,
        default=DEFAULT_THRESHOLDS["ROI-31"],
        help=f"ROI-31(模型31)判定阈值(默认 {DEFAULT_THRESHOLDS['ROI-31']})",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="输入为目录时仅处理第一层图片",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")

    mode = args.save
    if mode is None:
        if sys.stdin.isatty():
            answer = input(
                "保存策略:输入 all 保存所有叠加图,输入 misclassified 只保存误判图: "
            ).strip().lower()
            mode = "all" if answer.startswith("a") else "misclassified"
        else:
            mode = "all"
            print("[dual] 未指定 --save 且非交互终端,默认保存所有叠加图")

    output_dir = Path(args.output_dir).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    images = discover_images(input_path, recursive=not args.non_recursive)
    images = [
        path for path in images
        if not _is_relative_to(path.resolve(), output_dir)
    ]
    if not images:
        print(f"error: 未在 {input_path} 找到图片")
        return 2

    detector = DualROIDetector(
        device=args.device,
        batch_size=args.batch_size,
        thresholds={"ROI-30": args.threshold_30, "ROI-31": args.threshold_31},
    )

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    saved_count = misclassified_count = failed_count = 0
    print(f"\n[dual] 处理 {len(images)} 张图片 -> {output_dir} "
          f"(batch_size={args.batch_size}, 保存策略: {mode})\n")
    total = len(images)
    index = 0
    for chunk in _chunks(images, args.batch_size):
        results, failed = detector.inspect_batch(chunk, batch_size=args.batch_size)
        for path, message in failed:
            failed_count += 1
            index += 1
            print(f"[{index}/{total}] ERROR {path}: {message}", file=sys.stderr)
        for image_path, inspection in results.items():
            index += 1
            relative = (
                Path(image_path).name
                if input_path.is_file()
                else Path(image_path).relative_to(input_path)
            )
            destination = save_destination(mode, inspection, relative, output_dir)
            annotated_path = ""
            if destination is not None:
                annotated_path = str(destination.with_suffix(".png"))
                _write_image(Path(annotated_path), inspection.annotated)
                saved_count += 1
            if inspection.is_misclassified is True:
                misclassified_count += 1

            rows.append(inspection.as_record(annotated_path=annotated_path))
            print(
                f"[{index}/{total}] {inspection.predicted_label:<7} "
                f"truth={inspection.truth_label:<8} "
                + " | ".join(
                    f"{r.name} {r.label} {r.score:.4f} (thr={r.threshold:.4f})"
                    for r in inspection.rois
                )
                + f"  {image_path}"
            )

    elapsed = (time.perf_counter() - start) * 1000
    _write_reports(
        output_dir, rows, detector.threshold_summary(), args, elapsed,
        device=str(detector.device),
    )

    print("\n[dual] 汇总")
    print(f"  device:   {detector.device}")
    for info in detector.threshold_summary():
        print(
            f"  {info['name']}: 模型{info['model']} 阈值={info['threshold']:.6f} "
            f"ROI={info['roi']}"
        )
    print(f"  total:    {len(images)}")
    print(f"  saved:    {saved_count}")
    print(f"  failed:   {failed_count}")
    print(f"  misclassified: {misclassified_count}")
    print(f"  elapsed:  {elapsed:.1f} ms")
    print(f"  reports:  {output_dir / 'results.csv'}")
    print(f"            {output_dir / 'results.json'}")
    return 1 if failed_count else 0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    args: argparse.Namespace,
    elapsed_ms: float,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "input_path", "truth_label", "predicted_label",
            "is_misclassified", "annotated_path", "inference_ms",
        ]
    )
    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    inference_times = [row["inference_ms"] for row in rows]
    payload = {
        "input_path": str(Path(args.input).expanduser().resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "save_mode": args.save,
        "thresholds": thresholds,
        "elapsed_ms": elapsed_ms,
        "total": len(rows),
        "saved": sum(1 for row in rows if row["annotated_path"]),
        "misclassified": sum(
            1 for row in rows if row["is_misclassified"] is True
        ),
        "avg_inference_ms": (
            round(sum(inference_times) / len(inference_times), 3)
            if inference_times else 0.0
        ),
        "min_inference_ms": round(min(inference_times), 3) if inference_times else 0.0,
        "max_inference_ms": round(max(inference_times), 3) if inference_times else 0.0,
        "records": rows,
    }
    with (output_dir / "results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
