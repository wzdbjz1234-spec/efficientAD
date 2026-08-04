import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from batch_detector import (
    BatchDetector,
    DetectionResult,
    EfficientADDetector,
    _discover_images,
    _parse_roi,
    build_parser,
)


class _FakeDetector:
    weight_paths = {
        "student": "C:/weights/student_final.pth",
        "autoencoder": "C:/weights/autoencoder_final.pth",
        "normalization": "C:/weights/norm_params.json",
    }

    def detect_and_annotate(
        self,
        image,
        *,
        draw_heatmap=True,
        heatmap_alpha=0.45,
        heatmap_max=None,
    ):
        height, width = image.shape[:2]
        anomaly_map = np.full((height, width), 0.25, dtype=np.float32)
        result = DetectionResult(
            is_anomaly=True,
            score=0.25,
            label="ANOMALY",
            inference_time_ms=3.5,
            processing_time_ms=5.0,
            threshold=0.1,
            st_map_weight=0.0,
            ae_map_weight=0.025,
            anomaly_map=anomaly_map,
            roi=(0, 0, width, height),
        )
        annotated = image.copy()
        annotated[:, :, 1] = 255
        return result, annotated

    @staticmethod
    def colorize_heatmap(anomaly_map, *, heatmap_max=None):
        gray = np.clip(anomaly_map * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


class _NormalDetector(_FakeDetector):
    def detect_and_annotate(self, image, **kwargs):
        result, annotated = super().detect_and_annotate(image, **kwargs)
        result.is_anomaly = False
        result.label = "NORMAL"
        result.score = 0.05
        return result, annotated


def _write_test_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode test image")
    encoded.tofile(path)


class BatchDetectorTests(unittest.TestCase):
    @staticmethod
    def _artifact_resolver(
        model_dir: Path,
        model_product: str | None = None,
    ) -> EfficientADDetector:
        detector = EfficientADDetector.__new__(EfficientADDetector)
        detector.model_dir = model_dir
        detector.dataset = "mvtec_ad"
        detector.model_product = model_product
        detector.product = model_product
        return detector

    def test_processes_directory_and_preserves_relative_structure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = input_dir / "results"
            _write_test_image(input_dir / "a.png")
            _write_test_image(input_dir / "nested" / "b.jpg")

            summary = BatchDetector(_FakeDetector()).process(
                input_dir,
                output_dir,
                recursive=True,
            )

            self.assertEqual(summary.succeeded, 2)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.anomaly_count, 2)
            self.assertTrue(
                (output_dir / "annotated" / "a_result.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "annotated"
                    / "nested"
                    / "b_result.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "heatmaps"
                    / "nested"
                    / "b_heatmap.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "anomaly_heatmaps"
                    / "nested"
                    / "b_heatmap.png"
                ).is_file()
            )

            with (output_dir / "results.json").open(encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["total"], 2)
            self.assertEqual(report["failed"], 0)
            self.assertEqual(
                report["records"][0]["student_weight"],
                "C:/weights/student_final.pth",
            )
            self.assertEqual(
                report["records"][0]["inference_time_ms"],
                3.5,
            )
            self.assertTrue(
                Path(
                    report["records"][0]["anomaly_heatmap_path"]
                ).is_file()
            )

            discovered = _discover_images(
                input_dir,
                recursive=True,
                excluded_root=output_dir,
            )
            self.assertEqual(len(discovered), 2)

    def test_records_decode_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            _write_test_image(input_dir / "good.png")
            bad_path = input_dir / "bad.png"
            bad_path.write_bytes(b"not an image")

            summary = BatchDetector(_FakeDetector()).process(
                input_dir,
                output_dir,
            )

            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 1)
            failed = next(record for record in summary.records if record.error)
            self.assertEqual(Path(failed.input_path), bad_path.resolve())
            self.assertIn("could not decode", failed.error)

    def test_normal_heatmap_is_not_saved_in_anomaly_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "normal.png"
            output_dir = root / "output"
            _write_test_image(input_path)

            summary = BatchDetector(_NormalDetector()).process(
                input_path,
                output_dir,
            )

            self.assertEqual(summary.normal_count, 1)
            self.assertTrue(
                (output_dir / "heatmaps" / "normal_heatmap.png").is_file()
            )
            self.assertFalse(
                (
                    output_dir
                    / "anomaly_heatmaps"
                    / "normal_heatmap.png"
                ).exists()
            )
            self.assertEqual(
                summary.records[0].anomaly_heatmap_path,
                "",
            )

    def test_roi_parser_requires_four_positive_size_values(self):
        self.assertEqual(_parse_roi("10,20,30,40"), (10, 20, 30, 40))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_roi("10,20,30")
        with self.assertRaises(ValueError):
            _parse_roi("10,20,0,40")

    def test_repeated_masks_are_parsed_and_can_be_saved(self):
        args = build_parser().parse_args([
            "--input", "images",
            "--output-dir", "results",
            "--roi", "100,200,300,400",
            "--mask", "10,20,30,40",
            "--mask", "50,60,70,80",
            "--save-roi-config", "saved.json",
        ])
        self.assertEqual(args.roi, (100, 200, 300, 400))
        self.assertEqual(
            args.mask,
            [(10, 20, 30, 40), (50, 60, 70, 80)],
        )

        detector = EfficientADDetector.__new__(EfficientADDetector)
        detector.roi = args.roi
        detector.masks = args.mask
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "roi.json"
            detector.save_roi_configuration(target)
            with target.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["roi"], [100, 200, 300, 400])
        self.assertEqual(
            payload["masks"],
            [[10, 20, 30, 40], [50, 60, 70, 80]],
        )

    def test_model_product_cli_name_and_legacy_alias(self):
        common = ["--input", "images", "--output-dir", "results"]
        new_args = build_parser().parse_args(
            [*common, "--model-product", "new_name"]
        )
        legacy_args = build_parser().parse_args(
            [*common, "--product", "legacy_name"]
        )
        self.assertEqual(new_args.model_product, "new_name")
        self.assertEqual(legacy_args.model_product, "legacy_name")

    def test_single_model_product_is_discovered_automatically(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            artifacts = (
                model_dir / "trainings" / "mvtec_ad" / "trained_product"
            )
            artifacts.mkdir(parents=True)
            (artifacts / "student_final.pth").touch()
            detector = self._artifact_resolver(model_dir)

            self.assertEqual(detector._resolve_artifacts_dir(), artifacts)
            self.assertEqual(detector.model_product, "trained_product")

    def test_multiple_model_products_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            dataset_dir = model_dir / "trainings" / "mvtec_ad"
            for product in ("product_a", "product_b"):
                artifacts = dataset_dir / product
                artifacts.mkdir(parents=True)
                (artifacts / "student_final.pth").touch()
            detector = self._artifact_resolver(model_dir)

            with self.assertRaisesRegex(
                ValueError,
                "Multiple model products.*product_a, product_b",
            ):
                detector._resolve_artifacts_dir()

    def test_explicit_model_product_reports_available_products(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            artifacts = (
                model_dir / "trainings" / "mvtec_ad" / "actual_product"
            )
            artifacts.mkdir(parents=True)
            (artifacts / "student_final.pth").touch()
            detector = self._artifact_resolver(
                model_dir,
                model_product="missing_product",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Available model products: actual_product",
            ):
                detector._resolve_artifacts_dir()

    def test_mask_is_excluded_from_output_heatmap(self):
        detector = EfficientADDetector.__new__(EfficientADDetector)
        detector.device = torch.device("cpu")
        detector.threshold = 1.0
        detector.masks = [(10, 5, 20, 15)]
        roi = (100, 200, 60, 40)
        detector.valid_input_mask = detector._build_valid_input_mask(
            roi, detector.masks
        )

        valid = detector.output_valid_mask(40, 60)
        self.assertTrue(np.all(valid[5:20, 10:30] == 0))
        self.assertTrue(np.all(valid[:5, :] == 1))

        anomaly_map = np.linspace(
            0, 1, num=40 * 60, dtype=np.float32
        ).reshape(40, 60)
        heatmap = detector.colorize_heatmap(anomaly_map)
        self.assertTrue(np.all(heatmap[5:20, 10:30] == 0))
        self.assertGreater(int(heatmap[30, 40].sum()), 0)

    def test_heatmaps_use_one_absolute_scale_across_images(self):
        detector = EfficientADDetector.__new__(EfficientADDetector)
        detector.threshold = 1.0
        detector.valid_input_mask = None

        low_score_map = np.full((4, 4), 0.1, dtype=np.float32)
        threshold_map = np.full((4, 4), 1.0, dtype=np.float32)
        low_heatmap = detector.colorize_heatmap(low_score_map)
        threshold_heatmap = detector.colorize_heatmap(threshold_map)

        self.assertLess(
            int(low_heatmap[0, 0, 2]),
            int(threshold_heatmap[0, 0, 2]),
        )
        self.assertGreater(
            int(low_heatmap[0, 0, 0]),
            int(threshold_heatmap[0, 0, 0]),
        )

    def test_heatmap_max_cli_option(self):
        args = build_parser().parse_args(
            [
                "--input", "input",
                "--output-dir", "output",
                "--model-dir", "model",
                "--heatmap-max", "0.02",
            ]
        )
        self.assertEqual(args.heatmap_max, 0.02)

    def test_mask_outside_roi_is_rejected(self):
        detector = EfficientADDetector.__new__(EfficientADDetector)
        detector.device = torch.device("cpu")
        with self.assertRaises(ValueError):
            detector._build_valid_input_mask(
                (0, 0, 60, 40),
                [(50, 30, 20, 20)],
            )


if __name__ == "__main__":
    unittest.main()
