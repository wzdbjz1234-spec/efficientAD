import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from dual_detector import (
    DEFAULT_THRESHOLDS,
    SPECS,
    ImageInspection,
    ROIInspection,
    _Unit,
    annotate,
    build_parser,
    ground_truth_from_path,
    save_destination,
)



def _make_image(height=48, width=64):
    return np.random.default_rng(0).integers(0, 256, (height, width, 3)).astype(np.uint8)


def _make_inspection(rois, truth=None, predicted=0):
    return ImageInspection(
        input_path="C:/x/test/good/a.png",
        truth=truth,
        truth_label="NORMAL" if truth == 0 else "ANOMALY" if truth == 1 else "UNKNOWN",
        predicted=predicted,
        predicted_label="ANOMALY" if predicted else "NORMAL",
        is_misclassified=None if truth is None else (predicted != truth),
        rois=rois,
        annotated=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class GroundTruthTests(unittest.TestCase):
    def test_good_path_is_normal(self):
        self.assertEqual(ground_truth_from_path(Path("raw/test/good/a.png")), 0)

    def test_broken_path_is_anomaly(self):
        self.assertEqual(ground_truth_from_path(Path("raw/test/broken/a.png")), 1)

    def test_plain_folder_is_unknown(self):
        self.assertIsNone(ground_truth_from_path(Path("raw/images/a.png")))

    def test_conflicting_folders_are_unknown(self):
        self.assertIsNone(
            ground_truth_from_path(Path("good/test/broken/a.png"))
        )


class AnnotateTests(unittest.TestCase):
    def test_heatmap_blends_inside_roi(self):
        image = _make_image(48, 64)
        roi = (8, 10, 20, 15)
        x, y, w, h = roi
        anomaly_map = np.full((h, w), 2.0, dtype=np.float32)
        inspection = ROIInspection(
            name="ROI-30",
            roi=roi,
            score=2.0,
            threshold=1.0,
            is_anomaly=True,
            label="ANOMALY",
            anomaly_map=anomaly_map,
        )
        result = annotate(image, [inspection])
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        self.assertTrue(
            np.any(result[y : y + h, x : x + w] != image[y : y + h, x : x + w]),
            "heatmap should modify the ROI region",
        )
        outside = np.concatenate(
            [result[:y].reshape(-1), result[y + h :].reshape(-1)]
        )
        self.assertTrue(
            np.any(outside != np.concatenate(
                [image[:y].reshape(-1), image[y + h :].reshape(-1)]
            )),
            "top bar and text should modify pixels outside the ROI",
        )

    def test_annotate_without_rois_is_safe(self):
        result = annotate(_make_image(), [])
        self.assertEqual(result.shape, (48, 64, 3))


class SaveDestinationTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(tempfile.mkdtemp())

    def test_all_mode_keeps_everything(self):
        inspection = _make_inspection(rois=[], truth=0, predicted=0)
        destination = save_destination("all", inspection, Path("test/good/a.png"), self.output_dir)
        self.assertEqual(destination, self.output_dir / "annotated" / "test" / "good" / "a.png")

    def test_misclassified_mode_keeps_false_positive(self):
        inspection = _make_inspection(rois=[], truth=0, predicted=1)
        destination = save_destination("misclassified", inspection, Path("a.png"), self.output_dir)
        self.assertEqual(destination, self.output_dir / "misclassified" / "a.png")

    def test_misclassified_mode_keeps_false_negative(self):
        inspection = _make_inspection(rois=[], truth=1, predicted=0)
        destination = save_destination("misclassified", inspection, Path("a.png"), self.output_dir)
        self.assertEqual(destination, self.output_dir / "misclassified" / "a.png")

    def test_misclassified_mode_drops_correct(self):
        inspection = _make_inspection(rois=[], truth=0, predicted=0)
        self.assertIsNone(save_destination("misclassified", inspection, Path("a.png"), self.output_dir))

    def test_misclassified_mode_keeps_unknown(self):
        inspection = _make_inspection(rois=[], truth=None, predicted=0)
        destination = save_destination("misclassified", inspection, Path("a.png"), self.output_dir)
        self.assertEqual(destination, self.output_dir / "unknown" / "a.png")

    def test_inspection_record(self):
        inspection = _make_inspection(
            rois=[
                ROIInspection(
                    name="ROI-30", roi=(0, 0, 1, 1), score=0.25,
                    threshold=0.1, is_anomaly=True, label="ANOMALY",
                    anomaly_map=np.zeros((1, 1), dtype=np.float32),
                )
            ],
            truth=0,
            predicted=1,
        )
        record = inspection.as_record(annotated_path="out.png")
        self.assertEqual(record["truth_label"], "NORMAL")
        self.assertEqual(record["predicted_label"], "ANOMALY")
        self.assertTrue(record["is_misclassified"])
        self.assertEqual(record["ROI-30_score"], 0.25)
        self.assertEqual(record["inference_ms"], 0.0)

    def test_inspection_record_includes_inference_ms(self):
        inspection = _make_inspection(
            rois=[
                ROIInspection(
                    name="ROI-30", roi=(0, 0, 1, 1), score=0.25,
                    threshold=0.1, is_anomaly=True, label="ANOMALY",
                    anomaly_map=np.zeros((1, 1), dtype=np.float32),
                    inference_ms=1.25,
                )
            ],
            truth=0,
            predicted=1,
        )
        inspection.inference_ms = 1.25
        record = inspection.as_record()
        self.assertEqual(record["inference_ms"], 1.25)
        self.assertEqual(record["ROI-30_ms"], 1.25)


if __name__ == "__main__":
    unittest.main()


class ParserTests(unittest.TestCase):
    def test_threshold_defaults(self):
        args = build_parser().parse_args(["--input", "x"])
        self.assertEqual(args.threshold_30, DEFAULT_THRESHOLDS["ROI-30"])
        self.assertEqual(args.threshold_30, 0.2)
        self.assertEqual(args.threshold_31, DEFAULT_THRESHOLDS["ROI-31"])
        self.assertEqual(args.threshold_31, 0.912657)
        self.assertEqual(args.batch_size, 16)


class _FakeStudent(torch.nn.Module):
    def forward(self, tensors):
        output = torch.zeros(tensors.shape[0], 768, 8, 8)
        output[:, -384:] = 2.0
        return output


class _FakeAutoencoder(torch.nn.Module):
    def forward(self, tensors):
        return torch.zeros(tensors.shape[0], 384, 8, 8)


def _make_unit(q_start=0.0, q_end=1.0):
    return _Unit(
        spec=SPECS[0],
        student=_FakeStudent(),
        autoencoder=_FakeAutoencoder(),
        q_ae_start=torch.tensor(q_start),
        q_ae_end=torch.tensor(q_end),
        threshold=0.5,
        roi=(0, 0, 8, 8),
        masks=[],
    )


class TeacherFreeForwardTests(unittest.TestCase):
    def test_forward_shape_and_value(self):
        from dual_detector import DualROIDetector

        detector = object.__new__(DualROIDetector)
        maps = detector._forward(_make_unit(), torch.zeros(2, 3, 256, 256))
        self.assertEqual(maps.shape, (2, 8, 8))
        np.testing.assert_allclose(maps[:, 2:6, 2:6], 0.4)

    def test_forward_zero_denominator(self):
        from dual_detector import DualROIDetector

        detector = object.__new__(DualROIDetector)
        maps = detector._forward(
            _make_unit(q_start=0.5, q_end=0.5), torch.zeros(1, 3, 256, 256)
        )
        np.testing.assert_allclose(maps, 0.0)
