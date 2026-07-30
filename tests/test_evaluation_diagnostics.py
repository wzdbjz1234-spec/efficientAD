import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from efficientad_tools.cli import _visualize_misclassifications
from efficientad_tools.evaluation import evaluate_dataset
from efficientad_tools.predictor import FeatureAnalysis, Prediction
from efficientad_tools.visualization import save_feature_visualization


class _FakePredictor:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, path, **kwargs):
        score = self.scores[Path(path).name]
        return SimpleNamespace(
            score=score,
            student_teacher_score=score * 0.4,
            autoencoder_score=score * 0.6,
        )


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_explicit_threshold_labels_false_positives_and_false_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name, names in {
                'good': ('normal_ok.png', 'normal_fp.png'),
                'broken': ('anomaly_fn.png', 'anomaly_ok.png'),
            }.items():
                class_dir = root / 'test' / class_name
                class_dir.mkdir(parents=True)
                for name in names:
                    Image.new('RGB', (12, 10)).save(class_dir / name)

            predictor = _FakePredictor({
                'normal_ok.png': 0.1,
                'normal_fp.png': 0.9,
                'anomaly_fn.png': 0.2,
                'anomaly_ok.png': 0.8,
            })
            result = evaluate_dataset(predictor, root, threshold=0.5)

            self.assertEqual(
                result.metrics['confusion_matrix'],
                {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1},
            )
            outcomes = {
                Path(record.path).name: record.outcome
                for record in result.records
            }
            self.assertEqual(outcomes['normal_fp.png'], 'false_positive')
            self.assertEqual(outcomes['anomaly_fn.png'], 'false_negative')
            self.assertEqual(result.metrics['decision_threshold'], 0.5)
            self.assertEqual(result.metrics['threshold_source'], 'command_line')

            output_dir = root / 'diagnostics'

            def fake_save(_image, _prediction, output, **_kwargs):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                return output

            with patch(
                'efficientad_tools.cli.save_feature_visualization',
                side_effect=fake_save,
            ) as save_mock:
                updated = _visualize_misclassifications(
                    predictor,
                    result,
                    root,
                    output_dir,
                    top_k=2,
                )

            self.assertEqual(save_mock.call_count, 2)
            visualized = {
                Path(record.path).name: record.visualization_path
                for record in updated.records
                if record.visualization_path is not None
            }
            self.assertEqual(
                set(visualized),
                {'normal_fp.png', 'anomaly_fn.png'},
            )
            self.assertIn('false_positive', visualized['normal_fp.png'])
            self.assertIn('false_negative', visualized['anomaly_fn.png'])

    def test_feature_diagnostic_contains_all_model_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'input.png'
            output_path = root / 'diagnostic.png'
            Image.new('RGB', (24, 20), color=(40, 80, 120)).save(image_path)

            shape = (4, 8, 8)
            rng = np.random.default_rng(42)
            teacher = rng.normal(size=shape)
            student_teacher = rng.normal(size=shape)
            autoencoder = rng.normal(size=shape)
            student_autoencoder = rng.normal(size=shape)
            prediction = Prediction(
                score=0.8,
                student_teacher_score=0.7,
                autoencoder_score=0.9,
                anomaly_map=rng.random((20, 24)),
                student_teacher_map=rng.random((20, 24)),
                autoencoder_map=rng.random((20, 24)),
                features=FeatureAnalysis(
                    teacher=teacher,
                    student_teacher=student_teacher,
                    autoencoder=autoencoder,
                    student_autoencoder=student_autoencoder,
                    student_teacher_diff=(teacher - student_teacher) ** 2,
                    autoencoder_diff=(autoencoder - student_autoencoder) ** 2,
                ),
            )

            saved = save_feature_visualization(
                image_path,
                prediction,
                output_path,
                top_k=2,
                title='FALSE POSITIVE',
            )
            self.assertEqual(saved, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
