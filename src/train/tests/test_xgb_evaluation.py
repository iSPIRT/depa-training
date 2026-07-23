import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pytrain import xgb_train
from pytrain.utilities.eval_tools import METRIC_REGISTRY, compute_metrics


class FakeBooster:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, _matrix):
        return self.predictions


class XgbEvaluationTests(unittest.TestCase):
    def make_trainer(self, output_path, private=False):
        trainer = xgb_train.Train_XGB()
        trainer.X_test = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        trainer.y_test = np.asarray([0, 0, 1, 1])
        trainer.task_type = "classification"
        trainer.is_private = private
        trainer.privacy_config = {"epsilon": 5.0, "delta": 0.00001}
        trainer.feature_min = [0.0]
        trainer.feature_max = [3.0]
        trainer.paths = {"trained_model_output_path": str(output_path)}
        trainer.config = {
            "task_type": "classification",
            "metrics": ["accuracy", "roc_auc"],
            "paths": trainer.paths.copy(),
        }
        return trainer

    def test_standard_model_passes_raw_scores_to_metrics(self):
        scores = np.asarray([0.4, 0.45, 0.1, 0.2])
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer.bst = FakeBooster(scores)
            trainer.non_dp_bst = None

            with patch.object(xgb_train.xgb, "DMatrix", return_value=object()), patch.object(
                xgb_train, "compute_metrics", return_value={}
            ) as metrics:
                trainer.inference_and_eval()

        np.testing.assert_array_equal(metrics.call_args.args[0], scores)

    def test_private_and_baseline_models_pass_raw_scores(self):
        private_scores = np.asarray([0.1, 0.2, 0.7, 0.8])
        baseline_scores = np.asarray([0.15, 0.25, 0.65, 0.75])
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp), private=True)
            trainer.bst = FakeBooster(private_scores)
            trainer.non_dp_bst = FakeBooster(baseline_scores)

            with patch.object(
                xgb_train.dp_xgb, "DMatrix", return_value=object()
            ), patch.object(
                xgb_train.xgb, "DMatrix", return_value=object()
            ), patch.object(
                xgb_train, "compute_metrics", return_value={}
            ) as metrics:
                trainer.inference_and_eval()

        self.assertEqual(metrics.call_count, 2)
        np.testing.assert_array_equal(metrics.call_args_list[0].args[0], private_scores)
        np.testing.assert_array_equal(metrics.call_args_list[1].args[0], baseline_scores)

    def test_discrete_metrics_threshold_scores_but_auc_keeps_ranking(self):
        targets = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.4, 0.45, 0.1, 0.2])
        hard_labels = (scores > 0.5).astype(int)
        config = {
            "task_type": "classification",
            "metrics": ["accuracy", "roc_auc"],
            "paths": {},
        }

        score_metrics = compute_metrics(scores, targets, config=config)
        hard_metrics = compute_metrics(hard_labels, targets, config=config)

        self.assertEqual(score_metrics["accuracy"], hard_metrics["accuracy"])
        self.assertEqual(score_metrics["roc_auc"], 0.0)
        self.assertEqual(hard_metrics["roc_auc"], 0.5)

    def test_precision_recall_curve_keeps_score_thresholds(self):
        targets = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.4, 0.45, 0.1, 0.2])
        hard_labels = (scores > 0.5).astype(int)
        metric = METRIC_REGISTRY["precision_recall_curve"]["fn"]
        meta = {"is_binary": True, "n_classes": 2, "task": "classification"}

        _, _, score_thresholds = metric(scores, targets, {}, meta)
        _, _, hard_thresholds = metric(hard_labels, targets, {}, meta)

        self.assertGreater(len(score_thresholds), len(hard_thresholds))


if __name__ == "__main__":
    unittest.main()
