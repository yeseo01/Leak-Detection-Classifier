import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.main import (
    CLASS_LABELS,
    build_confusion_matrix,
    evaluate_metrics,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([
            "in",
            "in",
            "noise",
            "noise",
            "normal",
            "normal",
            "other",
            "other",
            "out",
            "out",
        ])

        self.y_pred = np.array([
            "in",
            "noise",
            "noise",
            "noise",
            "normal",
            "out",
            "other",
            "in",
            "out",
            "other",
        ])

    def test_confusion_matrix_matches_sklearn(self):
        custom_matrix = build_confusion_matrix(
            self.y_true,
            self.y_pred,
        )

        sklearn_matrix = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=CLASS_LABELS,
        )

        np.testing.assert_array_equal(
            custom_matrix,
            sklearn_matrix,
        )

    def test_metrics_match_sklearn(self):
        matrix = build_confusion_matrix(
            self.y_true,
            self.y_pred,
        )

        recall, precision, accuracy = evaluate_metrics(matrix)

        expected_recall = recall_score(
            self.y_true,
            self.y_pred,
            labels=CLASS_LABELS,
            average="macro",
        )

        expected_precision = precision_score(
            self.y_true,
            self.y_pred,
            labels=CLASS_LABELS,
            average="macro",
        )

        expected_accuracy = accuracy_score(
            self.y_true,
            self.y_pred,
        )

        self.assertAlmostEqual(recall, expected_recall)
        self.assertAlmostEqual(precision, expected_precision)
        self.assertAlmostEqual(accuracy, expected_accuracy)


if __name__ == "__main__":
    unittest.main()
