import numpy as np

from src.models.evaluate import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    metrics = compute_metrics(y_true, y_pred, labels=["A", "B"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert set(metrics["per_class"].keys()) == {"A", "B"}


def test_compute_metrics_with_errors_and_proba():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "A", "A", "B"]
    y_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8]])
    metrics = compute_metrics(y_true, y_pred, y_proba, labels=["A", "B"])

    assert metrics["accuracy"] == 0.75
    assert 0 < metrics["macro_f1"] < 1
    assert metrics["roc_auc_ovr_macro"] is not None
