"""Reusable multiclass evaluation metrics."""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }

    if y_proba is not None:
        try:
            # roc_auc_score's multi_class="ovr" path requires a 1D score array
            # (probability of the positive class) for exactly 2 classes —
            # passing the full (n_samples, 2) matrix raises. This project's
            # real target has 9 classes, but keep this correct generally
            # since compute_metrics is meant to be reusable.
            score = y_proba[:, 1] if len(labels) == 2 else y_proba
            multi_class_kwargs = {} if len(labels) == 2 else {"multi_class": "ovr"}
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true, score, labels=labels, average="macro", **multi_class_kwargs
            )
        except ValueError as e:
            # Can fail if a class is absent from a small fold/holdout split.
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_error"] = str(e)

    return metrics
