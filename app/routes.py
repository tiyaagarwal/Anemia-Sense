import json
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from app.services import prediction_service as svc

bp = Blueprint("main", __name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD_PATH = REPO_ROOT / "reports" / "data_quality" / "DATASET_CARD.md"
DATA_QUALITY_REPORT_PATH = REPO_ROOT / "reports" / "data_quality" / "data_quality_report.json"
LATEST_METRICS_PATH = REPO_ROOT / "reports" / "metrics" / "latest_metrics.json"
GLOBAL_IMPORTANCE_PATH = REPO_ROOT / "reports" / "figures" / "global_feature_importance.json"


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@bp.route("/")
def landing():
    return render_template("landing.html", metadata=svc.model_metadata())


@bp.route("/health")
def health():
    return jsonify({"status": "ok", "model": svc.model_metadata()["model_name"]})


@bp.route("/predict", methods=["GET"])
def predict_form():
    return render_template(
        "predict.html",
        feature_columns=svc.get_feature_columns(),
        ranges=svc.get_physiological_ranges(),
    )


@bp.route("/predict", methods=["POST"])
def predict_submit():
    try:
        values = svc.validate_input(request.form.to_dict())
    except svc.InputValidationError as e:
        return render_template("error.html", errors=e.errors), 400

    result = svc.predict(values)
    return render_template("results.html", result=result, input_values=values)


@bp.route("/performance")
def performance():
    metrics = _read_json(LATEST_METRICS_PATH)
    importance = _read_json(GLOBAL_IMPORTANCE_PATH)
    experiments = _read_json(REPO_ROOT / "reports" / "metrics" / "experiments.json") or []
    cv_results_sorted = (
        sorted(metrics["cross_validation_benchmark"].items(), key=lambda kv: kv[1]["macro_f1_mean"], reverse=True)
        if metrics else []
    )
    return render_template(
        "performance.html",
        metrics=metrics,
        importance=importance,
        experiments=experiments,
        cv_results_sorted=cv_results_sorted,
    )


@bp.route("/methodology")
def methodology():
    data_quality = _read_json(DATA_QUALITY_REPORT_PATH)
    dataset_card = DATASET_CARD_PATH.read_text() if DATASET_CARD_PATH.exists() else ""
    return render_template("methodology.html", data_quality=data_quality, dataset_card=dataset_card)


# --- REST API ---

@bp.route("/api/v1/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        values = svc.validate_input(payload)
    except svc.InputValidationError as e:
        return jsonify({"error": "validation_failed", "details": e.errors}), 400

    result = svc.predict(values)
    return jsonify(result)


@bp.route("/api/v1/health")
def api_health():
    return jsonify({"status": "ok", "model": svc.model_metadata()["model_name"]})
