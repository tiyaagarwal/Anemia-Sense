from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from app.routes import bp
    app.register_blueprint(bp)

    _warm_up()

    return app


def _warm_up() -> None:
    """SHAP's explainer has a one-time initialization cost on its first call
    (~2-3s). Pay that cost once here at startup instead of making it the
    first real user's problem."""
    from app.services.prediction_service import _background, predict

    sample = _background.iloc[0].to_dict()
    predict(sample)
