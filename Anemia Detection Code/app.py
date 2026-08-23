import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        f"{MODEL_PATH} / {SCALER_PATH} not found. Train the model first with: python train.py"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
fields = (
    pd.read_csv("data/anemia.csv")
    .drop(columns=["Result", "Name", "Address", "Phone"], errors="ignore")
    .columns.tolist()
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html", fields=fields)


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/result", methods=["POST"])
def result():
    name = request.form.get("name", "").strip()
    if not name:
        return render_template("error.html", message="Name is required."), 400

    values = []
    for field in fields:
        raw = request.form.get(field)
        if raw is None or raw.strip() == "":
            return render_template("error.html", message=f"Missing value for {field}."), 400
        try:
            values.append(float(raw))
        except ValueError:
            return render_template("error.html", message=f"{field} must be a number, got '{raw}'."), 400

    values_scaled = scaler.transform([values])
    prediction = model.predict(values_scaled)[0]
    prediction_result = "Anemic" if prediction == 1 else "Not Anemic"
    return render_template("result.html", name=name, result=prediction_result)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, use_reloader=False)
