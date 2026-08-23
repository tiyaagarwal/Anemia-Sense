"""Flask route + REST API tests. Requires the model artifact to exist
(run `python -m src.models.train` first — CI does this before tests)."""

import pytest

from app import create_app

VALID_SAMPLE = {
    "WBC": 7.5, "LYMp": 35, "NEUTp": 55, "LYMn": 2.5, "NEUTn": 4.0,
    "RBC": 5.0, "HGB": 14.5, "HCT": 43, "MCV": 88, "MCH": 29,
    "MCHC": 33, "PLT": 280, "PDW": 15, "PCT": 0.2,
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize("path", ["/", "/predict", "/performance", "/methodology"])
def test_pages_load(client, path):
    res = client.get(path)
    assert res.status_code == 200


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_predict_form_has_no_pii_fields(client):
    body = client.get("/predict").get_data(as_text=True)
    assert 'name="address"' not in body
    assert 'name="phone"' not in body
    assert 'name="name"' not in body
    assert 'name="HGB"' in body


def test_predict_form_submit_valid(client):
    res = client.post("/predict", data=VALID_SAMPLE)
    assert res.status_code == 200
    body = res.get_data(as_text=True).lower()
    assert "prediction" in body


def test_predict_form_submit_missing_field(client):
    incomplete = {k: v for k, v in VALID_SAMPLE.items() if k != "HGB"}
    res = client.post("/predict", data=incomplete)
    assert res.status_code == 400


def test_predict_form_submit_out_of_range(client):
    bad = {**VALID_SAMPLE, "MCV": 999}
    res = client.post("/predict", data=bad)
    assert res.status_code == 400


def test_api_predict_valid(client):
    res = client.post("/api/v1/predict", json=VALID_SAMPLE)
    assert res.status_code == 200
    body = res.get_json()
    assert "prediction" in body
    assert "confidence" in body
    assert "top_features" in body
    assert len(body["top_features"]) > 0


def test_api_predict_missing_fields(client):
    res = client.post("/api/v1/predict", json={"WBC": 7.5})
    assert res.status_code == 400
    assert res.get_json()["error"] == "validation_failed"


def test_api_predict_non_json_body(client):
    res = client.post("/api/v1/predict", data="not json", content_type="text/plain")
    assert res.status_code == 400


def test_api_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"
