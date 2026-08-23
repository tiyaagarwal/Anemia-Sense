"""Flask route tests. Requires model.pkl/scaler.pkl to already exist
(run `python train.py` first — CI does this before running pytest)."""

import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_home_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200


def test_predict_form_has_no_unused_pii_fields(client):
    res = client.get("/predict")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert 'name="address"' not in body
    assert 'name="phone"' not in body
    assert 'name="Hemoglobin"' in body


def test_result_valid_input_returns_prediction(client):
    res = client.post(
        "/result",
        data={"name": "Test", "Gender": "0", "Hemoglobin": "9", "MCH": "21.5", "MCHC": "29.6", "MCV": "71.2"},
    )
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Anemic" in body


def test_result_missing_field_returns_400(client):
    res = client.post(
        "/result",
        data={"name": "Test", "Gender": "0", "Hemoglobin": "9", "MCH": "21.5", "MCHC": "29.6"},
    )
    assert res.status_code == 400


def test_result_non_numeric_field_returns_400(client):
    res = client.post(
        "/result",
        data={"name": "Test", "Gender": "0", "Hemoglobin": "not-a-number", "MCH": "21.5", "MCHC": "29.6", "MCV": "71.2"},
    )
    assert res.status_code == 400


def test_result_missing_name_returns_400(client):
    res = client.post(
        "/result",
        data={"name": "", "Gender": "0", "Hemoglobin": "9", "MCH": "21.5", "MCHC": "29.6", "MCV": "71.2"},
    )
    assert res.status_code == 400
