import json
import pytest
import pickle
import pandas as pd
from fastapi.testclient import TestClient
# Import our app from main.py.
from main import app


# Instantiate the testing client with our app.
# client = TestClient(app)


@pytest.fixture
def df():
    with TestClient(app) as client:
        response = client.get("/artifacts/censusdf")
        df = pd.DataFrame(response.json())
    return df


@pytest.fixture
def model():
    with TestClient(app) as client:
        response = client.get("/artifacts/model")
        model = pickle.dumps(response.json())
    return model


@pytest.fixture
def encoder():
    with TestClient(app) as client:
        response = client.get("/artifacts/encoder")
        encoder = pickle.dumps(response.json())
    return encoder


@pytest.fixture
def binarizer():
    with TestClient(app) as client:
        response = client.get("/artifacts/binarizer")
        binarizer = pickle.dumps(response.json())
    return binarizer


# Write tests using the same syntax as with the requests module.

def test_read_artifacts():
    with TestClient(app) as client:
        response = client.get("/artifacts/binarizer")
        assert response.status_code == 200
        assert len(response.json()) > 0


def test_post_hello():
    data = {"greeting": "Hello!"}
    with TestClient(app) as client:
        res = client.post("/greet/", data=json.dumps(data))
        assert res.status_code == 200
        assert len(res.json()) == 1


def test_post_scores_gt50k():
    data = {
        "age": 49,
        "workclass": "Federal-gov",
        "fnlgt": 125892,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        "salary": ">50K"
    }
    with TestClient(app) as client:
        res = client.post("/inference/", data=json.dumps(data))
        assert res.status_code == 200
        assert len(res.json()["sample data"]) > 0


def test_post_scores_le50k():
    data = {
        "age": 46,
        "workclass": "Local_gov",
        "fnlgt": 172822,
        "education": "HS_grad",
        "education_num": 9,
        "marital_status": "Married_civ_spouse",
        "occupation": "Transport_moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United_States",
        "salary": "<=50K"
    }
    with TestClient(app) as client:
        res = client.post("/inference/", data=json.dumps(data))
        assert res.status_code == 200
        assert len(res.json()["sample data"]) > 0


def test_home():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert res.json() == {"greeting": "Hello!"}


def test_inference_path():
    with TestClient(app) as client:
        res = client.get("/inference/")
        assert res.status_code == 200
        assert res.json()["training set"]["f1-score"] > 0.7
        assert res.json()["testing set"]["f1-score"] > 0.7


def test_get_slice_scores_sex():
    with TestClient(app) as client:
        res = client.get("/inference/sex")
        assert res.status_code == 200
        assert res.json()["sex_Male"]["recall"] > 0.8
        assert res.json()["sex_Female"]["recall"] > 0.8


def test_get_malformed_path():
    with TestClient(app) as client:
        res = client.get("/workclass/")
        assert res.status_code != 200
