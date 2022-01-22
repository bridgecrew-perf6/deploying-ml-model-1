import json
from fastapi.testclient import TestClient
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.

def test_post_greeting():
    data = {"greeting": "Hello!"}
    r = client.post("/", data=json.dumps(data))
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_post_scores():
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
    r = client.post("/scores/", data=json.dumps(data))
    assert r.status_code == 200
    assert len(r.json()) > 1


def test_get_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello!"}


def test_get_path():
    r = client.get("/scores/")
    assert r.status_code == 200
    assert r.json()["training set"]["f1-score"] > 0.7
    assert r.json()["testing set"]["f1-score"] > 0.7


def test_get_path_sex():
    r = client.get("/scores/sex")
    assert r.status_code == 200
    assert r.json()["sex_Male"]["recall"] > 0.8
    assert r.json()["sex_Female"]["recall"] > 0.8


# def test_get_path_education():
#     r = client.get("/scores/education")
#     assert r.status_code == 200
#     # assert len(r.json()) == 16


def test_get_path_malformed():
    r = client.get("/workclass/")
    assert r.status_code != 200
