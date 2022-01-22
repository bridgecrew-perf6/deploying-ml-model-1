import json
from fastapi.testclient import TestClient
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.

def test_post_greeting():
    r = client.post("/scores/",
                    data=json.dumps({"scores": {
                        "greeting": "Hello!"
                        }}))
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_post_scores():
    r = client.post("/scores/",
                    data=json.dumps({"scores": {
                        "training set": {
                            "precision": 0.7953266245949173,
                            "recall": 0.7760026626726577,
                            "f1-score": 0.7855458221024259
                        },
                        "testing set": {
                            "precision": 0.8227513227513228,
                            "recall": 0.8298865910607072,
                            "f1-score": 0.8263035536366656
                        }
                    }
                    }))
    assert r.status_code == 200
    assert len(r.json()["scores"]) > 1


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
