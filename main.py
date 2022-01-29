import json
import pickle
import subprocess
import pandas as pd

from fastapi import FastAPI
# from typing import Union, List, Optional
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

import starter.model_slice as sms
import starter.ingest_data as sid


class Greet(BaseModel):
    greeting: str = "Hello!"


class DataItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(..., alias="capital-gain")
    capital_loss: float = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")
    salary: str

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 49,
                "workclass": "Federal-gov",
                "fnlgt": 125892,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": ">50K",
            }
        }


app = FastAPI()

artifacts = {}


@app.on_event("startup")
async def startup_event():
    global census_df, model, encoder, binarizer
    model = pickle.load(open(
        "./model/trained_adaboost_model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))
    census_df = pd.read_csv(open("./data/census_clean.csv", "r"))

    artifacts["censusdf"] = census_df.to_dict()
    artifacts["model"] = json.loads(json.dumps(model, default=str))
    artifacts["encoder"] = json.loads(json.dumps(encoder, default=str))
    artifacts["binarizer"] = json.loads(json.dumps(binarizer, default=str))


@app.get("/")
async def home():
    return {"greeting": "Hello!"}


@app.post("/greet/")
async def post_hello(selam: Greet):
    return selam


@app.post("/inference/")
async def post_scores(data_item: DataItem):
    item = json.dumps(jsonable_encoder(data_item))
    # response = (subprocess.run(
    #     ['python', 'starter/ingest_data.py',
    #      item],
    #     capture_output=True).stdout.decode('utf-8'))

    response = sid.run(model, encoder, binarizer, item)

    return response


@app.get("/inference/")
async def get_scores():
    # response = (subprocess.run(
    #     ['python', 'starter/model_slice.py'],
    #     capture_output=True).stdout.decode('utf-8'))
    #
    # response = json.loads(response.replace("\n", '')
    #                       .replace('"', "#")
    #                       .replace("'", '"')
    #                       .replace('#', "'"))
    # print(len(response))
    response = sms.run(census_df, model, encoder, binarizer)
    return response


@app.get("/inference/{cat_feat}")
async def get_slice_scores(cat_feat: str):
    # response = (subprocess.run(
    #     ['python', 'starter/model_slice.py', f'{cat_feat}'],
    #     capture_output=True).stdout.decode('utf-8'))
    #
    # response = json.loads(response.replace("\n", '')
    #                       .replace('"', "#")
    #                       .replace("'", '"')
    #                       .replace('#', "'"))
    # print(len(response))
    response = sms.run(census_df, model, encoder, binarizer, cat_feat)

    return response


@app.get("/artifacts/{artifact_id}")
async def read_artifacts(artifact_id: str):
    return artifacts[artifact_id]
