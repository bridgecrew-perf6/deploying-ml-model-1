import os
import json
import subprocess

from fastapi import FastAPI
# from typing import Union, List, Optional
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


class Greet(BaseModel):
    greeting: str = "Hello!"


class DataItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str
    salary: str

    class Config:
        schema_extra = {
            "example": {
                'age': 49,
                'workclass': 'Federal-gov',
                'fnlgt': 125892,
                'education': 'Bachelors',
                'education_num': 13,
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': 'United-States',
                'salary': '>50K',
            }
        }


app = FastAPI()


@app.get("/")
async def print_hello():
    return {"greeting": "Hello!"}


@app.post("/")
async def post_hello(greet: Greet):
    return greet


@app.post("/scores/")
async def create_scores(data_item: DataItem):
    item = json.dumps(jsonable_encoder(data_item))
    response = (subprocess.run(
        ['python', 'starter/ingest_data_4inference.py',
         item],
        capture_output=True).stdout.decode('utf-8'))
    return response


@app.get("/scores/")
async def get_scores():
    response = (subprocess.run(
        ['python', 'starter/model_performance_on_slices.py'],
        capture_output=True).stdout.decode('utf-8'))

    response = json.loads(response.replace("\n", '')
                          .replace('"', "#")
                          .replace("'", '"')
                          .replace('#', "'"))
    print(len(response))
    return response


@app.get("/scores/{cat_feat}")
async def get_slice_scores(cat_feat: str):
    response = (subprocess.run(
        ['python', 'starter/model_performance_on_slices.py', f'{cat_feat}'],
        capture_output=True).stdout.decode('utf-8'))

    response = json.loads(response.replace("\n", '')
                          .replace('"', "#")
                          .replace("'", '"')
                          .replace('#', "'"))
    print(len(response))

    return response
