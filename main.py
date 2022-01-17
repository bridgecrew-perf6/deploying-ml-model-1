import os
import json
import subprocess

from fastapi import FastAPI
# from typing import Union, List, Optional
from pydantic import BaseModel


class ModelScores(BaseModel):
    scores: dict


app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}


@app.post("/scores/")
async def create_scores(model_scores: ModelScores):
    return model_scores


@app.get("/scores/")
async def get_scores():
    # run_command = "python starter/model_performance_on_slices.py"
    # model_scores = os.system(run_command)
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
