import json
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

skewed_features = ['capital-gain', 'capital-loss']

record = '{"age": 49,\
          "workclass": "Federal-gov",\
          "fnlgt": 125892,\
          "education": "Bachelors",\
          "education-num": 13,\
          "marital-status": "Married-civ-spouse",\
          "occupation": "Exec-managerial",\
          "relationship": "Husband",\
          "race": "White",\
          "sex": "Male",\
          "capital-gain": 0,\
          "capital-loss": 0,\
          "hours-per-week": 40,\
          "native-country": "United-States",\
          "salary": ">50K"\
          }'


def helper_get_scores(model, predictors, target):
    preds = inference(model, predictors)
    precision, recall, fbeta = compute_model_metrics(target, preds)
    scores = {
        "precision": precision,
        "recall": recall,
        "f1-score": fbeta,

    }
    return scores


def run(model, encoder, binarizer, data=record):
    model_scores = {}
    data_json = json.loads(data)
    # print(data_json)
    sample_df = pd.DataFrame(data_json, index=[0])
    sample_df.columns = [col.replace("_", "-") for col in sample_df.columns]
    # print(sample_df)
    X, y, _, _ = process_data(
        sample_df, categorical_features=cat_features,
        skewed_features=skewed_features,
        label="salary", training=False, encoder=encoder, lb=binarizer
    )

    scores = helper_get_scores(model, X, y)
    model_scores["sample data"] = scores

    # print(model_scores)
    return model_scores

