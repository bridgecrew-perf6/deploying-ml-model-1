import os
import sys
import json
from sklearn.model_selection import train_test_split
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


def helper_split_data(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size)

    return train, test


def helper_get_scores(model, predictors, target):
    preds = inference(model, predictors)
    precision, recall, fbeta = compute_model_metrics(target, preds)
    scores = {
        "precision": precision,
        "recall": recall,
        "f1-score": fbeta,

    }
    return scores


def run(df, model, encoder, binarizer, categorical_var=None):
    model_scores = {}
    slice_output = {}
    if categorical_var is not None:
        categorical_unique_values = df[categorical_var].unique().tolist()

        for val in categorical_unique_values:
            X_feat = df[df[categorical_var] == val]
            X_val, y_val, _, _ = process_data(
                X_feat, categorical_features=cat_features,
                skewed_features=skewed_features,
                label="salary", training=False, encoder=encoder, lb=binarizer
            )
            popl_segment = f"{categorical_var}_{val}"
            scores = helper_get_scores(model, X_val, y_val)
            model_scores[popl_segment] = scores

        slice_output[categorical_var] = model_scores
        output = json.dumps(slice_output, default=str) + "/n"
        with open("slice_output.txt", "a") as f:
            f.write(output)

        # print(model_scores)
        return model_scores
    else:
        train, test = helper_split_data(df)

        for split, label in zip([train, test], ['training', 'testing']):
            X, y, _, _ = process_data(
                split, categorical_features=cat_features,
                skewed_features=skewed_features,
                label="salary", training=False, encoder=encoder, lb=binarizer
            )

            model_scores[f"{label} set"] = helper_get_scores(model, X, y)

        # print(model_scores)
        return model_scores
