# import os
# import io
import sys
# import pickle
# import dvc.api
# import pandas as pd
from sklearn.model_selection import train_test_split

import get_model_artifacts
from ml.data import process_data
from ml.model import compute_model_metrics, inference


def helper_split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)

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


def main(categorical=None):
    model_scores = {}
    if categorical is not None:
        categorical_unique_values = data[categorical].unique().tolist()

        for val in categorical_unique_values:
            X_feat = data[data[categorical] == val]
            X_val, y_val, _, _ = process_data(
                X_feat, categorical_features=cat_features,
                skewed_features=skewed_features,
                label="salary", training=False, encoder=encoder, lb=lb
            )
            popl_segment = f"{categorical}_{val}"
            scores = helper_get_scores(clf, X_val, y_val)
            model_scores[popl_segment] = scores

        print(model_scores)
        return model_scores
    else:
        train, test = helper_split_data(data)

        for split, label in zip([train, test], ['training', 'testing']):
            X, y, _, _ = process_data(
                split, categorical_features=cat_features,
                skewed_features=skewed_features,
                label="salary", training=False, encoder=encoder, lb=lb
            )

            model_scores[f"{label} set"] = helper_get_scores(clf, X, y)

        print(model_scores)
        return model_scores


if __name__ == "__main__":

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

    clf, encoder, lb, data = get_model_artifacts.main()

    args = sys.argv

    if len(args) == 1:
        main()

    if len(args) > 1:
        main(args[1])
