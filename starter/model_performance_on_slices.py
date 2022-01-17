import os, sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference

data_path = os.path.join(os.getcwd(), "data/census_clean.csv")
model_path = os.path.join(os.getcwd(), "model/trained_adaboost_model.pkl")


def helper_read_data():
    data = pd.read_csv(data_path)
    return data


def helper_read_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


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
        model_scores["training set"] = helper_get_scores(clf, X, y)
        X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features,
            skewed_features=skewed_features,
            label="salary", training=False, encoder=encoder, lb=lb
        )

        model_scores["testing set"] = helper_get_scores(clf, X_test, y_test)

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

    data = helper_read_data()
    clf = helper_read_model()
    train, test = helper_split_data(data)
    X, y, encoder, lb = process_data(
        train, categorical_features=cat_features,
        skewed_features=skewed_features,
        label="salary", training=True, encoder=None, lb=None
    )

    args = sys.argv

    if len(args) == 1:
        main()

    if len(args) > 1:
        main(args[1])
