import os, sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference

data_path = os.path.join(os.getcwd(), "data/census_clean.csv")
model_path = os.path.join(os.getcwd(), "model/trained_adaboost_model.pkl")
encoder_path = os.path.join(os.getcwd(), "model/encoder.pkl")
label_binarizer_path = os.path.join(os.getcwd(), "model/lb.pkl")


def helper_read_data():
    data = pd.read_csv(data_path)
    return data


def helper_read_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def helper_read_encoder():
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return encoder


def helper_read_lb():
    with open(label_binarizer_path, "rb") as f:
        lb = pickle.load(f)

    return lb


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
        for split, label in zip([train, test], ['tarining', 'testing']):

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

    data = helper_read_data()
    clf = helper_read_model()
    encoder = helper_read_encoder()
    lb = helper_read_lb()

    train, test = helper_split_data(data)

    args = sys.argv

    if len(args) == 1:
        main()

    if len(args) > 1:
        main(args[1])
