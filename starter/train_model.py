# Script to train machine learning model.

# Add the necessary imports for the starter code.
import warnings
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (train_model,
                      compute_model_metrics, inference)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(data, test_size=0.20)

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

skewed_feats = ['capital-gain', 'capital-loss']

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, skewed_features=skewed_feats,
    label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, skewed_features=skewed_feats,
    label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save model
with open('model/trained_adaboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

preds_train = inference(model, X_train)
preds_test = inference(model, X_test)

precision_train, recall_train, fbeta_train = compute_model_metrics(
    y_train, preds_train)

precision_test, recall_test, fbeta_test = compute_model_metrics(
    y_test, preds_test)

# Print scores
printed_metrics = f"""
Classifier: {model.__class__.__name__}
Training:
Precision: {precision_train}, Recall: {recall_train}, Fbeta: {fbeta_train}
Test:
Precision: {precision_test}, Recall: {recall_test}, Fbeta: {fbeta_test}
"""

print(printed_metrics)
