# Script to train machine learning model.

# Add the necessary imports for the starter code.
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Split the data into features and target label
salary_raw = data['salary']
features_raw = data.drop('salary', axis=1)

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
# train, test = train_test_split(data, test_size=0.20)
num_features = ['age',
                'fnlgt',
                'education-num',
                'capital-gain',
                'capital-loss',
                'hours-per-week']

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

# For highly-skewed feature distributions such as 'capital-gain' and
# 'capital-loss', we apply a logarithmic transformation on the data so that
# the very large and very small values do not negatively affect the
# performance of a learning algorithm. Using a logarithmic transformation
# significantly reduces the range of values caused by outliers.
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed_features] = (features_raw[skewed_features]
                                             .apply(lambda x: np.log(x + 1)))
# Normalize numerical features

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()

features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[num_features] = scaler.fit_transform(
    features_log_transformed[num_features])

# One-hot encode the 'features_log_minmax_transform' data using
# pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the the 'salary_raw' data to numerical values
le = LabelEncoder()
salary = le.fit_transform(salary_raw)

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    salary,
                                                    test_size=0.2,
                                                    random_state=16)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Train and save a model.
clf = AdaBoostClassifier(random_state=16)
clf = clf.fit(X_train, y_train)
predictions_test = clf.predict(X_test)
predictions_train = clf.predict(X_train)

# Compute F-score on the the training set using fbeta_score()
f_train = fbeta_score(y_train, predictions_train, beta=0.5)

# Compute F-score on the test set which is y_test
f_test = fbeta_score(y_test, predictions_test, beta=0.5)

# Print scores
print(f"Classifier {clf.__class__.__name__}\n"
      f"Training f_beta score (beta = 0.5) {f_train}\n"
      f"Testing f_beta score (beta = 0.5) {f_test}")

# trained_model.pkl
with open('model/trained_adaboost_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )

# Proces the test data with the process_data function.
