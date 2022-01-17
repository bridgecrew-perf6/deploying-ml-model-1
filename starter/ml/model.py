from sklearn.metrics import (fbeta_score, precision_score,
                             recall_score, make_scorer)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             random_state=16)
    # parameters = {
    #     'base_estimator__max_depth': [2, 3, 5],
    #     'base_estimator__min_samples_split': [100, 200, 500],
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 1]
    # }
    #
    # # Make an fbeta_score scoring object using make_scorer()
    # scorer = make_scorer(fbeta_score, beta=0.5)
    #
    # # Perform grid search on the classifier using 'scorer' as the scoring
    # # method using GridSearchCV()
    # grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
    #
    # # Fit the grid search object to the training data and find the optimal
    # # parameters using fit()
    # grid_fit = grid_obj.fit(X_train, y_train)

    model = clf.fit(X_train, y_train)
    #
    # # Get the estimator
    # best_model = grid_fit.best_estimator_

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds
