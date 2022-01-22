import io
import pickle
import dvc.api
import pandas as pd


def main():
    with dvc.api.open("model/trained_adaboost_model.pkl",
                      repo='https://github.com/hailuteju/deploying-ml-model',
                      mode="rb") as f:
        model = pickle.load(f)

    with dvc.api.open("model/encoder.pkl",
                      repo='https://github.com/hailuteju/deploying-ml-model',
                      mode="rb") as f:
        encoder = pickle.load(f)

    with dvc.api.open("model/lb.pkl",
                      repo='https://github.com/hailuteju/deploying-ml-model',
                      mode="rb") as f:
        lb = pickle.load(f)

    census_data_clean = dvc.api.read(
        'data/census_clean.csv',
        repo='https://github.com/hailuteju/deploying-ml-model'
    )
    census_data_csv = io.StringIO(census_data_clean)
    df = pd.read_csv(census_data_csv)

    return model, encoder, lb, df

