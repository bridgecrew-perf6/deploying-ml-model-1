import os
import pytest
# import pickle
import pandas as pd


@pytest.fixture(scope="session")
def data():
    """ A function to read the cleaned version of the dataset."""
    df = pd.read_csv(os.getcwd(), "data/census_clean.csv")
    return df


def test_data_shape(data):
    """ If the data has no null values then it passes this test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_column_presence_and_type(data):
    """A dictionary with the column names as key and a function that verifies
    the expected dtype for that column. We do not check strict dtypes (like
    np.int32 vs np.int64) but general dtypes (like is_integer_dtype)"""
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(data[col_name]), \
            f"Column {col_name} failed test {format_verification_funct}"


def test_slice_averages(data):
    """ Test to see if our mean for hours-per-week worked per `education` slice
    is in the range 34 to 48. """
    for feature in data["education"].unique():
        avg_value = data[data["education"] == feature][
            "hours-per-week"].mean()
        assert (48 > avg_value > 34) \
            , f"For {feature}, average hours per week worked of {avg_value} " \
              f"not between 34 and 48."
