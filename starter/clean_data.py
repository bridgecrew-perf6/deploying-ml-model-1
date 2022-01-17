import numpy as np
import pandas as pd

df = pd.read_csv('data/census.csv')

df.columns = list(map(str.strip, df.columns))

categorical_features = df.select_dtypes(include=['object']).columns

df[categorical_features] = (df[categorical_features]
                            .apply(lambda s: s.str.strip()))

df.replace('?', np.NaN, inplace=True)

df_clean = df.dropna(how='any')

df_clean.to_csv('data/census_clean.csv', index=False)