import pandas as pd
from sklearn.base import BaseEstimator

def make_prediction(model: BaseEstimator, data: pd.DataFrame) -> pd.Series:
    predictions = model.predict(data)
    return pd.Series(predictions)