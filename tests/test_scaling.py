import numpy as np
from models import RandomForest, GradientBoosting, RidgeRegression, SupportVectorMachine
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def min_max_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform min-max scaling on the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to scale.

    Returns
    -------
    pd.DataFrame
        The scaled data.
    """
    return (data - data.min()) / (data.max() - data.min())

def z_score_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform z-score scaling on the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to scale.

    Returns
    -------
    pd.DataFrame
        The scaled data.
    """
    return (data - data.mean()) / data.std()

def train_and_evaluate(models: list[object], train_data: pd.DataFrame, test_data: pd.DataFrame, features: list[str], target: str) -> list[list]:
    stats = []
    params = {}
    for model_name, model in models.items():
        model.train(train_data[features], train_data[target])
        test_metrics = model.evaluate(test_data[features], test_data[target])
        train_metrics = model.evaluate(train_data[features], train_data[target])
        params[model_name] = model.best_params
        stats.append([model_name] + list(test_metrics) + list(train_metrics))
    return stats, params

if __name__ == "__main__":
    
    for file in os.listdir("data/_processed/Brera"):
        df = pd.read_csv("data/_processed/Brera/" + file, parse_dates=["date"]).set_index("date").to_period("D")
        print(file)
        print(df.describe())

    




