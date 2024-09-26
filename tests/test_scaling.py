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
    
    for file in os.listdir("data/_processed/Brera/"):

        data = pd.read_csv(f"data/_processed/Brera/{file}")
        data.drop(columns=["date"], inplace=True)

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        features, target = data.columns.difference(["date", "value"]), 'value'

        models = {
            "Random Forest": RandomForest.RandomForestModel(),
            "Gradient Boosting": GradientBoosting.GradientBoostingModel(),
            "Ridge Regression": RidgeRegression.RidgeRegressionModel(),
            "Support Vector Machine": SupportVectorMachine.SupportVectorMachineModel(),
        }

        stats, params = train_and_evaluate(models, train, test, features, target)
        print(stats)


    




