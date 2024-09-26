import pandas as pd
from models import GradientBoosting, RandomForest, SupportVectorMachine, RidgeRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import os
import json

def prepare_data(path: str, scenario: int, near_loc: list[pd.DataFrame]) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date").to_period("D")

    if scenario in [2, 3]:
        fourier = CalendarFourier(freq="YE", order=1)
        dp = DeterministicProcess(index=data.index, constant=False, order=1, seasonal=False, additional_terms=[fourier], drop=True)
        data = pd.concat([data, dp.in_sample()], axis=1)
        if scenario == 3:
            for near in near_loc:
                data = data.merge(near["value"], on="date", how="left", suffixes=("", "_near"))

    return data.dropna()

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

def dev_models(path: str, scenario: int = 1, near_loc: list[pd.DataFrame] = None) -> pd.DataFrame:
    data = prepare_data(path, scenario, near_loc)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    features, target = data.columns.difference(["date", "value"]), 'value'
    
    models = {
        "Random Forest": RandomForest.RandomForestModel(),
        "Gradient Boosting": GradientBoosting.GradientBoostingModel(),
        "Ridge Regression": RidgeRegression.RidgeRegressionModel(),
        "Support Vector Machine": SupportVectorMachine.SupportVectorMachineModel(),
    }
    
    stats, params = train_and_evaluate(models, train_data, test_data, features, target)
    del data, train_data, test_data, models
    return pd.DataFrame(stats, columns=["Model", "Test MAE", "Test RMSE", "Test R2", "Train MAE", "Train RMSE", "Train R2"]), params

def process_files(folder: str):
    for file in os.listdir(f"data/_processed/{folder}/"):
        near_locs = [pd.read_csv(f"data/_processed/{near}/{file}", parse_dates=["date"]).set_index("date").to_period("D")
                     for near in ["Brera", "Citta Studi", "Montalbino"] if near != folder and os.path.exists(f"data/_processed/{near}/{file}")]

        for scenario in range(1, 4):
            stats, params = dev_models(f"data/_processed/{folder}/{file}", scenario=scenario, near_loc=near_locs if scenario == 3 else None)
            stats.to_csv(f"data/stats/Scenario {scenario}/{folder}/{file}", index=False)

            if os.path.exists(f"data/stats/best_params.json"):
                with open(f"data/stats/best_params.json", "r") as f:
                    best_params = json.load(f)
            else:
                best_params = {}

            file_name = file.split(".")[0]

            if str(scenario) in best_params:
                
                if folder in best_params[str(scenario)]:
                    best_params[str(scenario)][folder][file_name] = params
                else:
                    best_params[str(scenario)][folder] = {file_name: params}
            else:
                best_params[str(scenario)] = {folder: {file_name: params}}



            with open(f"data/stats/best_params.json", "w") as f:
                json.dump(best_params, f)
                
            print(f"Stats for scenario {scenario} of {file} in {folder} done")

def main():
    for folder in ["Brera", "Citta Studi", "Montalbino"]:
        process_files(folder)

if __name__ == "__main__":
    main()
