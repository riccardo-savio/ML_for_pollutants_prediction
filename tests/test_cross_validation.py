import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import json

def load_data():
    df = pd.read_csv("data/_processed/Brera/PM10.csv")
    X = df.drop(columns=["value", "date"])
    y = df["value"]

    return X, y

X, y = load_data()

def gradient_boosting(X : pd.DataFrame, y : pd.Series, params : dict = {}):
    
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=5).mean()
    return model, score

def random_forest(X : pd.DataFrame, y : pd.Series, params : dict = {}):
    
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=5).mean()
    return model, score

def ridge_regression(X : pd.DataFrame, y : pd.Series, params : dict = {}):
    
    model = Ridge(**params)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=5).mean()
    return model, score

def support_vector_machine(X : pd.DataFrame, y : pd.Series, params : dict = {}):
   
    model = SVR(**params)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=5).mean()
    return model, score


def hyperopt_search(X : pd.DataFrame, y : pd.Series, model, space : dict, algo : str = "tpe.suggest", max_evals : int = 100):

    def objective(params):
        model1 = model(**params)
        score = cross_val_score(model1, X, y, cv=5).mean()
        return -score

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    return best


def main():
    model_gboost, score = gradient_boosting(X, y)
    print(f"Gradient Boosting: {score}")

    model_rf, score = random_forest(X, y)
    print(f"Random Forest: {score}")

    model_rr, score = ridge_regression(X, y)
    print(f"Ridge Regression: {score}")

    model_svm, score = support_vector_machine(X, y)
    print(f"Support Vector Machine: {score}")

    space_gradient_boosting = {
    "n_estimators": hp.choice("n_estimators", range(100, 1000)),
    "max_depth": hp.choice("max_depth", range(1, 20)),
    "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "min_samples_split": hp.choice("min_samples_split", range(2, 10)),
    "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 10)),
    }

    space_random_forest = {
    "n_estimators": hp.choice("n_estimators", range(100, 1000)),
    "max_depth": hp.choice("max_depth", range(1, 20)),
    "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 10)),
    "max_features": hp.choice("max_features", range(1, X.shape[1])),
    }

    space_ridge_regression = {
    "alpha": hp.uniform("alpha", 0.01, 1),
    }

    space_support_vector_machine = {
    "C": hp.uniform("C", 0.01, 1),
    "epsilon": hp.uniform("epsilon", 0.01, 1),
    "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
    }

    best = json.load(open("best_params.json"))

    """ best["GradientBoosting"] = hyperopt_search(X, y, GradientBoostingRegressor, space_gradient_boosting)
    best["RandomForest"] = hyperopt_search(X, y, RandomForestRegressor, space_random_forest)
    best["Ridge"] = hyperopt_search(X, y, Ridge, space_ridge_regression)
    best["SVM"] = hyperopt_search(X, y, SVR, space_support_vector_machine) 

    with open("best_params.json", "w") as f:
        f.write(str(best))"""

    print("After hyperparameter opt:")

    _, score = gradient_boosting(X, y, best["GradientBoosting"])
    print(f"Gradient Boosting: {score}")

    _, score = random_forest(X, y, best["RandomForest"])
    print(f"Random Forest: {score}")

    _, score = ridge_regression(X, y, best["Ridge"])
    print(f"Ridge Regression: {score}")

    _, score = support_vector_machine(X, y, best["SVM"])
    print(f"Support Vector Machine: {score}")

if __name__ == "__main__":
    main()