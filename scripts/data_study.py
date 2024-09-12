
from sklearn.model_selection import cross_val_score
import pandas as pd

def linear_regression(data, x, y):
    from sklearn.linear_model import LinearRegression

    X = data[x]
    y = data[y]

    model = LinearRegression()
    model.fit(X, y)

    return model


def random_forest(data, x, y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = data[x]
    y = data[y]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])

    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )

    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_score_


def decision_tree(data, x, y):
    from sklearn.tree import DecisionTreeRegressor

    X = data[x]
    y = data[y]

    model = DecisionTreeRegressor(max_depth=6)
    model.fit(X, y)

    
    score = cross_val_score(model, X, y, cv=10).mean()
    return model, score


def gradient_boosting(data, x, y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = data[x]
    y = data[y]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingRegressor()) 
    ])

    param_grid = {
        'gb__n_estimators': [100, 200, 300],
        'gb__learning_rate': [0.01, 0.1, 1],
        'gb__max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_score_


def neural_network(data, x, y):
    from sklearn.neural_network import MLPRegressor

    X = data[x]
    y = data[y]

    model = MLPRegressor()
    model.fit(X, y)

    return model


def ridge_regression(data, x, y):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = data[x]
    y = data[y]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': [0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_score_


def support_vector(data, x, y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = data[x]
    y = data[y]

    # Supponiamo che train_data e test_data siano già definiti, e che features e target siano stati specificati

    # Pipeline per includere scaling e SVM in un unico processo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scala le feature
        ('svr', SVR())                 # Step 2: Applica l'SVM
    ])

    # Definizione della griglia dei parametri da esplorare
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': ['scale', 0.01, 0.1, 1],
        'svr__kernel': ['rbf']  # Usiamo il kernel RBF per la non-linearità
    }

    # Creazione del GridSearchCV
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    # Addestramento del modello con la ricerca dei migliori parametri
    grid_search.fit(X, y)
    # Modello migliore trovato dalla grid search

    return grid_search.best_estimator_, grid_search.best_score_


def gaussian_process(data, x, y):
    from sklearn.gaussian_process import GaussianProcessRegressor

    X = data[x]
    y = data[y]

    model = GaussianProcessRegressor()
    model.fit(X, y)

    return model


def study(path: str, scenario: int = 1, near_loc: list[pd.DataFrame] = None) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    import numpy as np

    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    data = pd.read_csv(path, parse_dates=["date"])
    data = data.set_index("date").to_period("D")

    if scenario in [2, 3]:
        fourier = CalendarFourier(freq="YE", order=2)  # 10 sin/cos pairs for "A"nnual seasonality
        dp = DeterministicProcess(
            index=data.index,
            constant=False,               # dummy feature for bias (y-intercept)
            order=1,                     # trend (order 1 means linear)
            seasonal=False,               # weekly seasonality (indicators)
            additional_terms=[fourier],  # annual seasonality (fourier)
            drop=True,                   # drop terms to avoid collinearity
        )
        data = pd.concat([data, dp.in_sample()], axis=1)

        if scenario == 3:
            for near in near_loc:
                data = data.merge(near["value"], on="date", how="left", suffixes=("", "_near"))

    data = data.dropna()

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    features = data.columns.difference(["date", "value"])
    target = 'value'


    # Train the random forest model
    rf_model, _ = random_forest(train_data, features, target)
    # Train the gradient boosting model
    gb_model, _ = gradient_boosting(train_data, features, target)
    # Train the ridge regression model
    rr_model, _ = ridge_regression(train_data, features, target)
    # Train the support vector model
    sv_model, _ = support_vector(train_data, features, target)

    # Evaluate the models on the training set
    train_rf_predictions = rf_model.predict(train_data[features])
    train_gb_predictions = gb_model.predict(train_data[features])
    train_rr_predictions = rr_model.predict(train_data[features])
    train_sv_predictions = sv_model.predict(train_data[features])
    # Evaluate the models on the test set
    rf_predictions = rf_model.predict(test_data[features])
    gb_predictions = gb_model.predict(test_data[features])
    rr_predictions = rr_model.predict(test_data[features])
    sv_predictions = sv_model.predict(test_data[features])

    rf_mae = mean_absolute_error(test_data[target], rf_predictions)
    gb_mae = mean_absolute_error(test_data[target], gb_predictions)
    rr_mae = mean_absolute_error(test_data[target], rr_predictions)
    sv_mae = mean_absolute_error(test_data[target], sv_predictions)

    train_rf_mae = mean_absolute_error(train_data[target], train_rf_predictions)
    train_gb_mae = mean_absolute_error(train_data[target], train_gb_predictions)
    train_rr_mae = mean_absolute_error(train_data[target], train_rr_predictions)
    train_sv_mae = mean_absolute_error(train_data[target], train_sv_predictions)

    stats = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Gradient Boosting",
                "Ridge Regression",
                "Support Vector"
            ]
        }
    )

    stats["Train MAE"] = [
        train_rf_mae,
        train_gb_mae,
        train_rr_mae,
        train_sv_mae
    ]
    stats["Test MAE"] = [
        rf_mae,
        gb_mae,
        rr_mae,
        sv_mae,
    ]

    # Print the evaluation metrics
    rf_rmse = root_mean_squared_error(test_data[target], rf_predictions)
    gb_rmse = root_mean_squared_error(test_data[target], gb_predictions)
    rr_rmse = root_mean_squared_error(test_data[target], rr_predictions)
    sv_rmse = root_mean_squared_error(test_data[target], sv_predictions)

    train_rf_rmse = root_mean_squared_error(train_data[target], train_rf_predictions)
    train_gb_rmse = root_mean_squared_error(train_data[target], train_gb_predictions)
    train_rr_rmse = root_mean_squared_error(train_data[target], train_rr_predictions)
    train_sv_rmse = root_mean_squared_error(train_data[target], train_sv_predictions)

    stats["Train RMSE"] = [
        train_rf_rmse,
        train_gb_rmse,
        train_rr_rmse,
        train_sv_rmse
    ]
    stats["Test RMSE"] = [
        rf_rmse,
        gb_rmse,
        rr_rmse,
        sv_rmse
    ]

    # calculate r squared
    from sklearn.metrics import r2_score

    train_rf_r2 = r2_score(train_data[target], train_rf_predictions)
    train_gb_r2 = r2_score(train_data[target], train_gb_predictions)
    train_rr_r2 = r2_score(train_data[target], train_rr_predictions)
    train_sv_r2 = r2_score(train_data[target], train_sv_predictions)

    rf_r2 = r2_score(test_data[target], rf_predictions)
    gb_r2 = r2_score(test_data[target], gb_predictions)
    rr_r2 = r2_score(test_data[target], rr_predictions)
    sv_r2 = r2_score(test_data[target], sv_predictions)

    stats["Train R2"] = [
        train_rf_r2,
        train_gb_r2,
        train_rr_r2,
        train_sv_r2
    ]
    stats["test R2"] = [rf_r2, gb_r2, rr_r2, sv_r2]

    return stats


def main():
    import os

    for folder in ["Brera", "Citta Studi", "Montalbino"]:
        for file in os.listdir(f"data/_processed/{folder}/"):

            stats = study(f"data/_processed/{folder}/{file}", scenario = 1)
            print("Stats for scenario 1 of", file, "in", folder, "done")
            stats.to_csv(f"data/stats/Scenario 1/{folder}/{file}", index=False)

            stats = study(f"data/_processed/{folder}/{file}", scenario = 2)
            print("Stats for scenario 2 of", file, "in", folder, "done")
            stats.to_csv(f"data/stats/Scenario 2/{folder}/{file}", index=False)

            near_locs = []
            for near in ["Brera", "Citta Studi", "Montalbino"]:
                if near == folder:
                    continue
                try:
                    near_loc = pd.read_csv(f"data/_processed/{near}/{file}", parse_dates=["date"])
                    near_loc = near_loc.set_index("date").to_period("D")
                    near_locs.append(near_loc)
                except FileNotFoundError:
                    continue

            stats = study(f"data/_processed/{folder}/{file}", scenario = 3, near_loc = near_locs)
            print("Stats for scenario 3 of", file, "in", folder, "done")
            stats.to_csv(f"data/stats/Scenario 3/{folder}/{file}", index=False)
    

if __name__ == "__main__":
    main()
