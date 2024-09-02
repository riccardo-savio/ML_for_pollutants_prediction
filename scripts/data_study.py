def linear_regression(data, x, y):
    from sklearn.linear_model import LinearRegression

    X = data[x]
    y = data[y]

    model = LinearRegression()
    model.fit(X, y)

    return model


def random_forest(data, x, y):
    from sklearn.ensemble import RandomForestRegressor

    X = data[x]
    y = data[y]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model


def decision_tree(data, x, y):
    from sklearn.tree import DecisionTreeRegressor

    X = data[x]
    y = data[y]

    model = DecisionTreeRegressor(max_depth=6)
    model.fit(X, y)

    return model


def gradient_boosting(data, x, y):
    from sklearn.ensemble import GradientBoostingRegressor

    X = data[x]
    y = data[y]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    return model


def neural_network(data, x, y):
    from sklearn.neural_network import MLPRegressor

    X = data[x]
    y = data[y]

    model = MLPRegressor()
    model.fit(X, y)

    return model


def ridge_regression(data, x, y):
    from sklearn.linear_model import Ridge

    X = data[x]
    y = data[y]

    model = Ridge()
    model.fit(X, y)

    return model


def support_vector(data, x, y):
    from sklearn.svm import SVR

    X = data[x]
    y = data[y]

    model = SVR()
    model.fit(X, y)

    return model


def gaussian_process(data, x, y):
    from sklearn.gaussian_process import GaussianProcessRegressor

    X = data[x]
    y = data[y]

    model = GaussianProcessRegressor()
    model.fit(X, y)

    return model


def study():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error

    p = "PM25"
    # Load the data
    data = pd.read_csv("data/_processed/Brera/multiPM25.csv")

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Specify the features and target variables
    c = []
    for i in range(1, 100):
        c.append("Sin_Month" + str(i))
        c.append("Cos_Month" + str(i))
    features = ["Humidity", "Rain", "Temp", "Wind"] + c
    target = p

    # Train the linear regression model
    linear_model = linear_regression(train_data, features, target)
    # Train the random forest model
    rf_model = random_forest(train_data, features, target)
    # Train the decision tree model
    dt_model = decision_tree(train_data, features, target)

    # Train the gradient boosting model
    gb_model = gradient_boosting(train_data, features, target)

    # Train the neural network model
    nn_model = neural_network(train_data, features, target)

    # Train the ridge regression model
    rr_model = ridge_regression(train_data, features, target)

    # Train the support vector model
    sv_model = support_vector(train_data, features, target)

    gaussian_process_model = gaussian_process(train_data, features, target)

    # Evaluate the models on the training set
    train_linear_predictions = linear_model.predict(train_data[features])
    train_rf_predictions = rf_model.predict(train_data[features])
    train_dt_predictions = dt_model.predict(train_data[features])
    train_gb_predictions = gb_model.predict(train_data[features])
    train_nn_predictions = nn_model.predict(train_data[features])
    train_rr_predictions = rr_model.predict(train_data[features])
    train_sv_predictions = sv_model.predict(train_data[features])
    train_gaussian_process_predictions = gaussian_process_model.predict(
        train_data[features]
    )

    # Evaluate the models on the test set
    linear_predictions = linear_model.predict(test_data[features])
    rf_predictions = rf_model.predict(test_data[features])
    dt_predictions = dt_model.predict(test_data[features])
    gb_predictions = gb_model.predict(test_data[features])
    nn_predictions = nn_model.predict(test_data[features])
    rr_predictions = rr_model.predict(test_data[features])
    sv_predictions = sv_model.predict(test_data[features])
    gaussian_process_predictions = gaussian_process_model.predict(test_data[features])

    linear_mae = mean_absolute_error(test_data[target], linear_predictions)
    rf_mae = mean_absolute_error(test_data[target], rf_predictions)
    dt_mae = mean_absolute_error(test_data[target], dt_predictions)
    gb_mae = mean_absolute_error(test_data[target], gb_predictions)
    nn_mae = mean_absolute_error(test_data[target], nn_predictions)
    rr_mae = mean_absolute_error(test_data[target], rr_predictions)
    sv_mae = mean_absolute_error(test_data[target], sv_predictions)
    gp_mae = mean_absolute_error(test_data[target], gaussian_process_predictions)

    train_linear_mae = mean_absolute_error(train_data[target], train_linear_predictions)
    train_rf_mae = mean_absolute_error(train_data[target], train_rf_predictions)
    train_dt_mae = mean_absolute_error(train_data[target], train_dt_predictions)
    train_gb_mae = mean_absolute_error(train_data[target], train_gb_predictions)
    train_nn_mae = mean_absolute_error(train_data[target], train_nn_predictions)
    train_rr_mae = mean_absolute_error(train_data[target], train_rr_predictions)
    train_sv_mae = mean_absolute_error(train_data[target], train_sv_predictions)
    train_gp_mae = mean_absolute_error(
        train_data[target], train_gaussian_process_predictions
    )

    stats = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest",
                "Decision Tree",
                "Gradient Boosting",
                "Neural Network",
                "Ridge Regression",
                "Support Vector",
                "Gaussian Process",
            ]
        }
    )

    stats["Train MAE"] = [
        train_linear_mae,
        train_rf_mae,
        train_dt_mae,
        train_gb_mae,
        train_nn_mae,
        train_rr_mae,
        train_sv_mae,
        train_gp_mae,
    ]
    stats["Test MAE"] = [
        linear_mae,
        rf_mae,
        dt_mae,
        gb_mae,
        nn_mae,
        rr_mae,
        sv_mae,
        gp_mae,
    ]

    # Print the evaluation metrics
    linear_rmse = root_mean_squared_error(test_data[target], linear_predictions)
    rf_rmse = root_mean_squared_error(test_data[target], rf_predictions)
    dt_rmse = root_mean_squared_error(test_data[target], dt_predictions)
    gb_rmse = root_mean_squared_error(test_data[target], gb_predictions)
    nn_rmse = root_mean_squared_error(test_data[target], nn_predictions)
    rr_rmse = root_mean_squared_error(test_data[target], rr_predictions)
    sv_rmse = root_mean_squared_error(test_data[target], sv_predictions)
    gp_rmse = root_mean_squared_error(test_data[target], gaussian_process_predictions)

    train_linear_rmse = root_mean_squared_error(
        train_data[target], train_linear_predictions
    )
    train_rf_rmse = root_mean_squared_error(train_data[target], train_rf_predictions)
    train_dt_rmse = root_mean_squared_error(train_data[target], train_dt_predictions)
    train_gb_rmse = root_mean_squared_error(train_data[target], train_gb_predictions)
    train_nn_rmse = root_mean_squared_error(train_data[target], train_nn_predictions)
    train_rr_rmse = root_mean_squared_error(train_data[target], train_rr_predictions)
    train_sv_rmse = root_mean_squared_error(train_data[target], train_sv_predictions)
    train_gp_rmse = root_mean_squared_error(
        train_data[target], train_gaussian_process_predictions
    )

    stats["Train RMSE"] = [
        train_linear_rmse,
        train_rf_rmse,
        train_dt_rmse,
        train_gb_rmse,
        train_nn_rmse,
        train_rr_rmse,
        train_sv_rmse,
        train_gp_rmse,
    ]
    stats["Test RMSE"] = [
        linear_rmse,
        rf_rmse,
        dt_rmse,
        gb_rmse,
        nn_rmse,
        rr_rmse,
        sv_rmse,
        gp_rmse,
    ]

    # calculate r squared

    from sklearn.metrics import r2_score

    linear_r2 = r2_score(test_data[target], linear_predictions)
    rf_r2 = r2_score(test_data[target], rf_predictions)
    dt_r2 = r2_score(test_data[target], dt_predictions)
    gb_r2 = r2_score(test_data[target], gb_predictions)
    nn_r2 = r2_score(test_data[target], nn_predictions)
    rr_r2 = r2_score(test_data[target], rr_predictions)
    sv_r2 = r2_score(test_data[target], sv_predictions)
    gp_r2 = r2_score(test_data[target], gaussian_process_predictions)

    train_linear_r2 = r2_score(train_data[target], train_linear_predictions)
    train_rf_r2 = r2_score(train_data[target], train_rf_predictions)
    train_dt_r2 = r2_score(train_data[target], train_dt_predictions)
    train_gb_r2 = r2_score(train_data[target], train_gb_predictions)
    train_nn_r2 = r2_score(train_data[target], train_nn_predictions)
    train_rr_r2 = r2_score(train_data[target], train_rr_predictions)
    train_sv_r2 = r2_score(train_data[target], train_sv_predictions)
    train_gp_r2 = r2_score(train_data[target], train_gaussian_process_predictions)

    stats["Train R2"] = [
        train_linear_r2,
        train_rf_r2,
        train_dt_r2,
        train_gb_r2,
        train_nn_r2,
        train_rr_r2,
        train_sv_r2,
        train_gp_r2,
    ]
    stats["test R2"] = [linear_r2, rf_r2, dt_r2, gb_r2, nn_r2, rr_r2, sv_r2, gp_r2]

    return stats


def main():
    stats = study()
    print(stats)


if __name__ == "__main__":
    main()
