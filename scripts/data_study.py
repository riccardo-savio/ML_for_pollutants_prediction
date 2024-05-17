
import os

from pandas import DataFrame




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


def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error
    
    # Load the data
    data = pd.read_csv("data/_processed/Brera/sin_cos_PM10.csv")

    PM2 = pd.read_csv("data/_processed/Brera/sin_cos_PM2.5.csv")[['data', 'PM2']]

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Specify the features and target variables
    features = ["PM10_dayb", "Humidity","Rain","Temp","Wind", "Cos_Month", "Cos_Day"]
    target = "PM10"
    
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

    #Evaluate the models on the training set
    train_linear_predictions = linear_model.predict(train_data[features])
    train_rf_predictions = rf_model.predict(train_data[features])
    train_dt_predictions = dt_model.predict(train_data[features])
    train_gb_predictions = gb_model.predict(train_data[features])
    train_nn_predictions = nn_model.predict(train_data[features])
    train_rr_predictions = rr_model.predict(train_data[features])
    train_sv_predictions = sv_model.predict(train_data[features])
    
    # Evaluate the models on the test set
    linear_predictions = linear_model.predict(test_data[features])
    rf_predictions = rf_model.predict(test_data[features])
    dt_predictions = dt_model.predict(test_data[features])
    gb_predictions = gb_model.predict(test_data[features])
    nn_predictions = nn_model.predict(test_data[features])
    rr_predictions = rr_model.predict(test_data[features])
    sv_predictions = sv_model.predict(test_data[features])
    
    # Print the evaluation metrics
    linear_rmse = root_mean_squared_error(test_data[target], linear_predictions)
    rf_rmse = root_mean_squared_error(test_data[target], rf_predictions)
    dt_rmse = root_mean_squared_error(test_data[target], dt_predictions)
    gb_rmse = root_mean_squared_error(test_data[target], gb_predictions)
    nn_rmse = root_mean_squared_error(test_data[target], nn_predictions)
    rr_rmse = root_mean_squared_error(test_data[target], rr_predictions)
    sv_rmse = root_mean_squared_error(test_data[target], sv_predictions)

    train_linear_rmse = root_mean_squared_error(train_data[target], train_linear_predictions)
    train_rf_rmse = root_mean_squared_error(train_data[target], train_rf_predictions)
    train_dt_rmse = root_mean_squared_error(train_data[target], train_dt_predictions)
    train_gb_rmse = root_mean_squared_error(train_data[target], train_gb_predictions)
    train_nn_rmse = root_mean_squared_error(train_data[target], train_nn_predictions)
    train_rr_rmse = root_mean_squared_error(train_data[target], train_rr_predictions)
    train_sv_rmse = root_mean_squared_error(train_data[target], train_sv_predictions)

    print("Linear Regression RMSE:", train_linear_rmse, linear_rmse)
    print("Random Forest RMSE:", train_rf_rmse, rf_rmse)
    print("Decision Tree RMSE:", train_dt_rmse, dt_rmse)
    print("Gradient Boosting RMSE:", train_gb_rmse, gb_rmse)
    print("Neural Network RMSE:", train_nn_rmse, nn_rmse)
    print("Ridge Regression RMSE:", train_rr_rmse, rr_rmse)
    print("Support Vector RMSE:", train_sv_rmse, sv_rmse)

    # calculate r squared

    from sklearn.metrics import r2_score

    linear_r2 = r2_score(test_data[target], linear_predictions)
    rf_r2 = r2_score(test_data[target], rf_predictions)
    dt_r2 = r2_score(test_data[target], dt_predictions)
    gb_r2 = r2_score(test_data[target], gb_predictions)
    nn_r2 = r2_score(test_data[target], nn_predictions)
    rr_r2 = r2_score(test_data[target], rr_predictions)
    sv_r2 = r2_score(test_data[target], sv_predictions)

    train_linear_r2 = r2_score(train_data[target], train_linear_predictions)
    train_rf_r2 = r2_score(train_data[target], train_rf_predictions)
    train_dt_r2 = r2_score(train_data[target], train_dt_predictions)
    train_gb_r2 = r2_score(train_data[target], train_gb_predictions)
    train_nn_r2 = r2_score(train_data[target], train_nn_predictions)
    train_rr_r2 = r2_score(train_data[target], train_rr_predictions)
    train_sv_r2 = r2_score(train_data[target], train_sv_predictions)

    print("Linear Regression R2:", train_linear_r2, linear_r2)
    print("Random Forest R2:", train_rf_r2, rf_r2)
    print("Decision Tree R2:", train_dt_r2, dt_r2)
    print("Gradient Boosting R2:", train_gb_r2, gb_r2)
    print("Neural Network R2:", train_nn_r2, nn_r2)
    print("Ridge Regression R2:", train_rr_r2, rr_r2)
    print("Support Vector R2:", train_sv_r2, sv_r2)
    

    


    


if __name__ == "__main__":
    main()
