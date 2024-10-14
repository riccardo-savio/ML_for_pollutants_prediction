import pandas as pd
import os 



def pollutants_improvements(a: str = "Sceanario 1", b: str = "Scenario 2") -> None:
    files = ["C6H6.csv", "NO2.csv", "PM10.csv", "PM25.csv", "O3.csv"]

    pols = pd.DataFrame({
        "Metrics": ["Test MAE", "Test RMSE", "Test R2", "Train MAE", "Train RMSE", "Train R2"]
    }).set_index("Metrics")

    for file in files:

        name = file.split(".")[0]

        pol1 = pd.DataFrame({
            "Test MAE": [0],
            "Test RMSE": [0],
            "Test R2": [0],
            "Train MAE": [0],
            "Train RMSE": [0],
            "Train R2": [0]
        }).transpose()

        
        pol1.columns = [name]

        pol2 = pol1.copy()
        for loc in ["Brera", "Citta Studi", "Montalbino"]:

            try:
                df1 = pd.read_csv(f"data/stats/{a}/{loc}/{file}").set_index("Model").mean()
                df2 = pd.read_csv(f"data/stats/{b}/{loc}/{file}").set_index("Model").mean()
            except:
                continue    
            pol1[name] = pol1[name] + df1
            pol2[name] = pol2[name] + df2
        
        pols[name] = (pol2[name] - pol1[name])/pol1[name]*100
    pols = pols.transpose()
    pols.columns = ["Test MAE", "Test RMSE", "Test R2", "Train MAE", "Train RMSE", "Train R2"]
    pols.to_csv(f"data/stats/{b}/improvements_pollutants.csv")
    print(pols)

def models_improvements(a:str = "Scenario 1", b:str = "Scenario 2"):
    files = ["C6H6.csv", "NO2.csv", "PM10.csv", "PM25.csv", "O3.csv"]
    Models1 = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "Ridge Regression", "Support Vector Machine"],
            "Test MAE": [0, 0, 0, 0],
            "Test RMSE": [0, 0, 0, 0],
            "Test R2": [0, 0, 0, 0],
            "Train MAE": [0, 0, 0, 0],
            "Train RMSE": [0, 0, 0, 0],
            "Train R2": [0, 0, 0, 0]
        }).set_index("Model")
    Models2 = Models1.copy()
    
    for loc in ["Brera", "Citta Studi", "Montalbino"]:
        for file in files:
            try:
                df1 = pd.read_csv(f"data/stats/{a}/{loc}/{file}").set_index("Model")
                df2 = pd.read_csv(f"data/stats/{b}/{loc}/{file}").set_index("Model")
            except:
                continue
            Models1 += df1
            Models2 += df2  
    
    models = (Models2 - Models1)/Models1*100
    models.to_csv(f"data/stats/{b}/improvements_models.csv")
    print(models)

def pollutants_stats(a: str = "Scenario 1") -> None:
    files = ["C6H6.csv", "NO2.csv", "PM10.csv", "PM25.csv", "O3.csv"]

    pols = pd.DataFrame({
        "Metrics": ["Test MAE", "Test RMSE", "Test R2", "Train MAE", "Train RMSE", "Train R2"]
    }).set_index("Metrics")

    

    for file in files:
        
        name = file.split(".")[0]

        pol1 = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting", "Ridge Regression", "Support Vector Machine"],
        "Test MAE": [0, 0, 0, 0],
        "Test RMSE": [0, 0, 0, 0],
        "Test R2": [0, 0, 0, 0],
        "Train MAE": [0, 0, 0, 0],
        "Train RMSE": [0, 0, 0, 0],
        "Train R2": [0, 0, 0, 0]
        }).set_index("Model")

        
        c = 0

        for loc in ["Brera", "Citta Studi", "Montalbino"]:

            try:
                df1 = pd.read_csv(f"data/stats/{a}/{loc}/{file}").set_index("Model")
            except:
                continue   
            c += 1 
            pol1 = pol1 + df1
        
        pols[name]= (pol1/c).mean(axis=0)
    
    pols.to_csv(f"data/stats/{a}/pollutants_stats.csv")
    print(pols)

def models_stats(a:str = "Scenario 1") -> None:
    files = ["C6H6.csv", "NO2.csv", "PM10.csv", "PM25.csv", "O3.csv"]
    Models = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "Ridge Regression", "Support Vector Machine"],
            "Test MAE": [0, 0, 0, 0],
            "Test RMSE": [0, 0, 0, 0],
            "Test R2": [0, 0, 0, 0],
            "Train MAE": [0, 0, 0, 0],
            "Train RMSE": [0, 0, 0, 0],
            "Train R2": [0, 0, 0, 0]
        }).set_index("Model")
    
    c = 0
    
    for loc in ["Brera", "Citta Studi", "Montalbino"]:
        for file in files:
            try:
                df1 = pd.read_csv(f"data/stats/{a}/{loc}/{file}").set_index("Model")
            except:
                continue
            c += 1
            Models += df1

    Models = Models/c
    Models.to_csv(f"data/stats/{a}/models_stats.csv")
    print(Models)

def find_best_models(a:str = "Scenario 1") -> None:

    files = ["C6H6.csv", "NO2.csv", "PM10.csv", "PM25.csv", "O3.csv"]
    
    
    def calculate_model_score(test_mae, test_rmse, test_r2, train_mae, train_rmse, train_r2, alpha=1, beta=1, gamma=2, delta=0):
    # Calcola la penalizzazione per l'overfitting
        overfitting_penalty = (
        abs(test_mae - train_mae) + 
        abs(test_rmse - train_rmse) + 
        abs(test_r2 - train_r2)
        )
        
        # Calcola il punteggio del modello
        score = (alpha / test_mae) + (beta / test_rmse) + (gamma * test_r2) - (delta * overfitting_penalty)
        
        return score

    data = {}
    for file in files:

        data[file.split(".")[0]] = {}

        dfs = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "Ridge Regression", "Support Vector Machine"],
            "Test MAE": [0, 0, 0, 0],
            "Test RMSE": [0, 0, 0, 0],
            "Test R2": [0, 0, 0, 0],
            "Train MAE": [0, 0, 0, 0],
            "Train RMSE": [0, 0, 0, 0],
            "Train R2": [0, 0, 0, 0]
        }).set_index("Model")

        for location in ["Brera", "Citta Studi", "Montalbino"]:
            try:
                df = pd.read_csv(f"data/stats/{a}/{location}/{file}").set_index("Model")
            except:
                continue
            dfs += df

        df["Score"] = df.apply(lambda x: calculate_model_score(x["Test MAE"], x["Test RMSE"], x["Test R2"], x["Train MAE"], x["Train RMSE"], x["Train R2"]), axis=1)
        best_model = df["Score"].idxmax()
        data[file.split(".")[0]] = f"{best_model}"
    print(data)
            
find_best_models("Scenario 1")
find_best_models("Scenario 2")
find_best_models("Scenario 3")

""" models_stats("Scenario 1")
pollutants_stats("Scenario 1")
pollutants_improvements("Scenario 1", "Scenario 2")
models_improvements("Scenario 1", "Scenario 2")
models_stats("Scenario 2")
pollutants_stats("Scenario 2")
pollutants_improvements("Scenario 2", "Scenario 3")
models_improvements("Scenario 2", "Scenario 3")
models_stats("Scenario 3")
pollutants_stats("Scenario 3") """
