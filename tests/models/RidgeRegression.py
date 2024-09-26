from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

class RidgeRegressionModel:
    def __init__(self, param_grid=None):
        # Imposta la griglia dei parametri di default
        self.param_grid = param_grid or {
            'ridge__alpha': [0.1, 1, 10]
        }
        self.model = Ridge()
        self.best_params = None
        self.train_score = None

    def train(self, X, y) -> float:
        # Definisce il pipeline con lo scaler e il modello Ridge
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        # Definisce la GridSearch con cross-validation
        grid_search = GridSearchCV(
            pipeline, self.param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        # Esegue l'allenamento
        grid_search.fit(X, y)
        
        # Salva il modello migliore
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.train_score = grid_search.best_score_
        print(f"Best params: {self.best_params}")
        return grid_search.best_score_

    def predict(self, X) -> list:
        return self.model.predict(X)

    def evaluate(self, X_test, y_test) -> tuple:

        # Predizione sui dati di test
        predictions = self.predict(X_test)
        
        # Calcola le metriche di valutazione
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mae, rmse, r2
    
    def save(self, path: str):
        # Salva il modello
        joblib.dump(self.model, path)