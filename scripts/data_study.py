import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
import os

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



def main():
    from sklearn.linear_model import LinearRegression
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from statsmodels.tools.eval_measures import rmse      
    import statsmodels.formula.api as smf
    

    data = pd.read_csv("data/ferno/NO2_comb.csv")
    dupl = list(data[data.duplicated()==True].index)
    data.drop(data.index[dupl], inplace=True)

    data['valore'] = np.abs(stats.zscore(data['valore']))
    outlPM10 = list(np.where(np.abs(data['valore']) > 3)[0])

    data['z_valore_HUMIDITY'] = np.abs(stats.zscore(data['valore_HUMIDITY']))
    outlHUMIDITY = list(np.where(np.abs(data['z_valore_HUMIDITY']) > 3)[0])

    data['z_valore_RAIN'] = np.abs(stats.zscore(data['valore_RAIN']))
    outlRAIN = list(np.where(np.abs(data['z_valore_RAIN']) > 3)[0])

    data['z_valore_WIND'] = np.abs(stats.zscore(data['valore_WIND']))
    outlWIND = list(np.where(np.abs(data['z_valore_WIND']) > 3)[0])

    data['z_valore_TEMP'] = np.abs(stats.zscore(data['valore_TEMP']))
    outlTEMP = list(np.where(np.abs(data['z_valore_TEMP']) > 3)[0])

    data['z_valore_DELTA'] = np.abs(stats.zscore(data['valore_DELTA']))
    outlDELTA = list(np.where(np.abs(data['z_valore_DELTA']) > 3)[0])

    outl = list(set(outlPM10 + outlHUMIDITY + outlRAIN + outlWIND + outlTEMP + outlDELTA))
    data.drop(data.index[outl], inplace=True)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    scores=[] #to store r squared
    rmse_list=[] #to store RMSE
    lrmodel = LinearRegression()
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        lrmodel.fit(X_train, y_train)
        y_predict = lrmodel.predict(X_test)
        scores.append(lrmodel.score(X_test, y_test))
        rmse_fold = rmse(y_test, y_predict)
        rmse_list.append(rmse_fold)

    lm = smf.ols(formula='valore ~ valore_HUMIDITY + valore_RAIN + valore_WIND + valore_TEMP + valore_DELTA', data = data).fit()
    comparison = pd.DataFrame({"y_test": y_test, "y_predict": y_predict})
    print("Summary: ", lm.summary())
    print("R Squared Mean: ", np.mean(scores))
    print("RMSE mean: ", np.mean(rmse_list))


main()
