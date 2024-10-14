import pandas as pd
from models import GradientBoosting
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

df = pd.read_csv("data/_processed/Brera/PM10.csv", parse_dates=["date"]).set_index("date").to_period("D")
#create range of dates from 01-01-2010 to 31-12-2010

range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D").to_period("D")



fig, ax = plt.subplots()
df = pd.read_csv("data/_processed/Brera/C6H6.csv", parse_dates=["date"]).set_index("date").to_period("D")
fourier = CalendarFourier(freq="YE", order=1)
dp = DeterministicProcess(index=df.index, constant=False, order=1, seasonal=False, additional_terms=[fourier], drop=True)
dp = dp.in_sample()   

from statsmodels.regression.linear_model import OLS
import numpy as np


df.plot(y="value", ax=ax)
plt.show()



""" df = pd.concat([df, dp.in_sample()], axis=1)

model = GradientBoosting.GradientBoostingModel()

model.train(df.dropna().drop(columns=["value"]), df.dropna()["value"])
df_pred = model.predict(df.dropna().drop(columns=["value"]))

#add column with predictions
df["value_pred"] = df_pred

df.plot(y=["value", "value_pred"])
plt.show() """

#plot correlogram of lagged values
""" from statsmodels.graphics.tsaplots import plot_acf
import os
for file in os.listdir("data/_processed/Brera"):
    df = pd.read_csv("data/_processed/Brera/" + file, parse_dates=["date"]).set_index("date").to_period("D")
    plot_acf(df["value"], lags=365)
    plt.show() """



