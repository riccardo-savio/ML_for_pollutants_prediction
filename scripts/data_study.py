import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import datetime as dt


df = pd.read_csv("data/sensors/12574.csv")


df["data"] = pd.to_datetime(df["data"])
df_filtered = df.filter(items=["data", "valore"])
df_grouped = df_filtered.groupby(df['data'].dt.to_period('D')).mean()
df_grouped["data"] = pd.to_datetime(df_grouped["data"])
df_grouped["data"]= df_grouped['data'].map(dt.datetime.toordinal)
df_grouped.to_csv("data/12574_grouped.csv", index=False, mode="w")

x = df_grouped["data"]
y = df_grouped["valore"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel, color="r")
plt.show()