import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


df = pd.read_csv("data/sensors/5507.csv")

df["data"] = pd.to_datetime(df["data"])
df_filtered = df.filter(items=["data", "valore"])
df_grouped = df_filtered.groupby(df["data"].dt.to_period("D")).mean()
df_grouped["data"] = pd.to_datetime(df_grouped["data"])
df_grouped.to_csv("data/12574_grouped.csv", index=False, mode="w")


plt.plot(df_grouped["data"], df_grouped["valore"])
plt.legend([df["nometiposensore"].iloc[0]])
plt.show()
