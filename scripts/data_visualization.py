import matplotlib.pyplot as plt
from pandas import DataFrame
import json


pollutants = {
    "Monossido di carbonio": "CO",
    "Biossido di azoto": "NO2",
    "Biossido di zolfo": "SO2",
    "Ozono": "O3",
    "PM10 (SM2005)": "PM10",
    "Particelle sospese PM2.5": "PM2.5",
    "Benzene": "C6H6",
}

sensor_names = {}


def plot_dfs(data: list[DataFrame], labels: list[str], standardize: bool = False, rolling: bool = False):
    import pandas as pd
    import matplotlib.dates as mdates
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    fig, axs = plt.subplots(len(data), 1, figsize=(20, 15))

    #find the minimum and maximum date in date column of all dataframes
    min_date = min([min(data[i]["date"]) for i in range(len(data))])
    max_date = max([max(data[i]["date"]) for i in range(len(data))])



    r= pd.date_range(start=min_date, end=max_date, freq="D").to_period("D")
    print(r)

    for i in range(len(axs)):

        row = pd.DataFrame({"date": r.to_timestamp(), "b": np.zeros(len(r))}).set_index("date")
        data[i]["date"] = pd.to_datetime(data[i]["date"])
        row = row.merge(data[i], how="left", on="date")
        #fill empty values with method ffill
        row["value"] = row["value"].fillna(method="bfill")
        print(row)
        
        if standardize:
            row["value"] = StandardScaler().fit_transform(row["value"].values.reshape(-1, 1))

        axs[i].plot("date", "value", data=row)
        axs[i].xaxis.set_major_locator(mdates.YearLocator(2))
        axs[i].xaxis.set_visible(True)
        axs[i].grid(True)
        axs[i].legend([labels[i]], loc="upper right")
        #aggiungi una curva di media mobile
        if rolling:
            row["value"] = row["value"].rolling(window=30).mean()
            axs[i].plot("date", "value", data=row, color="red")



    axs[len(axs) - 1].xaxis.set_visible(True)
    axs[len(axs) - 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for label in axs[len(axs) - 1].get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.09, hspace=0.2, left=0.05, right=0.95)
    plt.show()


def boxplot_dfs(data: list[DataFrame], columns: list[str]):
    import seaborn as sns
    import pandas as pd

    df = pd.concat(data, axis=1)
    df.columns = columns

    sns.boxplot(data=df)
    plt.savefig("imgs/boxplot.png")
    plt.show()


def heatmap(dfs: DataFrame, savefig: bool = False, path: str = "imgs/heatmap.png"):
    import seaborn as sns
    import numpy as np

    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(
        dfs.corr(),
        annot=True,
        mask=~np.tri(dfs.corr().shape[1], k=0, dtype=bool),
        cmap="coolwarm",
        
    )
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    if savefig:
        plt.savefig(path)
    plt.show()

def time_series_histogram(data: DataFrame, time_unit: str, pollutant: str):
    import pandas as pd

    data = data[["date", "value"]]
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"]).dt.strftime(time_unit)
    data = data.groupby("date").mean().reset_index()[["date", "value"]]

    if time_unit == "%m":
        data["date"] = pd.to_datetime(data["date"], format="%m").dt.strftime("%B")

    plt.title(f"Media annuale di {pollutant}")
    plt.bar(data["date"], data["value"], align="center", alpha=0.5, color="blue")
    plt.xticks(rotation=90)
    plt.ylabel("Valore medio")
    plt.tight_layout()
    plt.show()
    plt.clf()

def pollutant_skewness_kurtosis(data: DataFrame):
    from scipy.stats import skew, kurtosis
    

    skewness = skew(data["value"])
    kurt = kurtosis(data["value"])

    

    return {"skewness": skewness, "kurtosis": kurt}

def plot_auto_correlation(data: list[DataFrame], pollutants: list[str], savefig: bool = False, path: str = "imgs/autocorrelation.png"):
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axs = plt.subplots(len(data), 1, figsize=(20, 15))

    for i in range(len(data)):
        plot_acf(data[i]["value"], lags=365, ax=axs[i])
        axs[i].set_title(f"Autocorrelazione di {pollutants[i]}")

    plt.tight_layout()
    if savefig:
        plt.savefig(path)
    plt.show()

def bar_plot_stats(data: DataFrame, name: str, location: str, scenario: str, savefig: bool = False):

    import numpy as np
    import matplotlib.pyplot as plt


    plt.figure(figsize=(14, 6))

    #remove padding left
    plt.subplot(1, 2, 1)
    
    barWidth = 0.35
    r1 = np.arange(len(data["Model"]))
    r2 = [x + barWidth for x in r1]

    
    bars = plt.barh(r1, round(data["Train R2"], 4), color='cornflowerblue', height=barWidth, edgecolor='grey', label="Train data")
    bars1 = plt.barh(r2, round(data["Test R2"], 4), color='darkorange', height=barWidth, edgecolor='grey', label="Test data")

    ax = plt.gca()
    ax.bar_label(bars, padding=3, label_type='center')
    ax.bar_label(bars1, padding=3, label_type='center')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    
    plt.grid(True, axis='x')
    plt.yticks([r + barWidth/2 for r in range(len(data["Model"]))],  ["RFR", "GBOOST", "RIDGE", "SVR"])
    plt.title(f'Metrica di R$^{2}$ per {name} a {location}')

    # Creazione del grafico per MAE e RMSE
    plt.subplot(1, 2, 2)

    barWidth = 0.15
    r1 = np.arange(len(data["Model"]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    bar2 = plt.bar(r1, round(data["Train MAE"], 4), color='green', width=barWidth, edgecolor='grey', label='Train MAE')
    bar3 = plt.bar(r2, round(data["Test MAE"], 4), color='lightgreen', width=barWidth, edgecolor='grey', label='Test MAE')
    bar4 = plt.bar(r3, round(data["Train RMSE"], 4), color='blue', width=barWidth, edgecolor='grey', label='Train RMSE')
    bar5 = plt.bar(r4, round(data["Test RMSE"], 4), color='lightblue', width=barWidth, edgecolor='grey', label='Test RMSE')

    ax1 = plt.gca()

    box1 = ax.get_position()

    # Put a legend to the right of the current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

    plt.xticks([r + 1.5*barWidth for r in range(len(data["Model"]))], ["RFR", "GBOOST", "RIDGE", "SVR"])
    plt.title(f'Metriche di Errore per {name} a {location}')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)

    if savefig:
        plt.savefig(f"imgs/{scenario}/{location}_{name}.png")
    else:
        plt.show()
    plt.close()

def merge_dataframes(data: list[DataFrame], on: str) -> DataFrame:
    from functools import reduce
    return reduce(lambda x, y: x.merge(y, on=on), data).drop_duplicates()

def main():
    import os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from functools import reduce
    import seaborn as sns
    
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    """ path = "data/_processed/Brera/"
    data = pd.DataFrame({"date": pd.date_range(start="2013-01-01", end="2024-01-01", freq="D").to_period("D")}).set_index("date")
    i = 0
    for file in os.listdir(path):
        
        df = pd.read_csv(path + file, parse_dates=["date"]).set_index("date").to_period("D").drop(columns=["value_dayb"])
        if i != 0:
            df = df[["value"]]
        i += 1
        df.rename(columns={"value": file.split(".")[0]}, inplace=True)
        data = merge_dataframes([df, data], "date")
    
    heatmap(data) """

    for scenario in ["Scenario 1", "Scenario 2", "Scenario 3"]:
        for location in os.listdir("data/stats/" + scenario):
            for file in os.listdir("data/stats/" + scenario + "/" + location):
                df = pd.read_csv("data/stats/" + scenario + "/" + location + "/" + file)
                bar_plot_stats(df, file.split(".")[0], location, scenario, savefig=True)
    

if __name__ == "__main__":
    main()
