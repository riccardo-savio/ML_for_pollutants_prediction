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


def plot_dfs(data: list[DataFrame]):
    import pandas as pd
    import matplotlib.dates as mdates

    fig, axs = plt.subplots(len(data), 1, figsize=(20, 15))

    for i in range(len(axs)):
        print()
        row = data[i]
        row["data"] = pd.to_datetime(row["data"])

        axs[i].plot("data", "valore", data=row)
        axs[i].xaxis.set_major_locator(mdates.YearLocator(2))
        axs[i].xaxis.set_visible(False)
        axs[i].grid(True)

    axs[len(axs) - 1].xaxis.set_visible(True)
    axs[len(axs) - 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for label in axs[len(axs) - 1].get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")

    plt.show()


def boxplot_dfs(data: list[DataFrame], columns: list[str]):
    import seaborn as sns
    import pandas as pd

    df = pd.concat(data, axis=1)
    df.columns = columns

    sns.boxplot(data=df)
    plt.savefig("imgs/Brera .png")


def pollutants_heatmap(pollutants: list[DataFrame], columns: list[str]):
    import seaborn as sns
    import pandas as pd
    import numpy as np

    from functools import reduce

    df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="date", suffixes=("", "_")), pollutants
    )

    df.drop("date", axis=1, inplace=True)
    df.columns = columns

    sns.heatmap(
        df.corr(),
        annot=True,
        mask=~np.tri(df.corr().shape[1], k=0, dtype=bool),
        cmap="coolwarm",
    )
    plt.show()


def pollutant_meteo_correlation(
    pollutant: DataFrame, meteo: list[DataFrame], x_label: list[str], y_label: list[str]
):
    import pandas as pd
    import seaborn as sns
    import numpy as np

    HUMIDITY = pd.read_csv("data/ferno/humidity.csv")
    RAIN = pd.read_csv("data/ferno/rain.csv")
    WIND = pd.read_csv("data/ferno/wind.csv")
    TEMP = pd.read_csv("data/ferno/temp.csv")
    PM10 = pd.read_csv("data/ferno/NO2.csv")

    HUMIDITY.drop("idsensore", axis=1, inplace=True)
    RAIN.drop("idsensore", axis=1, inplace=True)
    WIND.drop("idsensore", axis=1, inplace=True)
    TEMP.drop("idsensore", axis=1, inplace=True)

    merged_data = (
        PM10.merge(RAIN, on="data", suffixes=("", "_rain"))
        .merge(WIND, on="data", suffixes=("", "_wind"))
        .merge(TEMP, on="data", suffixes=("", "_temp"))
        .merge(HUMIDITY, on="data", suffixes=("", "_PM10"))
    )

    merged_data.drop("data", axis=1, inplace=True)
    merged_data.columns = ["PM10", "RAIN", "WIND", "TEMP", "HUMIDITY"]

    sns.heatmap(
        merged_data.corr(),
        annot=True,
        mask=~np.tri(merged_data.corr().shape[1], k=-1, dtype=bool),
        cmap="coolwarm",
    )
    plt.show()


def plot_pollutant_meteo_rel(
    pollutants: list[DataFrame],
    meteo: list[DataFrame],
    x_labels: list[str],
    y_labels: list[str],
):
    import pandas as pd

    fig, axs = plt.subplots(len(pollutants), len(meteo), squeeze=False)

    for i in range(len(pollutants)):
        pollutant = pollutants[i][["data", "valore"]]
        pollutant["data"] = pd.to_datetime(pollutant["data"])

        for j in range(len(meteo)):
            if j == 0:
                axs[i][j].set_ylabel(y_labels[i])

            if i == len(pollutants) - 1:
                axs[i][j].set_xlabel(x_labels[j])

            meteo_s = meteo[j][["data", "valore"]]
            meteo_s["data"] = pd.to_datetime(meteo_s["data"])

            if x_labels[j] == "RN" or "WS":
                meteo_s["data"] = pd.to_datetime(meteo_s["data"]) + pd.Timedelta(days=1)

            meteo_s = meteo_s[["data", "valore"]].merge(
                pollutant, on="data", suffixes=(f"_{x_labels[j]}", f"_{y_labels[i]}")
            )
            meteo_s.drop("data", axis=1, inplace=True)
            meteo_s.columns = [x_labels[j], y_labels[i]]

            axs[i][j].scatter(x_labels[j], y_labels[i], data=meteo_s)
            axs[i][j].grid(True)

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

def bar_plot_stats(data: DataFrame, name: str, location: str, scenario: str):

    import numpy as np
    import matplotlib.pyplot as plt


    plt.figure(figsize=(14, 6))

    #remove padding left
    plt.subplot(1, 2, 1)
    
    barWidth = 0.35
    r1 = np.arange(len(data["Model"]))
    r2 = [x + barWidth for x in r1]

    
    bars = plt.barh(r1, round(data["Train R2"], 4), color='cornflowerblue', height=barWidth, edgecolor='grey', label="Train data")
    bars1 = plt.barh(r2, round(data["test R2"], 4), color='darkorange', height=barWidth, edgecolor='grey', label="Test data")

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
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks([r + 1.5*barWidth for r in range(len(data["Model"]))], ["RFR", "GBOOST", "RIDGE", "SVR"])
    plt.title(f'Metriche di Errore per {name} a {location}')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)


    plt.savefig(f"imgs/{scenario}/{location}_{name}.png")
    plt.close()

def main():
    import os
    import pandas as pd
    """ import os
    import pandas as pd
    import numpy as np

    for file in os.listdir("data/_processed/Brera/"):
        df = pd.read_csv(f"data/_processed/Brera/{file}", parse_dates=["date"])
        df["Day"] = df["date"].dt.day_of_year
        df["Month"] = df["date"].dt.month

        def encode(data, col, max_val):
            for i in range(1, 100):
                data["Sin_" + col + str(i)] = np.sin(
                    2 * i * np.pi * data[col] / max_val
                )
                data["Cos_" + col + str(i)] = np.cos(
                    2 * i * np.pi * data[col] / max_val
                )
            return data

        df = encode(df, "Day", 365)
        df = encode(df, "Month", 12)

        df.to_csv(f"data/_processed/Brera/multi{file}", index=False)
        # sns.heatmap(df.corr(), annot=True, mask=~np.tri(df.corr().shape[1], k=-1, dtype=bool), cmap="coolwarm")
        # plt.tight_layout()
        # plt.show() """
    
    """ dfs = []
    columns = []
    

    for poll in ["PM10", "PM25"]:
        df = pd.read_csv(f"data/_processed/Brera/{poll}.csv")[['date', 'value']]
        df.columns = ['date', f'{poll}_Brera']   
        columns.append(f'{poll}_Brera')
        dfs.append(df)
    
        df = pd.read_csv(f"data/_processed/Citta Studi/{poll}.csv")[['date', 'value']]
        df.columns = ['date', f'{poll}_Citta_Studi']  
        columns.append(f'{poll}_Citta_Studi') 
        dfs.append(df)

        df = pd.read_csv(f"data/_processed/Montalbino/{poll}.csv")[['date', 'value']]
        df.columns = ['date', f'{poll}_Montalbino']   
        columns.append(f'{poll}_Montalbino')
        dfs.append(df)

    pollutants_heatmap(dfs, columns)
 """
    """ import pandas as pd
    

    data = {}

    for folder in ["Brera", "Citta Studi", "Montalbino"]:
        data[folder] = {}
        for file in os.listdir(f"data/_processed/{folder}/"):
            df = pd.read_csv(f"data/_processed/{folder}/{file}")
            data[folder][file.split('.')[0]] = pollutant_skewness_kurtosis(df)

    with open ("data/stats/kurt_skew.json", "w") as f:
        json.dump(data, f, indent=4) """
    
    """ senarios = ["Scenario 1", "Scenario 2", "Scenario 3"]

    for scenario in senarios:

        folders = os.listdir(f"data/stats/{scenario}")

        for folder in folders:

            files = os.listdir(f"data/stats/{scenario}/{folder}")

            for file in files:

                df = pd.read_csv(f"data/stats/{scenario}/{folder}/{file}")

                bar_plot_stats(df, file.split(".")[0], folder, scenario)    """   


    df = pd.read_csv("data/_processed/Brera/PM25.csv")[["date", "value"]]
    #group by year and calculate average of value
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("Y")
    df = df.groupby(df["date"]).mean().reset_index()
    print(df)
    df.plot(x="date", y="value")
    plt.show()     






if __name__ == "__main__":
    main()
