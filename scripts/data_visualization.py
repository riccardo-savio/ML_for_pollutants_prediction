from ast import parse
from matplotlib.pylab import f
import matplotlib.pyplot as plt
from pandas import DataFrame
from pyparsing import col
from seaborn import kdeplot

from data_preprocessing import data_distribution, log_transform, merge_datasets, remove_outliers_iqr, remove_outliers_zscore
from data_study import study


pollutants = {
    "Monossido di carbonio": "CO",
    "Biossido di azoto": "NO2",
    "Biossido di zolfo": "SO2",
    "Ozono": "O3",
    "PM10 (SM2005)": "PM10",
    "Particelle sospese PM2.5": "PM2.5",
    "Benzene": "C6H6"
    }

sensor_names = {
}

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
    axs[len(axs) - 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    for label in axs[len(axs) - 1].get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

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

    df = reduce(lambda df1,df2: pd.merge(df1 ,df2, on='data', suffixes=('', '_')), pollutants)

    df.drop("data", axis=1, inplace=True)
    df.columns = columns

    sns.heatmap(df.corr(), annot=True, mask=~np.tri(df.corr().shape[1], k=-1, dtype=bool), cmap="coolwarm")
    plt.show()
    
def pollutant_meteo_correlation(pollutant: DataFrame, meteo: list[DataFrame], x_label: list[str], y_label: list[str]):
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

    sns.heatmap(merged_data.corr(), annot=True, mask=~np.tri(merged_data.corr().shape[1], k=-1, dtype=bool), cmap="coolwarm")
    plt.show()

def plot_pollutant_meteo_rel(pollutants: list[DataFrame], meteo: list[DataFrame], x_labels: list[str], y_labels: list[str]):

    import pandas as pd
    import numpy as np

    fig, axs = plt.subplots(len(pollutants), len(meteo), squeeze=False)

    for i in range(len(pollutants)):
        
        pollutant = pollutants[i][['data', 'valore']]
        pollutant['data'] = pd.to_datetime(pollutant['data'])

        for j in range(len(meteo)):

            if j == 0:
                axs[i][j].set_ylabel(y_labels[i])

            if i == len(pollutants) - 1:
                axs[i][j].set_xlabel(x_labels[j])

            meteo_s = meteo[j][["data", "valore"]]
            meteo_s["data"] = pd.to_datetime(meteo_s["data"])

            if x_labels[j] == "RN" or "WS":
                meteo_s["data"] = pd.to_datetime(meteo_s["data"]) + pd.Timedelta(days=1)

            

            
            meteo_s = meteo_s[["data", "valore"]].merge(pollutant, on="data", suffixes=(f"_{x_labels[j]}", f"_{y_labels[i]}"))
            meteo_s.drop("data", axis=1, inplace=True)
            meteo_s.columns = [x_labels[j], y_labels[i]]

            axs[i][j].scatter(x_labels[j], y_labels[i], data=meteo_s)
            axs[i][j].grid(True)






    plt.show()
    
def time_series_histogram(data: DataFrame, time_unit: str, pollutant: str): 
    import pandas as pd
    import seaborn as sns

    data.columns = ["data", "valore"]
    data["data"] = pd.to_datetime(data["data"]).dt.strftime(time_unit)
    data = data.groupby("data").mean().reset_index()[['data', 'valore']]

    if time_unit == "%m":
        data["data"] = pd.to_datetime(data["data"], format='%m').dt.strftime("%B")

    plt.title(f'Media annuale di {pollutant}')
    plt.bar(data["data"], data["valore"], align='center', alpha=0.5, color="blue")
    plt.xticks(rotation=90)
    plt.ylabel('Valore medio')
    plt.tight_layout()



def main():

    import os
    import pandas as pd
    import seaborn as sns
    import numpy as np


    xs = []
    columns = []
    """ for file in os.listdir(f"data/_processed/Brera/"):
        df = pd.read_csv(f"data/_processed/Brera/{file}", parse_dates=['data'])
        df['Day'] = df['data'].dt.day_of_year
        df['Month'] = df['data'].dt.month

        def encode(data, col, max_val):
            data['Sin_'+col] = np.sin(2 * np.pi * data[col]/max_val)
            data['Cos_'+col] = np.cos(2 * np.pi * data[col]/max_val)
            return data
        
        df = encode(df, 'Day', 365)
        df = encode(df, 'Month', 12)

        df.to_csv(f"data/_processed/Brera/sin_cos_{file}", index=False)
        sns.heatmap(df.corr(), annot=True, mask=~np.tri(df.corr().shape[1], k=-1, dtype=bool), cmap="coolwarm")
        plt.tight_layout()
        plt.show()
        ys = []
        rows = []
        for file in os.listdir(f"data/_raw/Brera/meteo/"):
        df = pd.read_csv(f"data/_raw/Brera/meteo/{file}")[['data', 'valore']]
        ys.append(df)
        rows.append(file.split(".")[0]) """


    df = study()
    print(df)
    df.hist(column=['Model'], bins=20, figsize=(20, 15))
    plt.show()

    













if __name__ == "__main__":
    main()