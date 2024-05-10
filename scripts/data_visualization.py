import matplotlib.pyplot as plt


pollutants = {
    "Monossido di carbonio": "CO",
    "Biossido di azoto": "NO2",
    "Biossido di zolfo": "SO2",
    "Ozono": "O3",
    "PM10 (SM2005)": "PM10",
    "Particelle sospese PM2.5": "PM2.5",
    "Benzene": "C6H6"
    }

def pollutants_plot():
    import pandas as pd
    import matplotlib.dates as mdates

    
    fig, axs = plt.subplots(7, 1, figsize=(20, 15))

    for i in range(len(axs)):
        print()
        data = pd.read_csv(f"data/comb/air_{list(pollutants.keys())[i]}.csv")
        data["data"] = pd.to_datetime(data["data"])
        data = data[data["data"] > "2000-01-01"]
        
        axs[i].plot("data", "valore", data=data)
        axs[i].xaxis.set_major_locator(mdates.YearLocator(2))
        axs[i].xaxis.set_visible(False)
        axs[i].set_ylabel(f'{pollutants[list(pollutants.keys())[i]]}')
        axs[i].grid(True)

    axs[len(axs) - 1].xaxis.set_visible(True)
    axs[len(axs) - 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    for label in axs[len(axs) - 1].get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.show()

def pollutants_heatmap():
    import seaborn as sns
    import pandas as pd
    import numpy as np

    CO = pd.read_csv("data/comb/air_Monossido di carbonio.csv")
    NO2 = pd.read_csv("data/comb/air_Biossido di azoto.csv")
    SO2 = pd.read_csv("data/comb/air_Biossido di zolfo.csv")
    NO = pd.read_csv("data/comb/air_Monossido di Azoto.csv")
    O3 = pd.read_csv("data/comb/air_Ozono.csv")
    PM10 = pd.read_csv("data/comb/air_PM10 (SM2005).csv")
    PM25 = pd.read_csv("data/comb/air_Particelle sospese PM2.5.csv")
    C6H6 = pd.read_csv("data/comb/air_Benzene.csv")
    

    CO["data"] = pd.to_datetime(CO["data"])
    NO2["data"] = pd.to_datetime(NO2["data"])
    SO2["data"] = pd.to_datetime(SO2["data"])
    O3["data"] = pd.to_datetime(O3["data"])
    PM10["data"] = pd.to_datetime(PM10["data"])
    PM25["data"] = pd.to_datetime(PM25["data"])
    C6H6["data"] = pd.to_datetime(C6H6["data"])

    combined_data = (
        CO.merge(NO2, on="data", suffixes=("", "_NO2"))
        .merge(SO2, on="data", suffixes=("", "_SO2"))
        .merge(O3, on="data", suffixes=("", "_O3"))
        .merge(PM10, on="data", suffixes=("", "_PM10"))
        .merge(PM25, on="data", suffixes=("", "_PM25"))
        .merge(C6H6, on="data", suffixes=("", "_C6H6"))
    )

    combined_data.drop("data", axis=1, inplace=True)
    combined_data.columns = ["CO", "NO2", "SO2", "O3", "PM10", "PM2.5", "C6H6", "AMMONIACA", "ARSENICO"]

    sns.heatmap(combined_data.corr(), annot=True, mask=~np.tri(combined_data.corr().shape[1], k=-1, dtype=bool))
    plt.show()

def pollutant_meteo_correlation():
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

def plot_pollutant_meteo():
    import pandas as pd
    import matplotlib.dates as mdates
    import datetime


    PM10 = pd.read_csv("data/ferno/PM10.csv")
    HUMIDITY = pd.read_csv("data/ferno/humidity.csv")
    RAIN = pd.read_csv("data/ferno/rain.csv")
    WIND = pd.read_csv("data/ferno/wind.csv")
    TEMP = pd.read_csv("data/ferno/temp.csv")
    

    HUMIDITY = HUMIDITY.merge(PM10, on="data", suffixes=("", "_PM10"))[['data', 'valore']]
    RAIN = RAIN.merge(PM10, on="data", suffixes=("", "_PM10"))[["data", "valore"]]
    WIND = WIND.merge(PM10, on="data", suffixes=("", "_PM10"))[["data", "valore"]]
    TEMP = TEMP.merge(PM10, on="data", suffixes=("", "_PM10"))[["data", "valore"]]
    PM10 = PM10[["data", "valore"]]

    PM10['data'] = pd.to_datetime(PM10['data'])
    HUMIDITY['data'] = pd.to_datetime(HUMIDITY['data'])
    RAIN['data'] = pd.to_datetime(RAIN['data'])
    WIND['data'] = pd.to_datetime(WIND['data'])
    TEMP['data'] = pd.to_datetime(TEMP['data'])
    

    fig, axs = plt.subplots(5, 1, figsize=(20, 15))



    axs[0].plot("data", "valore", data=PM10)
    axs[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel("PM10")
    axs[0].grid(True)

    axs[1].plot("data", "valore", data=RAIN)
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_visible(False)
    axs[1].set_ylabel("RAIN")
    axs[1].grid(True)

    axs[2].plot("data", "valore", data=WIND)
    axs[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[2].xaxis.set_visible(False)
    axs[2].set_ylabel("WIND")
    axs[2].grid(True)

    axs[3].plot("data", "valore", data=TEMP)
    axs[3].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[3].xaxis.set_visible(False)
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[3].set_ylabel("TEMP")
    axs[3].grid(True)

    axs[4].plot("data", "valore", data=HUMIDITY)
    axs[4].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[4].xaxis.set_visible(True)
    axs[4].set_ylabel("HUMIDITY")
    axs[4].grid(True)

    for label in axs[4].get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.show()

def plot_pollutant_meteo_rel():

    import pandas as pd
    import matplotlib.dates as mdates

    HUMIDITY = pd.read_csv("data/ferno/humidity.csv")
    RAIN = pd.read_csv("data/ferno/rain.csv")
    WIND = pd.read_csv("data/ferno/wind.csv")
    TEMP = pd.read_csv("data/ferno/temp.csv")
    PM10 = pd.read_csv("data/ferno/PM10.csv")
    NO2 = pd.read_csv("data/ferno/NO2.csv")
    O3 = pd.read_csv("data/ferno/O3.csv")

    PM10_RAIN = PM10.merge(RAIN, on="data", suffixes=("", "_rain"))
    PM10_WIND = PM10.merge(WIND, on="data", suffixes=("", "_wind"))
    PM10_TEMP = PM10.merge(TEMP, on="data", suffixes=("", "_temp"))
    PM10_HUMIDITY = PM10.merge(HUMIDITY, on="data", suffixes=("", "_humidity"))

    NO2_RAIN = NO2.merge(RAIN, on="data", suffixes=("", "_rain"))
    NO2_WIND = NO2.merge(WIND, on="data", suffixes=("", "_wind"))
    NO2_TEMP = NO2.merge(TEMP, on="data", suffixes=("", "_temp"))
    NO2_HUMIDITY = NO2.merge(HUMIDITY, on="data", suffixes=("", "_humidity"))

    O3_RAIN = O3.merge(RAIN, on="data", suffixes=("", "_rain"))
    O3_WIND = O3.merge(WIND, on="data", suffixes=("", "_wind"))
    O3_TEMP = O3.merge(TEMP, on="data", suffixes=("", "_temp"))
    O3_HUMIDITY = O3.merge(HUMIDITY, on="data", suffixes=("", "_humidity"))

    fig, axs = plt.subplots(3, 4, figsize=(20, 3))

    axs[0][0].scatter("valore_rain", "valore", data=PM10_RAIN)
    axs[0][0].set_ylabel("PM10")
    axs[0][0].grid(True)

    axs[0][1].scatter("valore_wind", "valore", data=PM10_WIND)
    axs[0][1].grid(True)

    axs[0][2].scatter("valore_temp", "valore", data=PM10_TEMP)
    axs[0][2].grid(True)

    axs[0][3].scatter("valore_humidity", "valore", data=PM10_HUMIDITY)
    axs[0][3].grid(True)

    axs[1][0].scatter("valore_rain", "valore", data=NO2_RAIN)
    axs[1][0].set_ylabel("NO2")
    axs[1][0].grid(True)

    axs[1][1].scatter("valore_wind", "valore", data=NO2_WIND)
    axs[1][1].grid(True)

    axs[1][2].scatter("valore_temp", "valore", data=NO2_TEMP)
    axs[1][2].grid(True)

    axs[1][3].scatter("valore_humidity", "valore", data=NO2_HUMIDITY)
    axs[1][3].grid(True)

    axs[2][0].scatter("valore_rain", "valore", data=O3_RAIN)
    axs[2][0].set_ylabel("O3")
    axs[2][0].set_xlabel("RAIN mm")
    axs[2][0].grid(True)

    axs[2][1].scatter("valore_wind", "valore", data=O3_WIND)
    axs[2][1].set_xlabel("WIND m/s")
    axs[2][1].grid(True)

    axs[2][2].scatter("valore_temp", "valore", data=O3_TEMP)
    axs[2][2].set_xlabel("TEMP deg")
    axs[2][2].grid(True)

    axs[2][3].scatter("valore_humidity", "valore", data=O3_HUMIDITY)
    axs[2][3].set_xlabel("HUMIDITY %")
    axs[2][3].grid(True)




    plt.show()
    
plot_pollutant_meteo_rel()