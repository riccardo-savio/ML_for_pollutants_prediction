import dis
from operator import ne
from tokenize import group
from numpy import mean
import pandas as pd
from sodapy import Socrata
from alive_progress import alive_bar
import os

# ["evzn-32bs", "cthp-zqrr", "nr8w-tj77", "g2hp-ar79"]
# 4459859

sensors = {
    "air": 'ib47-atvt',
    'meteo': 'nf78-nj6b'
}

meteo_codes = {
    'Precipitazione': 'pstb-pga6',
    'Velocità Vento': 'hu5q-68e3',
    'Temperatura': 'w9wd-u6jh',
    'Umidità Relativa': '823w-fh4c',
    'Direzione Vento': 'purm-rsjf',
}
pollutants = [
    "Monossido di carbonio",
    "Biossido di azoto",
    "Biossido di zolfo",
    "Ozono",
    "PM10 (SM2005)",
    "Particelle sospese PM2.5",
    "Benzene"
]


def dw_ARPA_data(data: dict, id_=""):
    import csv
    import os

    client = Socrata("www.dati.lombardia.it",
                     app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    if id_ != "":
        ids = [id_]
    else:
        if data["type"] == "rain":
            ids = pd.read_csv("data/sensors/rain/type/Precipitazione.csv")
            ids = ids["idsensore"].tolist()
        elif data["type"] == "wind":
            ids = pd.read_csv("data/sensors/wind/type/Velocità Vento.csv")
            ids = ids["idsensore"].tolist()
        elif data["type"] == "temp":
            ids = pd.read_csv("data/sensors/temp/type/Temperatura.csv")
            ids = ids["idsensore"].tolist()
        elif data["type"] == "humidity":
            ids = pd.read_csv(
                "data/sensors/humidity/type/Umidità Relativa.csv")
            ids = ids["idsensore"].tolist()


    for id in ids:

        results = client.get_all(
            data["code"],
            select="idsensore, data, valore",
            where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
        )

        if os.path.exists(f"data/sensors/{data['type']}/single") == False:
            os.makedirs(f"data/sensors/{data['type']}/single")
        path = f"data/sensors/{data['type']}/{data['code']}.csv" if id_ == "" else f"data/sensors/{data['type']}/single/{id_}.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(path, mode="w", index=False)
        

    
            
def calculate_distance(sensor1, sensor2, sensor3, sensor4, sensor5):
    import math

    min_distances = pd.DataFrame(
        columns=["air", "rain", "wind", "temp", "humidity"])

    for i in range(len(sensor1)):
        min1 = 100
        min2 = 100
        min3 = 100
        min4 = 100
        min_id1 = ""
        min_id2 = ""
        min_id3 = ""
        min_id4 = ""

        lat1 = sensor1.loc[i, "lat"]
        lon1 = sensor1.loc[i, "lng"]

        for j in range(len(sensor2)):
            lat2 = sensor2.loc[j, "lat"]
            lon2 = sensor2.loc[j, "lng"]
            distance1 = math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
            if distance1 < min1:
                min1 = distance1
                min_id1 = sensor2.loc[j, "idsensore"]

        for k in range(len(sensor3)):
            lat3 = sensor3.loc[k, "lat"]
            lon3 = sensor3.loc[k, "lng"]
            distance2 = math.sqrt((lat3 - lat1) ** 2 + (lon3 - lon1) ** 2)
            if distance2 < min2:
                min2 = distance2
                min_id2 = sensor3.loc[k, "idsensore"]

        for l in range(len(sensor4)):
            lat4 = float(sensor4.loc[l, "lat"])
            lon4 = float(sensor4.loc[l, "lng"])
            distance3 = math.sqrt((lat4 - lat1) ** 2 + (lon4 - lon1) ** 2)
            if distance3 < min3:
                min3 = distance3
                min_id3 = sensor4.loc[l, "idsensore"]

        for m in range(len(sensor5)):
            lat5 = float(sensor5.loc[m, "lat"])
            lon5 = float(sensor5.loc[m, "lng"])
            distance4 = math.sqrt((lat5 - lat1) ** 2 + (lon5 - lon1) ** 2)
            if distance4 < min4:
                min4 = distance4
                min_id4 = sensor5.loc[m, "idsensore"]

        min_distances = min_distances._append(
            {
                "air": sensor1.loc[i, "idsensore"],
                "rain": min_id1,
                "wind": min_id2,
                "temp": min_id3,
                "humidity": min_id4,
            },
            ignore_index=True,
        )
    return min_distances


def combine_data(comb: pd.DataFrame):
    for i in range(len(comb)):
        try:
            air_data = pd.read_csv(
                f"data/sensors/air/data/2010-2017/PM10 (SM2005)/{comb.loc[i, 'air']}.csv"
            )

            air_data["data"] = pd.to_datetime(
                air_data["data"]).dt.strftime("%Y-%m-%d")
            air_data["valore"] = pd.to_numeric(air_data["valore"])
            air_data = air_data[["data", "valore"]]

            air_data_delta = air_data.copy()
            air_data_delta["data"] = pd.to_datetime(air_data["data"]) + pd.Timedelta(
                days=1
            )
            air_data_delta["data"] = air_data_delta["data"].dt.strftime(
                "%Y-%m-%d")

            rain_data = pd.read_csv(
                f"data/sensors/rain/data/2011-2020/{comb.loc[i, 'rain']}.csv"
            )
            rain_data["data"] = pd.to_datetime(
                rain_data["data"]) + pd.Timedelta(days=1)
            rain_data["data"] = rain_data["data"].dt.strftime("%Y-%m-%d")
            rain_data = rain_data[["data", "valore"]]

            wind_data = pd.read_csv(
                f"data/sensors/wind/data/2011-2020/{comb.loc[i, 'wind']}.csv"
            )
            wind_data = wind_data[["data", "valore"]]

            temp_data = pd.read_csv(
                f"data/sensors/temp/data/2011-2020/{comb.loc[i, 'temp']}.csv"
            )
            temp_data = temp_data[["data", "valore"]]

            humidity_data = pd.read_csv(
                f"data/sensors/humidity/data/2011-2020/{comb.loc[i, 'humidity']}.csv"
            )
            humidity_data = humidity_data[["data", "valore"]]

            combined_data = (
                air_data.merge(
                    rain_data, on="data", how="inner", suffixes=("_air", "_rain")
                )
                .merge(wind_data, on="data", how="inner", suffixes=("_rain", "_wind"))
                .merge(temp_data, on="data", how="inner", suffixes=("_wind", "_temp"))
                .merge(
                    humidity_data,
                    on="data",
                    how="inner",
                    suffixes=("_temp", "_humidity"),
                )
                .merge(
                    air_data_delta,
                    on="data",
                    how="inner",
                    suffixes=("_humidity", "_delta"),
                )
            )

            combined_data.drop_duplicates(inplace=True)
            combined_data.to_csv(
                f"data/comb/air_rain_wind/{comb.loc[i, 'air']}.csv",
                index=False,
                mode="w",
            )

        except pd.errors.EmptyDataError:
            continue
        except FileNotFoundError as e:
            continue


def set_total_pollutant(pollutant: str):
    import pandas as pd
    data = pd.DataFrame(columns=["data", "valore"])

    folders = os.listdir("data/sensors/air/data")
    for folder in folders:
        folder_path = f"data/sensors/air/data/{folder}/{pollutant}"
        files = os.listdir(folder_path)
        for file in files:
            try:
                file_path = f"{folder_path}/{file}"
                file_data = pd.read_csv(file_path)
                file_data["data"] = pd.to_datetime(
                    file_data["data"]).dt.strftime("%Y-%m-%d")
                data = pd.concat([data, file_data[["data", "valore"]]])
            except pd.errors.EmptyDataError:
                os.remove(
                    f'C:/Users/Riccardo Savio/Documents/my-project/TestDataDistribution/{file_path}')

    data = data.groupby("data").mean().reset_index()
    data.sort_values("data", inplace=True)
    data = data[data['valore'] != 0.0]
    data.to_csv(f"data/comb/air_{pollutant}.csv", index=False)


def main():

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    df = pd.read_csv("data/citta_studi/meteo_sensors.csv")

    for id in df["idsensore"]:
        type = df.loc[df["idsensore"] == id, "tipologia"].values[0]
        if type == "Precipitazione":
            results = client.get_all(
                meteo_codes[type],
                select="idsensore, date_trunc_ymd(data), SUM(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )
        else:
            results = client.get_all(
                meteo_codes[type],
                select="idsensore, date_trunc_ymd(data), AVG(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )

        if os.path.exists(f"data/citta_studi/meteo/{type}") == False:
            os.makedirs(f"data/citta_studi/meteo/{type}")

        res = pd.DataFrame(results)
        res.columns = ["idsensore", "data", "valore"]
        res['data'] = pd.to_datetime(res['data']).dt.strftime("%Y-%m-%d")
        res.sort_values("data", inplace=True)
        res.to_csv(f"data/citta_studi/meteo/{type}/{id}.csv", mode="w", index=False)



main()