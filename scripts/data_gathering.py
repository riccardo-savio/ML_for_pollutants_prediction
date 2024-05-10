import pandas as pd
from sodapy import Socrata
from alive_progress import alive_bar
import os

# ["evzn-32bs", "cthp-zqrr", "nr8w-tj77", "g2hp-ar79"]
# 4459859
time_periods = [
    {"type": "rain", "code": "2kar-pnuk"},
    {"type": "wind", "code": "hu5q-68e3"},
    {"type": "temp", "code": "w9wd-u6jh"},
    {"type": "temp", "code": "d4kj-kbpj"},
    {"type": "humidity", "code": "xpun-8722"},
    {"type": "humidity", "code": "823w-fh4c"},

]
pollutants = {
    "Monossido di carbonio": {"unit":"CO", id:""},
    "Biossido di azoto": "NO2",
    "Biossido di zolfo": "SO2",
    "Ozono": "O3",
    "PM10 (SM2005)": "PM10",
    "Particelle sospese PM2.5": "PM2.5",
    "Benzene": "C6H6"
    }


def dw_ARPA_data(data: dict, id_ = ""):
    import csv, os

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

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
            ids = pd.read_csv("data/sensors/humidity/type/Umidità Relativa.csv")
            ids = ids["idsensore"].tolist()

    if not os.path.exists(f"data/sensors/{data['type']}/data/"):
        os.makedirs(f"data/sensors/{data['type']}/data/")

    with open(
        f"data/sensors/{data['type']}/data/{data['code']}.csv", "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["idsensore", "data", "valore"])
        f.close()

    for id in ids:
        if data["type"] == "rain":
            results = client.get_all(
                data["code"],
                select="idsensore, date_trunc_ymd(data), SUM(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )
        elif data["type"] == "wind":
            results = client.get_all(
                data["code"],
                select="idsensore, date_trunc_ymd(data), AVG(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )
        elif data["type"] == "temp":
            results = client.get_all(
                data["code"],
                select="idsensore, date_trunc_ymd(data), AVG(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )
        elif data["type"] == "humidity":
            results = client.get_all(
                data["code"],
                select="idsensore, date_trunc_ymd(data), AVG(valore)",
                group="idsensore, date_trunc_ymd(data)",
                where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0",
            )
        c = 0

        path = f"data/sensors/{data['type']}/data/{data['code']}.csv" if id_ == "" else f"data/sensors/{data['type']}/data/{id_}.csv"

        with open(
            path, "a", newline=""
        ) as f:
            writer = csv.writer(f)
            for item in results:
                c += 1
                if data["type"] == "rain":
                    writer.writerow(
                        [
                            item["idsensore"],
                            item["date_trunc_ymd_data"],
                            item["SUM_valore"],
                        ]
                    )
                elif data["type"] == "wind":
                    writer.writerow(
                        [
                            item["idsensore"],
                            item["date_trunc_ymd_data"],
                            item["AVG_valore"],
                        ]
                    )
                elif data["type"] == "temp":
                    writer.writerow(
                        [
                            item["idsensore"],
                            item["date_trunc_ymd_data"],
                            item["AVG_valore"],
                        ]
                    )
                elif data["type"] == "humidity":
                    writer.writerow(
                        [
                            item["idsensore"],
                            item["date_trunc_ymd_data"],
                            item["AVG_valore"],
                        ]
                    )
            f.close()


def get_ARPA_sensors(update: bool = False, type: str = ""):
    if update:
        sensors = Socrata(
            "www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx"
        )
        air_sensors_results = sensors.get("ib47-atvt", limit=10000)
        meteo_sensors_results = sensors.get("nf78-nj6b", limit=10000)
        air_sensors = pd.DataFrame.from_records(air_sensors_results).to_csv(
            "data/air_sensors.csv", index=False, mode="w"
        )

        meteo_sensors = pd.DataFrame.from_records(meteo_sensors_results).to_csv(
            "data/meteo_sensors.csv", index=False, mode="w"
        )
        air_sensors = pd.DataFrame.from_records(air_sensors_results)
        air_sensors = air_sensors[air_sensors["datastop"].isna()]
        air_sensors.to_csv("data/air_sensors.csv", index=False, mode="w")

        meteo_sensors = pd.DataFrame.from_records(meteo_sensors_results)
        meteo_sensors = meteo_sensors[meteo_sensors["datastop"].isna()]
        meteo_sensors.to_csv("data/meteo_sensors.csv", index=False, mode="w")

        sensors = {"air_sensors": air_sensors, "meteo_sensors": meteo_sensors}

        return (
            {"air_sensors": air_sensors, "meteo_sensors": meteo_sensors}
            if type == ""
            else sensors[type]
        )
    else:
        return (
            pd.read_csv("data/sensors.csv")
            if type == "air"
            else pd.read_csv("data/meteo_sensors.csv")
        )


def calculate_distance(sensor1, sensor2, sensor3, sensor4, sensor5):
    import math

    min_distances = pd.DataFrame(columns=["air", "rain", "wind", "temp", "humidity"])

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

            air_data["data"] = pd.to_datetime(air_data["data"]).dt.strftime("%Y-%m-%d")
            air_data["valore"] = pd.to_numeric(air_data["valore"])
            air_data = air_data[["data", "valore"]]

            air_data_delta = air_data.copy()
            air_data_delta["data"] = pd.to_datetime(air_data["data"]) + pd.Timedelta(
                days=1
            )
            air_data_delta["data"] = air_data_delta["data"].dt.strftime("%Y-%m-%d")

            rain_data = pd.read_csv(
                f"data/sensors/rain/data/2011-2020/{comb.loc[i, 'rain']}.csv"
            )
            rain_data["data"] = pd.to_datetime(rain_data["data"]) + pd.Timedelta(days=1)
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
                file_data["data"] = pd.to_datetime(file_data["data"]).dt.strftime("%Y-%m-%d")
                data = pd.concat([data, file_data[["data", "valore"]]])
            except pd.errors.EmptyDataError:
                os.remove(f'C:/Users/Riccardo Savio/Documents/my-project/TestDataDistribution/{file_path}')

    data = data.groupby("data").mean().reset_index()
    data.sort_values("data", inplace=True)
    data = data[data['valore'] != 0.0]
    data.to_csv(f"data/comb/air_{pollutant}.csv", index=False) 



def main():
    NO2 = pd.read_csv("data/ferno/NO2.csv")

    NO2_delta = NO2.copy()
    NO2_delta["data"] = pd.to_datetime(NO2["data"]) - pd.Timedelta(days=1)
    NO2_delta["data"] = NO2_delta["data"].dt.strftime("%Y-%m-%d")

    HUMIDITY = pd.read_csv("data/ferno/humidity.csv")
    HUMIDITY["data"] = pd.to_datetime(HUMIDITY["data"]).dt.strftime("%Y-%m-%d")
    RAIN = pd.read_csv("data/ferno/rain.csv")
    RAIN["data"] = pd.to_datetime(RAIN["data"]) + pd.Timedelta(days=1)
    RAIN["data"] = pd.to_datetime(RAIN["data"]).dt.strftime("%Y-%m-%d")

    WIND = pd.read_csv("data/ferno/wind.csv")
    WIND["data"] = pd.to_datetime(WIND["data"]).dt.strftime("%Y-%m-%d")
    TEMP = pd.read_csv("data/ferno/temp.csv")
    TEMP["data"] = pd.to_datetime(TEMP["data"]).dt.strftime("%Y-%m-%d")

    NO2_comb = (NO2
                .merge(HUMIDITY[['data', 'valore']], on="data", suffixes=("", "_HUMIDITY"))
                .merge(RAIN[['data', 'valore']], on="data", suffixes=("", "_RAIN"))
                .merge(WIND[['data', 'valore']], on="data", suffixes=("", "_WIND"))
                .merge(TEMP[['data', 'valore']], on="data", suffixes=("", "_TEMP"))
                .merge(NO2_delta[['data', 'valore']], on="data", suffixes=("", "_DELTA"))
                )[["valore", "valore_HUMIDITY", "valore_RAIN", "valore_WIND", "valore_TEMP", "valore_DELTA"]]
    NO2_comb = NO2_comb.round(5)
    NO2_comb.to_csv("data/ferno/NO2_comb.csv", index=False, mode="w")

main()
