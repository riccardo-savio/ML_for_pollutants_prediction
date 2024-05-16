import pandas as pd
from sodapy import Socrata
import os

# ["evzn-32bs", "cthp-zqrr", "nr8w-tj77", "g2hp-ar79"]
# 4459859

pollutants_name = {
    "Monossido di carbonio": "CO",
    "Biossido di Azoto": "NO2",
    "Ozono": "O3",
    "PM10 (SM2005)": "PM10",
    "Particelle sospese PM2.5": "PM2.5",
    "Benzene": "C6H6"
    }

meteo_name = {
    "Precipitazione": "Rain",
    "Velocità Vento": "Wind",
    "Temperatura": "Temp",
    "Umidità Relativa": "Humidity"
}



stations = {
    "air": [501, 705, 548, 528],
    'meteo': [501, 1327, 502, 620]
}

pollutants_codes = [
    'nr8w-tj77',
    'g2hp-ar79'
]


meteo_codes = {
    'Rain': ['2kar-pnuk', 'pstb-pga6'],
    'Wind': ['x9gp-z9xx', 'hu5q-68e3' ],
    'Temp':    ['d4kj-kbpj', 'w9wd-u6jh'],
    'Humidity': ['xpun-8722', '823w-fh4c'],
}


pollutants = [
    "Monossido di carbonio",
    "Biossido di Azoto",
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
        


def main():

    df = pd.read_csv(f"data/_raw/meteo_sensors.csv")
    for key, value in stations.items():
        if key == "meteo":
            for station in value:

                station_name = df[df["idstazione"] == station]['nomestazione'].tolist()[0]
                sensors = df[df["idstazione"] == station]['idsensore'].tolist()

                for sensor in sensors:

                    sensor_name = df[df["idsensore"] == sensor]['tipologia'].tolist()[0]
                    sensor_name = meteo_name.get(sensor_name, None)

                    if sensor_name is None:
                        continue

                    data = []

                    for code in meteo_codes[sensor_name]:

                        client = Socrata("www.dati.lombardia.it",
                            app_token="HgtqW8PtAIt17vyGLqsGoRyHx")
                        
                        op = 'SUM(valore)' if sensor_name == "Rain" else 'AVG(valore)'
                        
                        results = client.get_all(
                            code,
                            select=f"idsensore, date_trunc_ymd(data), {op}",
                            group="idsensore, date_trunc_ymd(data)",
                            where=f"idsensore = '{sensor}' AND stato = 'VA' AND valore >= 0",
                        )



                        sdf = pd.DataFrame(results)

                        if len(sdf) == 0:
                            continue

                        sdf.columns = ["idsensore", "data", "valore"]
                        sdf["data"] = pd.to_datetime(sdf["data"])
                        sdf.sort_values(by="data", inplace=True)
                        print(station_name, sensor_name, code, len(sdf))

                        data.append(sdf)
                    
                    s = pd.concat(data, axis=0)

                    if os.path.exists(f"data/_raw/{station_name}/meteo/") == False:
                        os.makedirs(f"data/_raw/{station_name}/meteo/")

                    s.to_csv(f"data/_raw/{station_name}/meteo/{sensor_name}.csv", mode="w", index=False)


                
    



if __name__ == "__main__":  
    main()