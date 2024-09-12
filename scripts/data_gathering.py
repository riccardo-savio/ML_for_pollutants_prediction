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
    "Particelle sospese PM2.5": "PM25",
    "Benzene": "C6H6",
}

meteo_name = {
    "Precipitazione": "Rain",
    "Velocità Vento": "Wind",
    "Temperatura": "Temp",
    "Umidità Relativa": "Humidity",
}

stations = {
    "air": {
        501: {"sensors": [20429, 17127, 5504, 20529], "name": "Montalbino"},
        705: {
            "sensors": [10273, 10283, 10282, 10279, 17126],
            "name": "Citta Studi",
        },
        548: {"sensors": [6057, 5551, 10320, 17122, 5725], "name": "Brera"},
    },
    "meteo": {
        501: {"sensors": [6597, 19019, 5911, 14121], "name": "Montalbino"},
        502: {"sensors": [5909, 6179, 5908, 19242], "name": "Citta Studi"},
        620: {"sensors": [6174, 19007, 19373, 5897], "name": "Brera"},
    },
}

pollutants_codes = ["nr8w-tj77", "g2hp-ar79"]

meteo_codes = {
    "Rain": ["2kar-pnuk", "pstb-pga6"],
    "Wind": ["x9gp-z9xx", "hu5q-68e3"],
    "Temp": ["d4kj-kbpj", "w9wd-u6jh"],
    "Humidity": ["xpun-8722", "823w-fh4c"],
}


def dw_ARPA_air_sensors():
    """Download air sensors data from ARPA Lombardia"""

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    print("Downloading air sensors data...")

    results = client.get_all(
        "ib47-atvt",
    )
    df = pd.DataFrame(results)
    if os.path.exists("data/_raw/") is False:
        os.makedirs("data/_raw/")
    df.to_csv("data/_raw/air_sensors.csv", mode="w", index=False)

    print("Air sensors data downloaded at data/_raw/air_sensors.csv")


def dw_ARPA_meteo_sensors():
    """Download meteo sensors data from ARPA Lombardia"""

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    print("Downloading meteo sensors data...")

    results = client.get_all(
        "nf78-nj6b",
    )
    df = pd.DataFrame(results)
    if os.path.exists("data/_raw/") is False:
        os.makedirs("data/_raw/")
    df.to_csv("data/_raw/meteo_sensors.csv", mode="w", index=False)

    print("Meteo sensors data downloaded at data/_raw/meteo_sensors.csv")


def dw_ARPA_pollutants():
    """Download pollutants data from ARPA Lombardia
    The data is grouped by sensor and pollutant
     The selected sensors are the following ones:
     - Montalbino: 20429, 17127, 5504, 20529
     - Citta Studi: 10273, 10283, 10282, 10279, 17126
     - Brera: 6057, 5551, 10320, 17122, 5531, 5725, 6956
    """

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    print("Downloading pollutants data...")

    for station in stations["air"].keys():
        for id in stations["air"][station]["sensors"]:
            df = pd.DataFrame()
            for pollutant_code in pollutants_codes:
                results = client.get_all(
                    pollutant_code,
                    select="idsensore, date_trunc_ymd(data), AVG(valore)",
                    group="idsensore, date_trunc_ymd(data)",
                    where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0 ",
                )

                if (
                    os.path.exists(f"data/_raw/{stations['air'][station]['name']}/air/")
                    is False
                ):
                    os.makedirs(f"data/_raw/{stations['air'][station]['name']}/air/")

                # add the results to the dataframe
                df = pd.concat([df, pd.DataFrame(results)])

            if len(df) == 0:
                continue

            path = f"data/_raw/{stations['air'][station]['name']}/air/{id}.csv"
            df.columns = ["idsensore", "date", "value"]
            df = df.sort_values(by="date")
            df["value"] = df["value"].astype(float).round(4)
            df.to_csv(path, mode="w", index=False)

    print("Pollutants data downloaded at data/_raw/*stations/air")


def dw_ARPA_meteo():
    """Download meteo data from ARPA Lombardia
    The data is grouped by sensor and meteo type
    The selected sensors are the following ones:
    - Montalbino: 6597, 19019, 5911, 14121
    - Citta Studi: 5909, 6179, 5908, 19242
    - Brera: 6174, 19007, 19373, 5897
    """

    client = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")

    print("Downloading meteo data...")

    for meteo_type in meteo_codes.keys():
        for station in stations["meteo"].keys():
            for id in stations["meteo"][station]["sensors"]:
                df = pd.DataFrame()

                op = "SUM(valore)" if meteo_type == "Rain" else "AVG(valore)"

                for code in meteo_codes[meteo_type]:
                    results = client.get_all(
                        code,
                        select=f"idsensore, date_trunc_ymd(data), {op}",
                        group="idsensore, date_trunc_ymd(data)",
                        where=f"idsensore = '{id}' AND stato = 'VA' AND valore >= 0 ",
                    )

                    if (
                        os.path.exists(
                            f"data/_raw/{stations['meteo'][station]['name']}/meteo/"
                        )
                        is False
                    ):
                        os.makedirs(
                            f"data/_raw/{stations['meteo'][station]['name']}/meteo/"
                        )

                    df = pd.concat([df, pd.DataFrame(results)])

                if len(df) == 0:
                    continue

                path = f"data/_raw/{stations['meteo'][station]['name']}/meteo/{id}.csv"
                df.columns = ["idsensore", "date", "value"]
                df = df.sort_values(by="date")
                df["value"] = df["value"].astype(float).round(4)
                df.to_csv(path, mode="w", index=False)

    print("Meteo data downloaded at data/_raw/*stations/meteo")


def main():
    dw_ARPA_air_sensors()
    dw_ARPA_meteo_sensors()
    dw_ARPA_pollutants()
    dw_ARPA_meteo()


if __name__ == "__main__":
    main()
