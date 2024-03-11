import pandas as pd
from sodapy import Socrata


def export_data(sensors_dat_results, sensors_df, sensor_id):
    sdata_1_df = pd.DataFrame.from_records(sensors_dat_results)
    merged_sensor_data = pd.merge(sensors_df, sdata_1_df, on="idsensore", how="right")
    merged_sensor_data.filter(
        items=[
            "idsensore",
            "nometiposensore",
            "nomestazione",
            "provincia",
            "comune",
            "data",
            "valore",
            "unitamisura",
            "stato",
        ]
    ).to_csv(f"data/sensors/{sensor_id}.csv", index=False, mode="a", header=False)


def get_dati_lombardia():
    sensors = Socrata("www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx")
    sensors_results = sensors.get("ib47-atvt", limit=1000)
    sensors_df = pd.DataFrame.from_records(sensors_results)
    sensors_df.to_csv(path_or_buf="data/sensors.csv", index=False, mode="w")

    rslt_df = sensors_df[
        sensors_df["nometiposensore"].isin(
            ["PM10 (SM2005)", "Particelle sospese PM2.5", "Particolato Totale Sospeso"]
        )
    ]
    rslt_df.to_csv("data/sensors_f.csv", index=False, mode="w")

    # cycle through the sensors and get the data
    for time_period in ["nicp-bhqi"]:
        for index, row in rslt_df.iterrows():
            sensor_id = row["idsensore"]
            sensor_data = Socrata(
                "www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx"
            )
            sensors_dat_results = sensor_data.get(
                time_period, limit=1000000, idsensore=sensor_id, stato="VA"
            )

            if sensors_dat_results:
                print(f"{sensor_id} {time_period}")
                export_data(sensors_dat_results, rslt_df, sensor_id)


get_dati_lombardia()
