import pandas as pd
from sodapy import Socrata
import os
# ["evzn-32bs", "cthp-zqrr", "nr8w-tj77", "g2hp-ar79"]

time_periods = {
    "evzn-32bs": "1968-1999",
    "cthp-zqrr": "2000-2009",
    "nr8w-tj77": "2010-2017",
    "g2hp-ar79": "2018-2022",
}

def get_sensor_data(sensor_id: str, time_period: str, nometiposensore: str,  update: bool = False ):
    if update:
        try:
            sensor_data = Socrata(
                "www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx"
            )
            sensors_data_results = sensor_data.get(
                    time_period, limit=1000000, idsensore=sensor_id, stato="VA"
                )
            sensor_data_df = pd.DataFrame.from_records(sensors_data_results)
            
            if not os.path.exists(f"data/sensors/{time_periods[time_period]}/{nometiposensore}/"):
                os.makedirs(f"data/sensors/{time_periods[time_period]}/{nometiposensore}/")
                
            sensor_data_df.to_csv(
                f"data/sensors/{time_periods[time_period]}/{nometiposensore}/{sensor_id}.csv",
                index=False,
                mode="w",
            )
        except:
            sensor_data_df = pd.DataFrame()
            
        return sensor_data_df
    else:
        try:
            return pd.read_csv(f"data/sensors/{time_periods[time_period]}/{nometiposensore}/{sensor_id}.csv")
        except:
            return pd.DataFrame()

def get_ARPA_sensors(update: bool = False):
    if update:
        sensors = Socrata(
            "www.dati.lombardia.it", app_token="HgtqW8PtAIt17vyGLqsGoRyHx"
        )
        sensors_results = sensors.get("ib47-atvt", limit=10000)
        sensors_df = pd.DataFrame.from_records(sensors_results)
        sensors_df.to_csv("data/sensors.csv", index=False, mode="w")
        return sensors_df
    else:
        return pd.read_csv("data/sensors.csv")


def main():
    sensosr_df = get_ARPA_sensors(update=True)
    for sensor_id in sensosr_df["idsensore"]:
        for time_period in time_periods:
            get_sensor_data(sensor_id, time_period, sensosr_df[sensosr_df["idsensore"] == sensor_id]["nometiposensore"].iloc[0], update=True)
            print(f"Sensor {sensor_id} for time period {time_periods[time_period]} has been updated")

main()



