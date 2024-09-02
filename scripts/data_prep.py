from data_gathering import main as dw_data
from data_gathering import pollutants_name, meteo_name


def main():
    import os
    import pandas as pd

    if not os.path.exists("data/_raw"):
        dw_data()

    if not os.path.exists("data/_processed"):
        os.makedirs("data/_processed")
    if not os.path.exists("data/_processed/Brera"):
        os.makedirs("data/_processed/Brera")
    if not os.path.exists("data/_processed/Citta Studi"):
        os.makedirs("data/_processed/Citta Studi")
    if not os.path.exists("data/_processed/Montalbino"):
        os.makedirs("data/_processed/Montalbino")

    air_sensors = pd.read_csv("data/_raw/air_sensors.csv")
    meteo_sensors = pd.read_csv("data/_raw/meteo_sensors.csv")

    for loc in ["Brera", "Citta Studi", "Montalbino"]:
        for file in os.listdir(f"data/_raw/{loc}/air/"):
            name = air_sensors[air_sensors["idsensore"] == int(file.split(".")[0])][
                "nometiposensore"
            ].values[0]
            name = pollutants_name[name]
            data = pd.read_csv(f"data/_raw/{loc}/air/{file}")

            for measure in os.listdir(f"data/_raw/{loc}/meteo/"):
                meteo = pd.read_csv(f"data/_raw/{loc}/meteo/{measure}")[
                    ["date", "value"]
                ]
                name_m = meteo_sensors[
                    meteo_sensors["idsensore"] == int(measure.split(".")[0])
                ]["tipologia"].values[0]
                meteo.columns = ["date", f"{meteo_name[name_m]}"]
                data = data.merge(meteo, on="date", how="inner")

            data["date"] = pd.to_datetime(data["date"])
            data.drop("idsensore", axis=1, inplace=True)
            data.drop_duplicates(inplace=True)
            data.to_csv(f"data/_processed/{loc}/{name}.csv", index=False)


    df = pd.read_csv("data/_processed/Brera/PM10.csv")
    print(df[['value', 'Rain', 'Temp', 'Humidity', 'Wind']].corr())

if __name__ == "__main__":
    main()
