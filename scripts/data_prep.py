from data_gathering import main as dw_data
from data_gathering import pollutants_name, meteo_name
import pandas as pd

def prep_folders():
    import os
    if not os.path.exists("data/_processed"):
        os.makedirs("data/_processed")
    if not os.path.exists("data/_processed/Brera"):
        os.makedirs("data/_processed/Brera")
    if not os.path.exists("data/_processed/Citta Studi"):
        os.makedirs("data/_processed/Citta Studi")
    if not os.path.exists("data/_processed/Montalbino"):
        os.makedirs("data/_processed/Montalbino")

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using the interquartile range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to remove outliers.
    column : str
        The name of the column in the DataFrame from which to remove outliers.

    Returns
    -------
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    # Calculate the first quartile (Q1)
    Q1 = df[column].quantile(0.25)
    # Calculate the third quartile (Q3)
    Q3 = df[column].quantile(0.75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Calculate the lower bound
    lower_bound = Q1 - 1.5 * IQR
    # Calculate the upper bound
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers from the DataFrame
    df_iqr = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_iqr



def merge_data():

    import pandas as pd
    import os

    air_sensors = pd.read_csv("data/_raw/air_sensors.csv")
    meteo_sensors = pd.read_csv("data/_raw/meteo_sensors.csv")

    for loc in ["Brera", "Citta Studi", "Montalbino"]:
        for file in os.listdir(f"data/_raw/{loc}/air/"):
            name = air_sensors[air_sensors["idsensore"] == int(file.split(".")[0])][
                "nometiposensore"
            ].values[0]
            name = pollutants_name[name]
            data = pd.read_csv(f"data/_raw/{loc}/air/{file}")
            data = remove_outliers_iqr(data, "value")


            for measure in os.listdir(f"data/_raw/{loc}/meteo/"):
                meteo = pd.read_csv(f"data/_raw/{loc}/meteo/{measure}")[
                    ["date", "value"]
                ]
                name_m = meteo_sensors[
                    meteo_sensors["idsensore"] == int(measure.split(".")[0])
                ]["tipologia"].values[0]
                meteo.columns = ["date", f"{meteo_name[name_m]}"]
                data = data.merge(meteo, on="date", how="inner")

            data["value_dayb"] = data["value"].shift(1)
            
            data["date"] = pd.to_datetime(data["date"])
            data.drop("idsensore", axis=1, inplace=True)
            data.dropna(inplace=True)
            data.drop_duplicates(inplace=True)
            data.to_csv(f"data/_processed/{loc}/{name}.csv", index=False)
    return data



def main():

    prep_folders()
    merge_data()

    

if __name__ == "__main__":
    main()
