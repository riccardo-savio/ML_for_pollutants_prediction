import requests
import json
import pandas as pd
from sodapy import Socrata
import csv

def get_dati_lombardia():

    sensors = Socrata("www.dati.lombardia.it", None)
    sensors_results = sensors.get("ib47-atvt", limit=1000)
    sensors_df = pd.DataFrame.from_records(sensors_results)
    sensors_df.to_csv('data/sensors.csv', index=False)

    sensor_data = Socrata("www.dati.lombardia.it", None)
    sensors_dat_results = sensor_data.get("nicp-bhqi", limit=3000000)
    sensor_data_df = pd.DataFrame.from_records(sensors_dat_results)
    sensor_data_df.to_csv('data/sensors_data.csv', index=False)

    df3 = pd.merge(sensors_df, sensor_data_df,on='idsensore',how='right')
    df4 = df3.filter(items=['idsensore', 'nometiposensore', 'nomestazione', 'provincia', 'comune', 'data', 'valore', 'unitamisura', 'stato'])
    df4.columns = ['idsensore', 'nometiposensore', 'nomestazione', 'provincia', 'comune', 'data', 'valore', 'unita_misura', 'stato']
    df4.to_csv('data/merged_sensors.csv', index=False)

get_dati_lombardia()