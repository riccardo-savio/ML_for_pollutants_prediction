import json
import csv

params = dict(json.load(open("data/stats/best_params.json")))

data = []

for scenario in params.keys():
    for location in params[scenario].keys():
        for pollutant in params[scenario][location].keys():
            for model in params[scenario][location][pollutant].keys():
                par = params[scenario][location][pollutant][model]
                par = ", ".join([f"{k}: {v}" for k, v in par.items()])
                data.append([scenario, location, pollutant, model, par])

with open("data/stats/best_params.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "location", "pollutant", "model", "params"])
    writer.writerows(data)