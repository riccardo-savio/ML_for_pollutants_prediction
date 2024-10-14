from models import GradientBoosting
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/_processed/Brera/PM25.csv", parse_dates=["date"]).set_index("date").to_period("D")

r2s = {}
columns = pd.DataFrame()
for i in range(0, 185):
    if i > 0:
        columns[f"lag_{i}"] = df["value"].shift(i)

    df1 = pd.concat([df, columns], axis=1)

    df1.dropna(inplace=True)

    train_data, test_data = train_test_split(df1, test_size=0.2, random_state=42)

    features, target = df.columns.difference(["date", "value"]), 'value'
    model = GradientBoosting.GradientBoostingModel(best_params={'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100})

    model.train(train_data[features], train_data[target])

    _, _, R2 = model.evaluate(test_data[features], test_data[target])
    r2s[f"lag_{i}"] = R2

pd.DataFrame(r2s, index=[0]).T.plot()
print(pd.DataFrame(r2s, index=[0]).T.idxmax())
plt.show()

