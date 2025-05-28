import enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Data preparation for regression model
rent_df = pd.read_csv("rent.csv").drop(
    [
        "id",
        "condition",
        "buildingMaterial",
        "type",
        "latitude",
        "longitude",
        "ownership",
    ],
    axis=1,
)

binary_cols = [
    "hasParkingSpace",
    "hasBalcony",
    "hasElevator",
    "hasSecurity",
    "hasStorageRoom",
]

rent_df[binary_cols] = rent_df[binary_cols].fillna(0)
rent_df[binary_cols] = rent_df[binary_cols].replace({"yes": 1, "no": 0})

num_cols = ["floor", "buildYear", "floorCount"]

rent_df[num_cols] = rent_df[num_cols].fillna(rent_df[num_cols].median())


distance_cols = [
    "schoolDistance",
    "clinicDistance",
    "postOfficeDistance",
    "kindergartenDistance",
    "restaurantDistance",
    "collegeDistance",
    "pharmacyDistance",
]

rent_df[distance_cols] = rent_df[distance_cols].fillna(rent_df[distance_cols].max())

lower = rent_df["price"].quantile(0.01)
upper = rent_df["price"].quantile(0.99)
rent_df = rent_df[(rent_df["price"] >= lower) & (rent_df["price"] <= upper)]


# regression model
cities = rent_df["city"].unique()
models = ("RandomForest", "Linear", "KNeighbours")
rkf = RepeatedKFold(n_splits=2, n_repeats=5)
x_test = rent_df[rent_df["city"] == "wroclaw"].drop(["price", "city"], axis=1).to_numpy()
y_test = rent_df[rent_df["city"] == "wroclaw"]["price"].to_numpy()
accuracies = np.zeros((len(cities), 10, 3, 3))

for i, city in enumerate(cities):
    x = rent_df[rent_df["city"] == city].drop(["price", "city"], axis=1).to_numpy()
    y = rent_df[rent_df["city"] == city]["price"].to_numpy()
    for j, (train_index, test_index) in enumerate(rkf.split(x,y)):
        maes=[]
        mses=[]
        mapes=[]
        for k, model in enumerate(
        (
            RandomForestRegressor(n_estimators=10, max_depth=5, random_state=54),
            LinearRegression(),
            KNeighborsRegressor(n_neighbors=5),
        )):
            model.fit(x[train_index], y[train_index])
            y_pred = model.predict(x_test)
            accuracies[i, j, k, 0] = mean_absolute_error(y_test,y_pred)
            accuracies[i, j, k, 1] = mean_squared_error(y_test,y_pred)
            accuracies[i, j, k, 2] = mean_absolute_percentage_error(y_test,y_pred)

np.save('accuracies_rent.npy', accuracies)

city_model_avg = accuracies.mean(axis=1)

for i, city in enumerate(cities):
    print(f"Metrics for {city}:")
    for j, model in enumerate(models):
        mae = city_model_avg[i, j, 0]
        mse = city_model_avg[i, j, 1]
        mape = city_model_avg[i, j, 2]
        print(f"  {model}: MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape*100:.2f}%")
