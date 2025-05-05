import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data preparation for regression model
buy_df = pd.read_csv("buy.csv").drop(
    ["id", "condition", "buildingMaterial", "type","latitude","longitude","ownership"], axis=1
)

binary_cols = [
    "hasParkingSpace",
    "hasBalcony",
    "hasElevator",
    "hasSecurity",
    "hasStorageRoom",
]
buy_df[binary_cols] = buy_df[binary_cols].fillna(0)
buy_df[binary_cols] = buy_df[binary_cols].replace({"yes": 1, "no": 0})

num_cols = ["floor", "buildYear", "floorCount"]
buy_df[num_cols] = buy_df[num_cols].fillna(buy_df[num_cols].median())

distance_cols = [
    "schoolDistance",
    "clinicDistance",
    "postOfficeDistance",
    "kindergartenDistance",
    "restaurantDistance",
    "collegeDistance",
    "pharmacyDistance",
]

buy_df[distance_cols] = buy_df[distance_cols].fillna(buy_df[distance_cols].max())

lower = buy_df["price"].quantile(0.01)
upper = buy_df["price"].quantile(0.99)
buy_df = buy_df[(buy_df["price"] >= lower) & (buy_df["price"] <= upper)]

print(buy_df)
print(buy_df.isnull().sum())


# regression model
x_test = buy_df[buy_df["city"] == "wroclaw"].drop(["price", "city"], axis=1)
y_test = buy_df[buy_df["city"] == "wroclaw"]["price"]

for city in buy_df['city'].unique():
    if city == 'wroclaw':
        continue
    x = buy_df[buy_df['city']==city].drop(['price','city'], axis=1)
    y = buy_df[buy_df['city']==city]['price']
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=54)
    model.fit(x, y)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(city)
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R^2: {r2:.4f}")

