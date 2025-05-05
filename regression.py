import pandas as pd
from sklearn.linear_model import LinearRegression

# Data preparation for regression model
buy_df = pd.read_csv("buy.csv").drop(["id"], axis=1)

binary_cols = [
    "hasParkingSpace",
    "hasBalcony",
    "hasElevator",
    "hasSecurity",
    "hasStorageRoom",
]
buy_df[binary_cols] = buy_df[binary_cols].fillna(0)
buy_df[binary_cols] = buy_df[binary_cols].replace({"yes": 1, "no": 0})

print(buy_df.head(10))

# test_x = buy_df[buy_df['city']=='wroclaw'].loc[:, ~buy_df.columns.isin(['price','city'])]
# test_y = buy_df[buy_df['city']=='wroclaw']['price']

# for city in buy_df['city'].unique():
#     if city == 'wroclaw':
#         continue
#     X = buy_df[buy_df['city']==city].loc[:, ~buy_df.columns.isin(['price','city'])]
#     y = buy_df[buy_df['city']==city]['price']
#     model = LinearRegression()
#     model.fit(X,y)
#     y_pred = model.predict(test_x)
