import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = 'poi_charts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

buy_df = pd.read_csv('buy.csv')
rent_df = pd.read_csv('rent.csv')

poi_features = ['poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance', 'pharmacyDistance']


def plot_price_correlations(df, title, filename):
    correlations = []
    for feature in poi_features:
        corr = df[feature].corr(df['price'])
        correlations.append(corr)

    corr_df = pd.DataFrame({
        'feature': poi_features,
        'correlation': correlations
    })

    plt.figure(figsize=(10, 6))
    bars = plt.barh(corr_df['feature'], corr_df['correlation'], color=[
        'red' if x < 0 else 'green' for x in corr_df['correlation']
    ])

    for bar in bars:
        width = bar.get_width()
        position = bar.get_y() + bar.get_height() / 2
        plt.text(width, position, f'{width:.2f}', va='center')

    # limit from -1 to 1
    plt.xlim(-1, 1)
    plt.xlabel('correlation with price')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

if __name__ == "__main__":
    # overall correlation for buy and rent
    plot_price_correlations(buy_df, 'overall - sale', f'{output_dir}/buy_overall.png')
    plot_price_correlations(rent_df, 'overall - rent', f'{output_dir}/rent_overall.png')

    for city in buy_df['city'].unique():
        city_df = buy_df[buy_df['city'] == city]
        plot_price_correlations(city_df, f'buy - {city}', f'{output_dir}/buy_{city}.png')

    for city in rent_df['city'].unique():
        city_df = rent_df[rent_df['city'] == city]
        plot_price_correlations(city_df, f'rent - {city}', f'{output_dir}/rent_{city}.png')
