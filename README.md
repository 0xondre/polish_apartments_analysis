# Analysis of apartments prices and rent in poland

This analysis consists of two main experiments:

0. **Correlation between points of interest and prices**  
   Investigating how proximity to various points of interest (e.g., schools, parks, transport) and count of points of interest in 500 m proximity affects apartment prices.
   
1. **Comparing apartment markets in other cities to Wrocław**  
   Identifying which city's apartment market is most similar to Wrocław using zero-shot learning techniques.


# Repository structure
In dataset/ we store raw datasets from kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland/data.  
In poi_charts/ we store charts of correlation between POI and price.  
In accuraccies.npy and accuracies_rent.npy we store MAE, MSE and MAPE for experiment 1.  
In apartments.ipynb we store code for experiment 0.  
In buy.csv and rent.csv we store merged raw data from dataset/.  
In regression.py and regression_rent.py we store code for experiment 1.  
In regression_buy_results.txt and regression_rent_results.txt we store interpreted data for experiment 1.  
