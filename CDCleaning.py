import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
data = pd.read_csv("D:\Bangalore_Restaurants.csv")
data1 = pd.read_csv("D:\Pune Restaurants.csv")
common_columns = [
'Restaurant_Name', 'Category', 'Locality', 'Pricing_for_2',
'Dining_Rating', 'Dining_Review_Count', 'Delivery_Rating','Delivery_Rating_Count']
combined_data = pd.merge(data, data1, on=common_columns, how='outer')
print(combined_data.shape)
print(combined_data.columns)
print(combined_data.isnull().sum())
combined_data['Delivery_Rating'].fillna(combined_data['Delivery_Rating'].mean(),inplace =True)
combined_data['Dining_Rating'].fillna(combined_data['Dining_Rating'].mean(),inplace =True)
combined_data['Dining_Review_Count'].fillna(combined_data['Dining_Review_Count'].mean(),inplace= True)
combined_data['Delivery_Rating_Count'].fillna(combined_data['Delivery_Rating_Count'].mean(),inplace= True)
print(combined_data.isnull().sum())
mode_fill_cols = [
    'Website_x', 'Website_y', 'Address_x', 'Address_y', 
    'Phone_No_x', 'Phone_No_y', 'Latitude_x', 'Latitude_y', 
    'Longitude_x', 'Longitude_y', 'Known_for1', 'Known_for2'
]
mean_fill_cols = ['Delivery_Rating_Count']
for col in mode_fill_cols:
    if combined_data[col].isnull().any():
        combined_data[col].fillna(combined_data[col].mode()[0], inplace=True)
for col in mean_fill_cols:
    if combined_data[col].isnull().any():
        combined_data[col].fillna(combined_data[col].mean(), inplace=True)
print("Missing values after cleaning:")
print(combined_data.isnull().sum())
