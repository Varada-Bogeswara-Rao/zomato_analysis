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
print(data.head(5))
print(data1.head(5))
print(data.shape)
print(data.columns)
print(data.isnull().sum())
print(data1.shape)
print(data1.columns)
print(data1.isnull().sum())
data1['Delivery_Rating'].fillna(data1['Delivery_Rating'].mean(),inplace =True)
data1['Known_for1'].fillna(data1['Known_for1'].mode()[0],inplace=True)
data1['Known_for2'].fillna(data1['Known_for2'].mode()[0],inplace=True)
data1['Delivery_Rating_Count'].fillna(data1['Delivery_Rating_Count'].mean(),inplace =True)
print(data1.isnull().sum())
data['Delivery_Rating'].fillna(data['Delivery_Rating'].mean(),inplace= True)
data['Dining_Rating'].fillna(data['Dining_Rating'].mean(),inplace=True)
data['Dining_Review_Count'].fillna(data['Dining_Review_Count'].mean(),inplace =True)
data['Delivery_Rating_Count'].fillna(data['Delivery_Rating_Count'].mean(),inplace = True)
print(data.isnull().sum())
 