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
sns.displot(data1['Pricing_for_2'])
sns.displot(data1['Delivery_Rating'])
sns.displot(data1['Dining_Rating'])
sns.displot(data1['Delivery_Rating'])
sns.displot(data['Pricing_for_2'])
sns.displot(data['Delivery_Rating'])
sns.displot(data['Dining_Rating'])
sns.displot(data['Delivery_Rating'])
plt.show()
plot = sns.displot(data['Pricing_for_2'])
plot.savefig('pricing_distribution.png')
