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
sns.pairplot(data)
sns.pairplot(data1)
corr_Pune = data1.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_Pune, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, square=True,cbar_kws={"shrink": 0.75})
plt.title('Correlation Heatmap of Pune data set')
plt.show()
corr_Beng = data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_Beng, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, square=True,cbar_kws={"shrink": 0.75})
plt.title('Correlation Heatmap of bangalore data set')
plt.show()

