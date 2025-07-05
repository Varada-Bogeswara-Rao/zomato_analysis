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
corr = combined_data.corr()
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, square=True,cbar_kws={"shrink": 0.75})
plt.title('Correlation Heatmap of combined_data')
plt.show()