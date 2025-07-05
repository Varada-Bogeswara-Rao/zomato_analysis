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
no_delivery_restaurants = data1[data1['Delivery_Rating'].isnull()]
locality_counts1 = no_delivery_restaurants['Locality'].value_counts()
locality_counts1=locality_counts1.head(10)
locality_counts1 =locality_counts1.sort_values(ascending=True, axis=0)
locality_counts1
plt.figure(figsize=(35,10))
plt.xlabel('number of restaurants without delivery ')
plt.ylabel('localities')
plt.barh(locality_counts1.index,locality_counts1.values, color='green')
plt.show()
loc_count1 = data1['Locality'].value_counts()
loc_count1 = loc_count1.sort_values(ascending=False, axis=0)
loc_count1=loc_count1.head(5)
loc_count1 = loc_count1.sort_values(ascending=True, axis=0)
loc_count1
plt.figure(figsize=(35,10))
plt.xlabel('number of restaurants ')
plt.ylabel('localities')
plt.barh(loc_count1.index,loc_count1.values, color='violet')
plt.show()
loc_count = data['Locality'].value_counts()
loc_count = loc_count.sort_values(ascending=False, axis=0)
loc_count=loc_count.head(5)
loc_count = loc_count.sort_values(ascending=True, axis=0)
loc_count
plt.figure(figsize=(35,10))
plt.xlabel('number of restaurants')
plt.ylabel('localities')
plt.barh(loc_count.index,loc_count.values, color='yellow')
plt.show()
no_delivery_restaurants = data1[data1['Delivery_Rating'].isnull()]
locality_counts1 = no_delivery_restaurants['Locality'].value_counts()
locality_counts1=locality_counts1.head(10)
locality_counts1 =locality_counts1.sort_values(ascending=True, axis=0)
locality_counts1
no_delivery_restaurants = data[data['Delivery_Rating'].isnull()]
locality_counts = no_delivery_restaurants['Locality'].value_counts()
locality_counts=locality_counts.head(10)
locality_counts =locality_counts.sort_values(ascending=True, axis=0)
locality_counts
plt.figure(figsize=(35,10))
plt.xlabel('number of restaurants without delivery ')
plt.ylabel('localities')
plt.barh(locality_counts.index,locality_counts.values, color='green')
plt.show()
plt.barh(loc_count1.index, loc_count1.values, color='green', label='totalrestaurants in Pune')
plt.barh(locality_counts1.index, locality_counts1.values, color='red',label='restaurants without delivery option in Pune')
plt.xlabel('Number of Restaurants')
plt.ylabel('Locality')
plt.title('Comparison of Restaurants Across Localities')
plt.legend()
plt.show()
plt.barh(loc_count.index, loc_count.values, color='yellow', label='totalrestaurants in Bangalore ')
plt.barh(locality_counts.index, locality_counts.values, color='red',label='restaurants without delivery option in Bangalore')
plt.xlabel('Number of Restaurants')
plt.ylabel('Locality')
plt.title('Comparison of Restaurants Across Localities')
plt.legend()
plt.show()
unique_names_data = data['Restaurant_Name'].unique()
unique_names_data1 = data1['Restaurant_Name'].unique()
num_unique_data = data['Restaurant_Name'].nunique()
num_unique_data1 = data1['Restaurant_Name'].nunique()
common_names = set(unique_names_data).intersection(unique_names_data1)
common_names_data = data[data['Restaurant_Name'].isin(common_names)]
common_names_data1 = data1[data1['Restaurant_Name'].isin(common_names)]
data['First_Word'] = data['Restaurant_Name'].str.split().str[0]
data1['First_Word'] = data1['Restaurant_Name'].str.split().str[0]
from wordcloud import WordCloud
common_names_text = ' '.join(common_names)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(common_names_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()