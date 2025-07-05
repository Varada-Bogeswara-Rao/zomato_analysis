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
if len(combined_data) == 9906:
 print("Successfully merged datasets with 9906 rows.")
else:
  print(f"Number of rows in the merged dataset: {len(combined_data)}. Pleasecheck your merge conditions.")