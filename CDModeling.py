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
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import category_encoders as ce
data = pd.read_csv("D:\Bangalore_Restaurants.csv")
data1 = pd.read_csv("D:\Pune Restaurants.csv")
common_columns = [
'Restaurant_Name', 'Category', 'Locality', 'Pricing_for_2',
'Dining_Rating', 'Dining_Review_Count', 'Delivery_Rating','Delivery_Rating_Count']
combined_data = pd.merge(data, data1, on=common_columns, how='outer')
combined_data[['Category', 'Locality']] = combined_data[['Category','Locality']].astype('category')
encoder = ce.TargetEncoder(cols=['Category', 'Locality'])
combined_data_encoded = encoder.fit_transform(combined_data,combined_data['Pricing_for_2'])
print(combined_data_encoded.head())
x = combined_data_encoded.iloc[:,[1,3,4,5,6,7]]
y = combined_data_encoded['Pricing_for_2']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1,random_state=353)
x_train.head()
y_train.head()
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)
r2_score(y_test,y_pred_xgb)
feature_importances = xgb_model.feature_importances_
print("Feature Importances:", feature_importances)
cv_scores = cross_val_score(xgb_model, x_train, y_train, cv=5, scoring='r2')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb})
print(comparison_df.head(5))
plt.scatter(y_test, y_pred_xgb)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()
