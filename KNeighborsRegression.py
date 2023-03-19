# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt")

# Get some insights into the data
print(df.head())  # print the first 5 rows
print(df.info())  # print the data types of columns
print(df.describe())  # print the statistical summary of the data

# Show some interesting visualization of the data
plt.figure(figsize=(12, 8))
plt.scatter(df['Sqft'], df['Price'])
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(df['Bedroom'], df['Price'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price vs Number of Bedrooms')
plt.show()

# Manage data for training & testing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Finding a better value of k
rmse_values = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse_values.append(np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(12, 8))
plt.plot(range(1, 21), rmse_values, marker='o')
plt.xlabel('Number of Neighbors')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Neighbors')
plt.show()