# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# configurations
import warnings
warnings.filterwarnings('ignore')

# load dataset
data_day = pd.read_csv('datasets/day.csv')
data_hour = pd.read_csv('datasets/hour.csv')

# display first few rows
print(data_day.head())
print(data_hour.head())
# # data analysis

# import necessary libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

data = data_hour.copy()

# set index
data['dteday'] = pd.to_datetime(data['dteday'])
data.set_index('dteday', inplace=True)

# check the First Few Rows of the Dataset
print(data.head())

# exploratory data Analysis
# overview of data
print(data.info())
print(data.describe())

# correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# plot rentals by time
plt.figure(figsize=(15, 6))
plt.plot(data['cnt'], color='blue')
plt.title('Total Bike Rentals Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.grid()
plt.show()

# seasonal and hourly trends
data['hour'] = data.index.hour
hourly_data = data.groupby('hr').sum()['cnt']
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_data.index, y=hourly_data.values)
plt.title('Total Bike Rentals by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rentals')
plt.grid()
plt.show()

# check stationarity
result = adfuller(data['cnt'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# check p value > 0.05 then series is non-stationary
# proceed with differencing
if result[1] > 0.05:
    data['cnt_diff'] = data['cnt'].diff().dropna()
    plt.figure(figsize=(15, 6))
    plt.plot(data['cnt_diff'], color='orange')
    plt.title('Differenced Bike Rentals')
    plt.xlabel('Date')
    plt.ylabel('Differenced Rentals')
    plt.grid()
    plt.show()

# ACF and PACF plots
plt.figure(figsize=(15, 6))
plot_acf(data['cnt'].dropna(), lags=30)
plt.title('ACF of Rentals')
plt.show()

plt.figure(figsize=(15, 6))
plot_pacf(data['cnt'].dropna(), lags=30)
plt.title('PACF of Rentals')
plt.show()

# define p, d, and q parameters based on ACF and PACF plots
p = 1
d = 1
q = 1

# fit ARIMA Model
model = ARIMA(data['cnt'], order=(p, d, q))
model_fit = model.fit()

# model summary
print(model_fit.summary())

# forecasting
forecast = model_fit.forecast(steps=30)  # Forecast for the next 30 days
plt.figure(figsize=(15, 6))
plt.plot(data['cnt'], label='Historical Rentals', color='blue')
plt.plot(forecast.index, forecast.values, label='Forecasted Rentals', color='red')
plt.title('Bike Rentals Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.legend()
plt.grid()
plt.show()

# evaluate model
# split data into train and test set 
# last 30 days as test set
train, test = data['cnt'][:-30], data['cnt'][-30:]

# fit ARIMA on train Set
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# predictions
predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(15, 6))
plt.plot(train.index, train, label='Training data', color='blue')
plt.plot(test.index, test, label='Test data', color='orange')
plt.plot(test.index, predictions, label='Predictions', color='green')
plt.title('Train/Test Split and Predictions')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.legend()
plt.grid()
plt.show()

# daily bike rental count based on environmental and seasonal settings

# import necessary Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# load dataset
data = data_hour.copy()

# format date index
data['dteday'] = pd.to_datetime(data['dteday'])

# feature engineering
# extract day, month, year from date
data['day'] = data['dteday'].dt.day
data['month'] = data['dteday'].dt.month
data['year'] = data['dteday'].dt.year
data['weekday'] = data['dteday'].dt.weekday

# drop unnecessary columns
data.drop(columns=['instant', 'dteday', 'casual', 'registered'], inplace=True)

# display updated dataframe
print(data.head())

# prepare features and target
X = data.drop('cnt', axis=1)  # Features
y = data['cnt']  # Target variable

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# prediction on test set
predictions = rf_model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# plot actual vs predicted values
plt.figure(figsize=(15, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Rentals')
plt.xlabel('Actual Rentals')
plt.ylabel('Predicted Rentals')
plt.grid()
plt.show()

# feature importance
feature_importances = rf_model.feature_importances_
features = data.drop('cnt', axis=1).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# plot Feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances for Bike Rentals Prediction')
plt.show()



