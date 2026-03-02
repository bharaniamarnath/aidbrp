import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def format_date(data):
    # Assuming `data` is your dataframe
    data['dteday'] = pd.to_datetime(data['dteday'])  # Convert to datetime format
    # Extract relevant features
    data['day'] = data['dteday'].dt.day
    data['month'] = data['dteday'].dt.month
    data['year'] = data['dteday'].dt.year
    data['weekday'] = data['dteday'].dt.weekday
    return data

def plot_correlation_matrix(data):
    data = format_date(data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def plot_rentals_over_time(data):
    plt.figure(figsize=(15, 6))
    plt.plot(data['cnt'], color='blue')
    plt.title('Total Bike Rentals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.grid()
    plt.show()

def plot_forecast(model_fit, data):
    forecast = model_fit.forecast(steps=30)
    plt.figure(figsize=(15, 6))
    plt.plot(data['cnt'], label='Historical Rentals', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecasted Rentals', color='red')
    plt.title('Bike Rentals Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.legend()
    plt.show()
