import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from data_analysis import plot_correlation_matrix, plot_rentals_over_time

# Load dataset
data_hour = pd.read_csv('datasets/hour.csv')

# Streamlit title
st.title('Bike Rental Data Analysis')

# Create tabs
tabs = st.tabs([
    "Correlation Matrix", 
    "Total Rentals Over Time", 
    "Check Stationarity",
    "Fit ARIMA Model"
])

# Tab for Correlation Matrix
with tabs[0]:
    st.header("Correlation Matrix")
    plot_correlation_matrix(data_hour)
    st.pyplot(plt)  # Use Streamlit to display the plot
    plt.clf()  # Clear the figure after using st.pyplot

# Tab for Total Rentals Over Time
with tabs[1]:
    st.header("Total Rentals Over Time")
    plot_rentals_over_time(data_hour)
    st.pyplot(plt)  # Use Streamlit to display the plot
    plt.clf()  # Clear the figure after using st.pyplot

# Tab for Check Stationarity
with tabs[2]:
    st.header("Check Stationarity")
    
    def check_stationarity(data):
        result = adfuller(data['cnt'])
        st.write(f'ADF Statistic: {result[0]}')
        st.write(f'p-value: {result[1]}')
        if result[1] > 0.05:
            st.write("The time series is non-stationary.")
        else:
            st.write("The time series is stationary.")

    check_stationarity(data_hour)

# Tab for Fit ARIMA Model
with tabs[3]:
    st.header("Fit ARIMA Model")
    
    p = st.number_input("Select p value:", min_value=0, max_value=5, value=1)
    d = st.number_input("Select d value:", min_value=0, max_value=5, value=1)
    q = st.number_input("Select q value:", min_value=0, max_value=5, value=1)

    if st.button("Fit ARIMA Model"):
        model = ARIMA(data_hour['cnt'], order=(p, d, q))
        model_fit = model.fit()
        st.write(model_fit.summary())
        
        # Plot the forecast using the new method
        from data_analysis import plot_forecast
        plot_forecast(model_fit, data_hour)
        st.pyplot(plt)  # Use Streamlit to display the plot
        plt.clf()  # Clear the figure after using st.pyplot
