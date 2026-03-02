import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def check_stationarity(data):
    result = adfuller(data['cnt'])
    return result[1]  # p-value

def fit_arima_model(data, p=1, d=1, q=1):
    model = ARIMA(data['cnt'], order=(p, d, q))
    model_fit = model.fit()
    return model_fit
