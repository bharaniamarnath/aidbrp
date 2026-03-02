import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def prepare_data(data):
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['day'] = data['dteday'].dt.day
    data['month'] = data['dteday'].dt.month
    data['year'] = data['dteday'].dt.year
    data['weekday'] = data['dteday'].dt.weekday
    data.drop(columns=['instant', 'dteday', 'casual', 'registered'], inplace=True)

    X = data.drop('cnt', axis=1)
    y = data['cnt']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def fit_random_forest_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model
