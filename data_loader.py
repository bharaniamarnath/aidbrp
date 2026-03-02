import pandas as pd

def load_data():
    data_day = pd.read_csv('datasets/day.csv')
    data_hour = pd.read_csv('datasets/hour.csv')
    return data_day, data_hour