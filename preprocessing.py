import utils 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


'''
This module is to preprocess data for all formats. 
This includes scaling, removing outliers, feature selection, etc.
'''

def MinMaxScaleDataframe(df):
    x = df.values   # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)


def preprocess_data(data, dtype):
    if dtype == "face":
        data = MinMaxScaleDataframe(data)
        return data
    elif dtype == "text":
        data = MinMaxScaleDataframe(data)
        return data
    elif dtype == "relation":
        data = MinMaxScaleDataframe(data)
        return data 

