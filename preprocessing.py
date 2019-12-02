import utils 

'''
This module is to preprocess data for all formats. 
This includes scaling, removing outliers, feature selection, etc.
'''

def preprocess_data(data, dtype):
    if dtype == "face":
        data, _ = utils.load_data_from_csv(dtype)
        #data = preprocess
        return data
    elif dtype == "text":
        data, _ = utils.load_data_from_csv(dtype)
        #data = preprocess
        return data
    elif dtype == "relation":
        data, _ = utils.load_data_from_csv(dtype)
        #data = preprocess
        return data 

