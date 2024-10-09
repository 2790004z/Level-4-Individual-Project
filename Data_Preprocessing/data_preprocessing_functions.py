import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.signal import savgol_filter

def impute_missing_values(data):
    # Impute missing values using IterativeImputer
    imputer = IterativeImputer(max_iter=100, random_state=42,sample_posterior=False,skip_complete=True)

    cols = data.columns
    index = data.index

    data = imputer.fit_transform(data)
    
    imputed_data = pd.DataFrame(data, columns=cols, index=index)
    return imputed_data

def min_max_normalization(data):
    return (data - data.min()) / (data.max() - data.min())

def savgol_smoothing(data, window_length=10, polyorder=2):
    for col in data.columns:
        data[col] = savgol_filter(data[col], window_length, polyorder)
    return data