import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def split_train_test_data(X, y, split_ratio):
    return train_test_split(X, y, test_size=split_ratio, random_state=0)


def scaling_continuous_feature(col_names):
    scaler = StandardScaler()
    scaler.fit(col_names)
    joblib.dump(scaler, "../models/scaler.joblib")
    s = scaler.transform(col_names)
    return(s)


def encoding_categorical_columns(col_name):
    encoder = OneHotEncoder(drop='first', sparse=False,handle_unknown='ignore')
    encoder.fit(col_name)
    joblib.dump(encoder, "../models/encoder.joblib")
    e = encoder.transform(col_name)
    return(e)


def merge_columns(s, e):
    mer_col = np.hstack((s, e))
    return(mer_col)
