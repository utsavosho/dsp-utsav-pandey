import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from .preprocess import split_train_test_data, scaling_continuous_feature, encoding_categorical_columns, merge_columns


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    X = data[['GrLivArea', 'PoolArea', 'OverallQual', 'PavedDrive']]
    y = data[['SalePrice']]
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, split_ratio=0.25)
    scaled_columns_train = scaling_continuous_feature(X_train[['GrLivArea', 'PoolArea']])
    encoded_columns_train = encoding_categorical_columns(X_train[['OverallQual', 'PavedDrive']])
    X_train_new = merge_columns(scaled_columns_train, encoded_columns_train)
    reg_multiple = LinearRegression()
    reg_multiple_new = reg_multiple.fit(X_train_new, y_train)
    joblib.dump(reg_multiple_new, "../models/model.joblib")
    encoder = joblib.load("../models/encoder.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    scaled_columns_test = scaler.transform(X_test[['GrLivArea', 'PoolArea']])
    encoded_columns_test = encoder.transform(X_test[['OverallQual', 'PavedDrive']])
    X_test_new = merge_columns(scaled_columns_test, encoded_columns_test)
    y_pred = reg_multiple_new.predict(X_test_new)
    rmsle_score = compute_rmsle(y_test, y_pred)
    return{"rmsle": rmsle_score}
