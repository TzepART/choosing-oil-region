import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression


def build_linear_regression_model(features_train, target_train: Series, count_jobs=1) -> LinearRegression:
    linear_regression_model = LinearRegression(n_jobs=count_jobs)
    linear_regression_model.fit(features_train, target_train)

    return linear_regression_model


def get_predictions_500(data: DataFrame, model) -> Series:
    data_500 = data.sample(500, replace=True, random_state=12345).reset_index(drop=True)
    features_train_500 = data_500.drop(['product'], axis=1).reset_index(drop=True)
    predictions = model.predict(features_train_500)

    return pd.Series(predictions)
