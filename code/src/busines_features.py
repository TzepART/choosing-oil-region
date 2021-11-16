import pandas as pd
import numpy as np
from pandas import Series


def revenue(target, predictions, count, additional_multiplier=1):
    predict_sorted = predictions.sort_values(ascending=False)
    selected = target[predict_sorted.index][:count]

    return additional_multiplier * selected.sum()


def get_mean_and_quantile_by_bootstrap(
        target: Series, predictions: Series,
        lower_quantile: float,
        upper_quantile: float,
        count_samples=200,
        additional_multiplier=1
):
    state = np.random.RandomState(12345)
    values = []
    for i in range(1000):
        target_subsample = target.sample(n=500, replace=True, random_state=state)
        predict_subsample = predictions[target_subsample.index]
        values.append(revenue(target_subsample, predict_subsample, count_samples, additional_multiplier))

    values = pd.Series(values)
    lower = values.quantile(lower_quantile)
    upper = values.quantile(upper_quantile)
    mean = values.mean()

    return lower, upper, mean, values


def profit_of_wells_by_predictions(predictions: Series, product_profit, break_even_product_quantity):
    return (predictions.sum() - len(predictions) * break_even_product_quantity) * product_profit


def transform_to_million_rub(value: float) -> str:
    return '{:.3f} млн. руб.'.format(value / 10 ** 6)
