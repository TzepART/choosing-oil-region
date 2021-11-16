from sklearn.metrics import mean_squared_error
from src.data_work import *
from src.model_features import *
from src.busines_features import *
from src.metric_features import *

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

data_0 = pd.read_csv('datasets/geo_data_0.csv')
data_1 = pd.read_csv('datasets/geo_data_1.csv')
data_2 = pd.read_csv('datasets/geo_data_2.csv')

data_0 = data_0.drop_duplicates(subset=['id'])
data_1 = data_1.drop_duplicates(subset=['id'])
data_2 = data_2.drop_duplicates(subset=['id'])

# result_data = union_several_dataframes([data_0, data_1, data_2])
data_0 = data_0.drop(columns=['id'])
data_1 = data_1.drop(columns=['id'])
data_2 = data_2.drop(columns=['id'])


def step_3(researched_data: DataFrame):
    # Bite data into training and validation sets in a 75:25 ratio
    data_train_valid = separate_on_train_and_valid_data(researched_data, target_field='product', valid_size_value=0.25)
    # Train the model and make predictions on the validation set.
    model = build_linear_regression_model(
        data_train_valid.features_train,
        data_train_valid.target_train,
        count_jobs=-1
    )
    # Save the predictions and correct answers on the validation set
    predictions = model.predict(data_train_valid.features_valid)
    real_target_values = data_train_valid.target_valid
    # Display the average stock of the predicted raw materials and the RMSE model
    series_predictions = pd.Series(predictions)
    mean = series_predictions.mean()
    mse = mean_squared_error(real_target_values, predictions)
    rmse = mse ** 0.5
    print('Средний запас предсказанного сырья:', mean)
    print('RMSE (Среднеквадратическая ошибка модели):', rmse)
    print('Доверительный интервал прогноза: ({},{})'.format(mean - 2 * rmse, mean + 2 * rmse))
    # print('Cross validation score:', cross_val_score_by_model(model, data_train_valid.features_valid, data_train_valid.target_valid, 5))

    return model, series_predictions, real_target_values


model_0, predictions_0, target_valid_0 = step_3(data_0)
model_1, predictions_1, target_valid_1 = step_3(data_1)
model_2, predictions_2, target_valid_2 = step_3(data_2)

# show_features_graphs(data_0)
# show_features_graphs(data_1)
# show_features_graphs(data_2)

region_budget = 10000000000
count_researching_points = 500
count_best_points = 200
product_profit = 450000
need_quantile = .025
cost_price_one_well = region_budget / count_best_points  # 50 млн. руб. - стоимость 1-ой скважины
break_even_product_quantity = np.ceil(cost_price_one_well / product_profit)  # безубыточное количество продукта - 112 шт


# print(data_0.sort_values(by='product', ascending=False).iloc[0:count_best_points]['product'].sum()*450000)
# print(data_1.sort_values(by='product', ascending=False).iloc[0:count_best_points]['product'].sum()*450000)
# print(data_2.sort_values(by='product', ascending=False).iloc[0:count_best_points]['product'].sum()*450000)


def step_5(data, model, region_number):
    predictions_500 = get_predictions_500(data, model)
    predictions_500 = predictions_500.sort_values(ascending=False)
    print('Прибыль для полученного объёма сырья для {}-го региона:'.format(region_number),
          transform_to_million_rub(profit_of_wells_by_predictions(
              predictions_500.iloc[0:count_best_points],
              product_profit,
              break_even_product_quantity
          )))


step_5(data_0, model_0, region_number=1)
step_5(data_1, model_1, region_number=2)
step_5(data_2, model_2, region_number=3)


def step_5_using_bootstrap(target_valid, predictions):
    lower, upper, mean, values = get_mean_and_quantile_by_bootstrap(
        target_valid,
        predictions,
        need_quantile,
        (1 - need_quantile),
        additional_multiplier=product_profit
    )
    print('Средняя выручка:', transform_to_million_rub(mean))
    print('Прибыль опираясь на среднюю выручку:', transform_to_million_rub(mean - region_budget))
    print('ДИ истинного среденего: ({}, {})'.format(transform_to_million_rub(lower), transform_to_million_rub(upper)))

    up_profit = upper - region_budget
    down_profit = lower - region_budget
    print('Интервал прибыли опираясь на ДИ истинного среденего: ({}, {})'.format(
        transform_to_million_rub(down_profit),
        transform_to_million_rub(up_profit))
    )

    if up_profit < 0:
        probability_of_losses_by_cond_interval = 100.0
    elif up_profit > 0 and down_profit < 0:
        probability_of_losses_by_cond_interval = (abs(down_profit) / (abs(down_profit) + up_profit)) * 100
    else:
        probability_of_losses_by_cond_interval = 0.0

    print('Доля отрицательной части ДИ истинного среденего: {:.2f}%'.format(probability_of_losses_by_cond_interval))

    profit_values = values.apply(lambda value: value - region_budget)
    probability_of_losses = len(profit_values[profit_values < 0]) / len(profit_values)
    print('Риск убытка: {:.2f}%'.format(probability_of_losses * 100))


step_5_using_bootstrap(target_valid_0, predictions_0)
step_5_using_bootstrap(target_valid_1, predictions_1)
step_5_using_bootstrap(target_valid_2, predictions_2)
