from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def get_accuracy_score_by_model(model, features, target):
    train_predictions = model.predict(features)

    return accuracy_score(target, train_predictions)


def show_features_graphs(data: DataFrame):
    columns = data.columns
    for column in columns:
        data[[column]].plot.hist(bins=100, alpha=0.5, figsize=(10, 6), grid=True)
        plt.title('Гистограмма распределения: {}'.format(str(column)))
        plt.show()


def cross_val_score_by_model(model, features: DataFrame, target: Series, cv_value=3):
    scores = cross_val_score(model, features, target, cv=cv_value)
    final_score = sum(scores) / len(scores)

    return final_score
