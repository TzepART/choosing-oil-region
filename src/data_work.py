import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def union_several_dataframes(data_frames: list) -> DataFrame:
    union_dataframe = pd.concat(data_frames, ignore_index=True, sort=False).reset_index(drop=True)

    return union_dataframe


@dataclass
class TrainValidAndTestData:
    features_train: pd.DataFrame
    features_valid: pd.DataFrame
    features_test: pd.DataFrame
    target_train: pd.Series
    target_valid: pd.Series
    target_test: pd.Series


def separate_on_train_valid_and_test_data(
        separating_data: pd.DataFrame,
        target_field: str,
        test_size_value=0.2,
        valid_size_value=0.25
) -> TrainValidAndTestData:
    data_teaching, data_test = train_test_split(separating_data, test_size=test_size_value, random_state=12345)
    data_train, data_valid = train_test_split(data_teaching, test_size=valid_size_value, random_state=12345,
                                              stratify=data_teaching[target_field])

    features_train = data_train.drop([target_field], axis=1).reset_index(drop=True)
    features_valid = data_valid.drop([target_field], axis=1).reset_index(drop=True)
    features_test = data_test.drop([target_field], axis=1).reset_index(drop=True)

    target_train = data_train[target_field].reset_index(drop=True)
    target_valid = data_valid[target_field].reset_index(drop=True)
    target_test = data_test[target_field].reset_index(drop=True)

    return TrainValidAndTestData(features_train, features_valid, features_test, target_train, target_valid, target_test)


@dataclass
class TrainAndValidData:
    features_train: pd.DataFrame
    features_valid: pd.DataFrame
    target_train: pd.Series
    target_valid: pd.Series


def separate_on_train_and_valid_data(
        separating_data: pd.DataFrame,
        target_field: str,
        valid_size_value=0.25
) -> TrainAndValidData:
    data_train, data_valid = train_test_split(separating_data, test_size=valid_size_value, random_state=12345)

    features_train = data_train.drop([target_field], axis=1).reset_index(drop=True)
    features_valid = data_valid.drop([target_field], axis=1).reset_index(drop=True)

    target_train = data_train[target_field].reset_index(drop=True)
    target_valid = data_valid[target_field].reset_index(drop=True)

    return TrainAndValidData(features_train, features_valid, target_train, target_valid)


def separate_on_train_and_valid_data_with_scale(
        separating_data: pd.DataFrame,
        target_field: str,
        valid_size_value=0.25
) -> TrainAndValidData:
    separating_data = scale_dataframe_data(separating_data)
    data_train, data_valid = train_test_split(separating_data, test_size=valid_size_value, random_state=12345)

    features_train = data_train.drop([target_field], axis=1).reset_index(drop=True)
    features_valid = data_valid.drop([target_field], axis=1).reset_index(drop=True)

    target_train = data_train[target_field].reset_index(drop=True)
    target_valid = data_valid[target_field].reset_index(drop=True)

    return TrainAndValidData(features_train, features_valid, target_train, target_valid)


def scale_dataframe_data(separated_data: DataFrame) -> DataFrame:
    scaler = StandardScaler()
    separated_scaler_data = pd.DataFrame(scaler.fit_transform(separated_data), columns=separated_data.columns)

    return separated_scaler_data


def scale_series_data(separated_data: Series) -> Series:
    scaler = StandardScaler()
    separated_scaler_data = pd.Series(scaler.fit_transform(separated_data))

    return separated_scaler_data


def normalize_data(data: DataFrame) -> DataFrame:
    return preprocessing.normalize(data)