"""
A collection of Python functions and classes
"""


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, random_split


__author__ = "Erik Matovic"
__version__ = "1.0"
__email__ = "xmatovice@stuba.sk"
__status__ = "Development"


def split_train_val(df: pd.DataFrame, split_scalar: float=0.1) -> Tuple[
    torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:
    """
    :param df: pandas dataframe
    :param split_scalar: % split into train-validation
    """
    return random_split(
        df, [len(df) - int(split_scalar * len(df)), int(split_scalar * len(df))])


def plot_accuracy(train_accuracy, validation_accuracy) -> None:
    """
    TODO: refactor
    """
    f = plt.figure(figsize=(10,6)) #plotting
    f.set_size_inches(18.5, 10.5)
    f.set_dpi(100)

    accuracy_csv = pd.DataFrame({
        "epochs": range(len(train_accuracy)), 
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy
    })
    # loss_csv.to_csv("../outputs/loss.csv")
    # gca stands for 'get current axis'
    ax = plt.gca()
    accuracy_csv.plot(kind='line',x='epochs',y='train_accuracy', ax=ax)
    accuracy_csv.plot(kind='line',x='epochs',y='validation_accuracy', color='red', ax=ax)
    plt.title(f'Train loss vs Validation accuracy on {len(train_accuracy)} epochs')
    plt.show()


def plot_loss(train_loss, val_loss) -> None:
    '''
    Visualize training loss vs. validation loss.
    Parameters
    ----------
    train_loss: training loss
    val_loss: validation loss
    Returns: None
    -------
    '''
    f = plt.figure(figsize=(10,6)) #plotting
    f.set_size_inches(18.5, 10.5)
    f.set_dpi(100)

    loss_csv = pd.DataFrame({"epochs": range(len(train_loss)), "train_loss": train_loss,
                             "val_loss": val_loss})
    # loss_csv.to_csv("../outputs/loss.csv")
    # gca stands for 'get current axis'
    ax = plt.gca()
    loss_csv.plot(kind='line',x='epochs',y='train_loss',ax=ax )
    loss_csv.plot(kind='line',x='epochs',y='val_loss', color='red', ax=ax)
    plt.title(f'Train loss vs Validation loss on {len(train_loss)} epochs')
    plt.show()
    # plt.savefig("../outputs/train_vs_val_loss.png")


def check_null_values(df: pd.DataFrame) -> None:
    """
    Print NULL values.
    :param: df - pandas dataframe
    :returns: nothing
    """
    for col in df:
        print(col, df[col].isnull().values.any())


def print_sum_null(df: pd.DataFrame) -> None:
    """
    Print sum of NULL values.
    :param: df - pandas dataframe
    :returns: nothing
    """
    print(df.isnull().sum())


def rescale(df: pd.DataFrame, col: str) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Rescale values using scikit-learn's MinMaxScaler.
    :param: df - dataframe
    :returns: rescaled values
    """
    scaler = MinMaxScaler()

    # The scaler expects the data to be shaped as (x, y), 
    # so we add a dimension using reshape.
    values = df[col].values.reshape(-1, 1)
    values_scaled = scaler.fit_transform(values)

    return scaler, values_scaled


def to_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    """
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def split_data(data_raw: np.ndarray, seq_len: int, train_split: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


def cohens_d_calculate(dataset1: pd.Series, dataset2: pd.Series) -> np.float64:
    """ 
    Calculate Cohens d to measure the strength of 
    the relationship between two variables in a dataset
    :param: dataset1 - pandas series
    :param: dataset2 - pandas series
    :returns: 
    """
    dataset_len = len(dataset1), len(dataset2) 
    var = np.var(dataset1, ddof=1), np.var(dataset2, ddof=1)
    pooled_std_dev = math.sqrt(
        ((dataset_len[0] - 1) * var[0] + (dataset_len[1] - 1) * var[1]) / 
        (dataset_len[0] + dataset_len[1] - 2)
    ) 
    dataset_mean = np.mean(dataset1), np.mean(dataset2)
    return abs(((dataset_mean[0] - dataset_mean[1]) / pooled_std_dev))
