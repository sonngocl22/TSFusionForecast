import json
import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def load_config(config_name):

    config_dir = os.path.join(os.path.dirname(__file__),'..','configs')
    file_path = os.path.join(config_dir, config_name)

    with open(file_path, 'r') as file:
        config = json.load(file)

    return config

def smape_loss(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
    y_true (array): True values.
    y_pred (array): Predicted values.

    Returns:
    float: SMAPE score.
    """
    # Avoid division by zero by adding a small epsilon
    epsilon = np.finfo(np.float64).eps
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred) + epsilon, 0.5 + epsilon)

    # Calculate SMAPE
    smape_value = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denominator)
    return smape_value

def get_indices_entire_sequence(data: pd.DataFrame, hyperparameters: dict) -> list:
    """
    Produce all the start and end index positions that are needed to produce
    the sub-sequences for the dataset.

    Args:
        data (pd.DataFrame): Partitioned data set, e.g., training data
        hyperparameters (dict): A dictionary containing the hyperparameters
        
    Return:
        indices: a list of tuples
    """

    window_size = hyperparameters['in_length'] + hyperparameters['target_sequence_length']
    step_size = hyperparameters['step_size']
    stop_position = len(data) - 1

    subseq_first_idx = 0
    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx < stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size

    return indices

def get_x_y(
        indices: list,
        data: pd.DataFrame,
        target_variable: str,
        feature_variable: list,
        target_sequence_length: int,
        input_seq_len: int,
        # target_col: str = 'price_de'
) -> Tuple[np.array, np.array]:
    
    # print ("Preparing data...")
    """
    Obtaining the model inputs and targets (X,Y)
    """
    
    x_data = data[feature_variable].values
    y_data = data[target_variable].values

    for i, idx in enumerate(indices):

        x_instance = x_data[idx[0]:idx[1]]
        y_instance = y_data[idx[0]:idx[1]]

        x = x_instance[0: input_seq_len]
        y = y_instance[input_seq_len:input_seq_len + target_sequence_length]

        assert len(x) == input_seq_len
        assert len(y) == target_sequence_length

        if i == 0:
            X = x.reshape(1, -1)
            Y = y.reshape(1, -1)
        else:
            X = np.concatenate((X, x.reshape(1, -1)), axis=0)
            Y = np.concatenate((Y, y.reshape(1, -1)), axis=0)

    return X, Y

def get_data_slices(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_slices: int = 14,
        slice_length: int = 24,
        target_col: str = 'price_de',
):
    """
    Returning the slices of training set dataframes and their corresponding test set dataframes.

    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        n_slices (int): Number of slices
        slice_length (int): Length of each slice
        target_col (str): Target column name
    Returns:
        train_slices (list): List of training slices
        test_slices (list): List of test slices
    """

    index_cutoffs = [slice_length * i for i in range(n_slices, -1, -1)]
    train_df_slices = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
    index_ceiling = [x.index.stop for x in train_df_slices]
    test_df_slices = [train_df[target_col].iloc[idx:idx + slice_length] if idx != index_ceiling[-1] else test_df[target_col] for idx in index_ceiling]

    return train_df_slices, test_df_slices

def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

def create_sequences(df, seq_length, step_size):

    data = df.values
    data, scaler = normalize_data(data)
    X, Y = [], []
    sequences_dict = {}

    for i in range(len(data) - seq_length - step_size):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + step_size)][:,-1].flatten()

        X.append(x)
        Y.append(y)

    sequences_dict = {'X' : np.array(X), 'y': np.array(Y), 'scaler' : scaler}

    return sequences_dict
