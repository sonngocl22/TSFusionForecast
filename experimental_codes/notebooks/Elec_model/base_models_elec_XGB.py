import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple
from sklearn.multioutput import MultiOutputRegressor
import os
import pickle

# getting directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code_dir = os.path.join(base_dir, 'code')
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
elec_dir = os.path.join(base_dir, 'notebooks', 'Elec_model')
# best_params_dir = os.path.join(code_dir, 'best_params_single')
# os.makedirs(best_params_dir, exist_ok=True)

assert os.path.isdir(os.path.join(elec_dir, 'best_params')), "Directory best_params does not exist"
assert os.path.isdir(os.path.join(elec_dir, 'base_models_ts')), "Directory base_models_ts does not exist"

train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'train_df.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'test_df.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_train_df.csv'))
y_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_test_df.csv'))
X_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_train_df.csv'))
X_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_test_df.csv'))

feature_variable = test_df.drop(columns='datetime_utc').columns
target_variable = 'price_de'
timestemp_col = 'datetime_utc'

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
    
    print ("Preparing data...")
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

    print ("Finished preparing data!")

    return X, Y

def smape(y_true, y_pred):
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


### Uncomment this to tune the hyperparameters

import optuna

train_val_dict = {'train_set' : train_df.iloc[:-8*24], 
                  'val_set' : train_df.iloc[-8*24:-8*24+24]}

hyperparameters = {
    "in_length": 7*24,             # Using 7 days (168 hours) of past observations
    "step_size": 24,               # Sliding the window by 24 steps each time
    "target_sequence_length": 24, # Forecasting 48 hours ahead
}


training_indices = get_indices_entire_sequence(
        data=train_val_dict['train_set'],
        hyperparameters=hyperparameters)

x_train, y_train = get_x_y(
        indices=training_indices, 
        data=train_val_dict['train_set'],
        target_variable=target_variable,
        feature_variable=feature_variable,
        target_sequence_length=hyperparameters["target_sequence_length"],
        input_seq_len=hyperparameters["in_length"]
        )

def objective(trial):


    n_estimators = trial.suggest_int('n_estimators', 50, 250)
    max_depth = trial.suggest_int('max_depth', 5, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_float('gamma', 0, 0.5)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        tree_method="hist",
        seed=42
    )

    trained_model = MultiOutputRegressor(model).fit(x_train, y_train)

    x_test = train_val_dict['train_set'][feature_variable].iloc[-hyperparameters["in_length"]:].to_numpy().reshape(1, -1)

    y_hat = trained_model.predict(x_test).reshape(-1, 1)

    y = train_val_dict['val_set'][target_variable].to_numpy().reshape(-1, 1)

    return smape(y, y_hat)

if os.path.exists(os.path.join(elec_dir, 'best_params', 'best_params_xgb.pkl')):
    print('Loading best params...')
    with open(os.path.join(elec_dir, 'best_params', 'best_params_xgb.pkl'), 'rb') as fin:
        study = pickle.load(fin)
        best_params = study.best_trial.params
else:
    print('Tuning hyperparameters...')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_trial.params

    with open(os.path.join(elec_dir, 'best_params', 'best_params_xgb.pkl'), 'wb') as fout:
        pickle.dump(study, fout)

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

### Finished tuning hyperparameters
    

hyperparameters = {
    "in_length": 7*24,             # Using 7 days (168 hours) of past observations
    "step_size": 24,               # Sliding the window by 24 steps each time
    "target_sequence_length": 24,  # Forecasting 24 hours ahead
    "n_estimators": best_params['n_estimators'],           # Number of gradient boosted trees
    "max_depth": best_params['max_depth'],                # Maximum depth of a tree
    "learning_rate": best_params['learning_rate'],
    "subsample": best_params['subsample'],              # Fraction of samples to be used for fitting each tree
    "min_child_weight": best_params['min_child_weight'],         # Minimum sum of instance weight (hessian) needed in a child
    "gamma": best_params['gamma'],
    "colsample_bytree": best_params['colsample_bytree'],       # Fraction of columns to be randomly sampled for each tree
    "reg_alpha": best_params['reg_alpha'],
    "reg_lambda": best_params['reg_lambda']
    # "selected_features": [target_variable]  # Features selected for training the model
}
# creating data slices to generate forecasts for the next 8 days
# index_cutoffs = [24*i for i in range(7, -1, -1)]
# train_df_list = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
# index_ceiling = [x.index.stop for x in train_df_list]
# test_df_list = [train_df['price_de'].iloc[idx:idx+hyperparameters['step_size']] if idx!=index_ceiling[-1] else test_df['price_de'] for idx in index_ceiling]

# creating data slices to generate forecasts for the next 15 days
index_cutoffs = [24*i for i in range(14, -1, -1)]
train_df_list = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
index_ceiling = [x.index.stop for x in train_df_list]
test_df_list = [train_df['price_de'].iloc[idx:idx+hyperparameters['step_size']] if idx!=index_ceiling[-1] else test_df['price_de'] for idx in index_ceiling]
y_hat_full = np.empty((0, 1))

for i, train_df_slice in enumerate(train_df_list):

    training_indices = get_indices_entire_sequence(
            data=train_df_slice,
            hyperparameters=hyperparameters)

    x_train, y_train = get_x_y(
            indices=training_indices, 
            data=train_df_slice,
            target_variable=target_variable,
            feature_variable=feature_variable,
            target_sequence_length=hyperparameters["target_sequence_length"],
            input_seq_len=hyperparameters["in_length"]
            )

    model = xgb.XGBRegressor(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            learning_rate=hyperparameters["learning_rate"],
            subsample=hyperparameters["subsample"],
            min_child_weight=hyperparameters["min_child_weight"],
            gamma=hyperparameters["gamma"],
            colsample_bytree=hyperparameters["colsample_bytree"],
            reg_alpha=hyperparameters["reg_alpha"],
            reg_lambda=hyperparameters["reg_lambda"],
            objective="reg:squarederror",
            tree_method="hist",
            seed=42
        )

    print(f"Training the model {i}...")
    trained_model = MultiOutputRegressor(model).fit(x_train, y_train)
    print("Finished training the model!")

    x_test = train_df_slice[feature_variable].iloc[-hyperparameters["in_length"]:].to_numpy().reshape(1, -1)

    y_hat = trained_model.predict(x_test).reshape(-1, 1)
    y = test_df_list[i].to_numpy().reshape(-1, 1)

    print(f'The SMAPE loss for {i}: {smape(y, y_hat)}')

    y_hat_full = np.vstack((y_hat_full, y_hat))

# assert len(y_hat_full) == hyperparameters['in_length'] + hyperparameters['target_sequence_length']
print(f'y_hat length: {len(y_hat_full)} vs correct length: {hyperparameters["in_length"] + hyperparameters["target_sequence_length"]}')

# save the forecasts
y_hat_df = pd.DataFrame({'y_hat_xgb': y_hat_full.flatten()})
y_hat_df.to_csv(os.path.join(elec_dir, 'base_models_ts', 'y_hat_df_xgb_bm14.csv'), index=False)
