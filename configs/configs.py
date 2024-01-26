
GLOBAL_PARAMS = {
    "in_length": 7*24,                  # Using 7 days (168 hours) of past observations
    "target_sequence_length": 24,       # Forecasting 24 hours ahead
    "step_size": 24,                    # Using a step size of 24 hours
    "timestemp_col" : 'datetime_utc',   # Name of the timestamp column
    "target_col" : 'price_de',          # Name of the target column
    "meta_data_len": 14,                # The length of the generated training meta-data (e.g. 14 days)
}

XGB_TUNING_PARAMS = {
    "n_trials": 30,
    "n_estimators": (50, 250),
    "max_depth": (5, 10),
    "learning_rate": (0.01, 0.1),
    "subsample": (0.5, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0, 0.5),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (0, 1),
    "reg_lambda": (0, 1),
}

LGB_TUNING_PARAMS = {
    "n_trials": 30,
    "n_estimators": (50, 300),
    "max_depth": (3, 12),
    "learning_rate": (0.005, 0.05),
    "subsample": (0.05, 1.0),
    "min_data_in_leaf": (5, 25),
    "num_leaves": (5, 25),
    "min_gain_to_split": (0, 10),
    "lambda_l1": (0, 10),
    "lambda_l2": (0, 10),
}

GRU_TUNING_PARAMS = {
    "n_trials": 30,
    "train_epochs": 500,
    "tune_epochs": 200,
    "learning_rate": (1e-5, 1e-1),
    "hidden_size": (10, 65),
    "num_layers": (2, 12),
    "batch_size": [32, 64, 128],
    "dropout": (0.1, 0.5),
}

LSTM_TUNING_PARAMS = {
    "n_trials": 3,
    "train_epochs": 60,
    "tune_epochs": 2,
    "learning_rate": (1e-5, 1e-1),
    "hidden_size": (10, 60),
    "num_layers": (2, 12),
    "batch_size": [32, 64, 128],
    "dropout": (0.1, 0.5),
}

SEED = 42

TEST = True
