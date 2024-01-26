import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
import torch
import numpy as np
import pandas as pd
import pickle
import argparse

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import warnings
# To ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*and is already saved during checkpointing*")
warnings.filterwarnings("ignore", ".*The number of training batches*")

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

# getting directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code_dir = os.path.join(base_dir, 'code')
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
elec_dir = os.path.join(base_dir, 'notebooks', 'Elec_model')
base_models_ts = os.path.join(elec_dir, 'base_models_ts')

# loading datasets
train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'train_df.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'test_df.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_train_df.csv'))
y_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_test_df.csv'))
X_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_train_df.csv'))
X_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_test_df.csv'))

unknown_cov = train_df.drop(columns=['datetime_utc','price_de']).iloc[-24*7:]
unknown_cov_cols = unknown_cov.columns

train_df['datetime_utc'] = pd.to_datetime(train_df['datetime_utc'])
train_df['datetime_utc'] = (train_df['datetime_utc'] - train_df['datetime_utc'].min()).dt.total_seconds() // 3600 + 1 #df_train_val['ds'].max() + 1
train_df['datetime_utc'] = train_df['datetime_utc'].astype(int)
train_df['unique_id'] = 'H1'

test_df['datetime_utc'] = pd.to_datetime(test_df['datetime_utc'])
test_df['datetime_utc'] = (test_df['datetime_utc'] - test_df['datetime_utc'].min()).dt.total_seconds() // 3600 + train_df['datetime_utc'].max() + 1
test_df['datetime_utc'] = test_df['datetime_utc'].astype(int)

test_df['unique_id'] = 'H1'

y_true = test_df['price_de'].values.copy()

feature_variable = test_df.drop(columns='datetime_utc').columns
target_variable = 'price_de'
timestemp_col = 'datetime_utc'
step_size = 24
pl.seed_everything(22)

params = {
    "seq_length": 24 * 7,             # Sequence length
    "target_seq_length": 24,          # Target sequence length for forecasting
    "input_size": len(feature_variable),     # Input size
    "output_size": len(feature_variable),                 # Output size
    "batch_size": 128,                 # Batch size
}

# tuning parameters
patience = 5
n_trials = 120
max_epochs = 4
gradient_clip_val_range=(0.1, 20.0)
hidden_size_range=(15, 200)
hidden_continuous_size_range=(5, 50)
attention_head_size_range=(1, 4)
learning_rate_range=(0.0005, 0.1)
dropout_range=(0.1, 0.6)

# [6.574, 9.9278, 9.927, 10.066, 12.795]


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


# Create the TimeSeriesDataSet for training
max_encoder_length = 7*24
min_encoder_length = 24
max_prediction_length = 24

training = TimeSeriesDataSet(
    train_df.iloc[:-params['target_seq_length']],
    time_idx="datetime_utc",
    target="price_de",
    group_ids=['unique_id'],
    max_encoder_length=max_encoder_length,
    min_encoder_length=min_encoder_length,
    # min_encoder_length=max_encoder_length // 2,
    # min_encoder_length=1,
    max_prediction_length=max_prediction_length,
    min_prediction_length=max_prediction_length // 2,
    # min_prediction_length=1,
    # time_varying_known_reals=['y_hat_xgb', 'y_hat_lgb', 'y_hat_gru', 'y_hat_lstm'],  # Base model forecasts
    time_varying_unknown_reals=list(unknown_cov_cols),
    target_normalizer=GroupNormalizer(
        groups=["unique_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
    )

validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=50, verbose=False, mode="min")

# creating the test data that includes the encoder and decoder data
encoder_data = train_df[lambda x: x.datetime_utc > x.datetime_utc.max() - max_encoder_length]
test_df.price_de = train_df.price_de[train_df.datetime_utc == train_df.datetime_utc.max()].values[0]
test_df[list(unknown_cov_cols)] = train_df[list(unknown_cov_cols)][train_df.datetime_utc == train_df.datetime_utc.max()].values[0]
decoder_data = test_df
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

batch_size = params['batch_size']  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=patience, verbose=False, mode="min")

### TUNING

if os.path.exists(os.path.join(elec_dir, 'best_params', 'best_params_tft.pkl')):
    print('Loading best params...')
    with open(os.path.join(elec_dir, 'best_params', 'best_params_tft.pkl'), 'rb') as fin:
        study = pickle.load(fin)
        best_params = study.best_trial.params

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

else:
    print('Tuning hyperparameters...')
    print(f'Tuning has started...')
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_tune_test",
        n_trials = n_trials,
        max_epochs = max_epochs,
        gradient_clip_val_range=gradient_clip_val_range,
        hidden_size_range=hidden_size_range,
        hidden_continuous_size_range=hidden_continuous_size_range,
        attention_head_size_range=attention_head_size_range,
        learning_rate_range=learning_rate_range,
        dropout_range=dropout_range,
        trainer_kwargs=dict(#limit_train_batches=50,
                            enable_checkpointing=False,
                            callbacks=[]),
        reduce_on_plateau_patience=10,
        use_learning_rate_finder=False,
    )

    best_params = study.best_trial.params

    with open(os.path.join(elec_dir, 'best_params', 'best_params_tft.pkl'), 'wb') as fout:
        pickle.dump(study, fout)

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))



### Training model with best params ###

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu",
    gradient_clip_val=best_params['gradient_clip_val'],
    # limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[early_stop_callback],
    logger=False,
    enable_model_summary=False,
    enable_checkpointing=False
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=best_params['learning_rate'],
    hidden_size=best_params['hidden_size'],
    attention_head_size=best_params['attention_head_size'],
    dropout=best_params['dropout'],
    hidden_continuous_size=best_params['hidden_continuous_size'],
    loss=SMAPE(),
    # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=10,
)

print(best_params)

trainer.fit(
tft,
train_dataloaders=train_dataloader,
val_dataloaders=val_dataloader,
)

new_raw_predictions = tft.predict(new_prediction_data, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

print(new_raw_predictions.output.prediction.cpu().numpy().flatten())
print(y_true)

print(smape_loss(new_raw_predictions.output.prediction.cpu().numpy().flatten(), y_true))