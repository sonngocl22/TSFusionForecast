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

feature_variable = test_df.drop(columns='datetime_utc').columns
target_variable = 'price_de'
timestemp_col = 'datetime_utc'
step_size = 24

params = {
    "seq_length": 24 * 7,             # Sequence length
    "target_seq_length": 24,          # Target sequence length for forecasting
    "input_size": len(feature_variable),     # Input size
    "output_size": len(feature_variable),                 # Output size
}

# loading base model forecasts as train and test sets
y_hat_xgb = pd.read_csv(os.path.join(base_models_ts, 'y_hat_df_xgb.csv'))
y_hat_lgb = pd.read_csv(os.path.join(base_models_ts, 'y_hat_df_lgb.csv'))
y_hat_gru = pd.read_csv(os.path.join(base_models_ts, 'y_hat_df_gru.csv'))
y_hat_lstm = pd.read_csv(os.path.join(base_models_ts, 'y_hat_df_lstm.csv'))

# creating train and val datasets
# train_df_xgb = train_df_lgb = train_df_gru = train_df_lstm = train_df[[timestemp_col]].iloc[-params['seq_length']:].copy()
train_df_ts = train_df[[timestemp_col]].iloc[-params['seq_length']:].copy()

train_df_ts['y_hat_xgb'] = y_hat_xgb['y_hat_xgb'].iloc[:-params['target_seq_length']].values
train_df_ts['y_hat_lgb'] = y_hat_lgb['y_hat_lgb'].iloc[:-params['target_seq_length']].values
train_df_ts['y_hat_gru'] = y_hat_gru['y_hat_gru'].iloc[:-params['target_seq_length']].values
train_df_ts['y_hat_lstm'] = y_hat_lstm['y_hat_lstm'].iloc[:-params['target_seq_length']].values
train_df_ts['y'] = train_df['price_de'].iloc[-params['seq_length']:].values

train_df_ts['datetime_utc'] = pd.to_datetime(train_df_ts['datetime_utc'])
train_df_ts['datetime_utc'] = (train_df_ts['datetime_utc'] - train_df_ts['datetime_utc'].min()).dt.total_seconds() // 3600 + 1 #df_train_val['ds'].max() + 1
train_df_ts['datetime_utc'] = train_df_ts['datetime_utc'].astype(int)
train_df_ts['unique_id'] = 'H1'


# creating test dataset
# test_df_xgb = test_df_lgb = test_df_gru = test_df_lstm = test_df[[timestemp_col]].copy()
test_df_ts = test_df[[timestemp_col]].copy()

test_df_ts['y_hat_xgb'] = y_hat_xgb['y_hat_xgb'].iloc[-params['target_seq_length']:].values
test_df_ts['y_hat_lgb'] = y_hat_lgb['y_hat_lgb'].iloc[-params['target_seq_length']:].values
test_df_ts['y_hat_gru'] = y_hat_gru['y_hat_gru'].iloc[-params['target_seq_length']:].values
test_df_ts['y_hat_lstm'] = y_hat_lstm['y_hat_lstm'].iloc[-params['target_seq_length']:].values
test_df_ts['y'] = test_df['price_de'].values

test_df_ts['datetime_utc'] = pd.to_datetime(test_df_ts['datetime_utc'])
test_df_ts['datetime_utc'] = (test_df_ts['datetime_utc'] - test_df_ts['datetime_utc'].min()).dt.total_seconds() // 3600 + train_df_ts['datetime_utc'].max() + 1
test_df_ts['datetime_utc'] = test_df_ts['datetime_utc'].astype(int)
test_df_ts['unique_id'] = 'H1'


print(train_df_ts.shape)
print(test_df_ts.shape)



# Create the TimeSeriesDataSet for training
max_encoder_length = 24
# min_encoder_length = 48
max_prediction_length = 24

training = TimeSeriesDataSet(
    train_df_ts.iloc[:-params['target_seq_length']],
    time_idx="datetime_utc",
    target="y",
    group_ids=['unique_id'],
    max_encoder_length=max_encoder_length,
    # min_encoder_length=min_encoder_length,
    min_encoder_length=max_encoder_length // 2,
    # min_encoder_length=1,
    max_prediction_length=max_prediction_length,
    min_prediction_length=max_prediction_length // 2,
    # min_prediction_length=1,
    time_varying_known_reals=['y_hat_xgb', 'y_hat_lgb', 'y_hat_gru', 'y_hat_lstm'],  # Base model forecasts
    target_normalizer=GroupNormalizer(
        groups=["unique_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
    )

validation = TimeSeriesDataSet.from_dataset(training, train_df_ts, predict=True, stop_randomization=True)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=50, verbose=False, mode="min")

# creating the test data that includes the encoder and decoder data
encoder_data = df_train_val[lambda x: x.ds > x.ds.max() - max_encoder_length]
df_test.y = df_train_val.y[df_train_val.ds == df_train_val.ds.max()].values[0]
decoder_data = df_test
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

batch_size = batch_size  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
