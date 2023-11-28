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

# M4 dataset fetch and evaluations
from ESRNN.m4_data import *
from ESRNN.utils_evaluation import evaluate_prediction_owa
from ESRNN.utils_visualization import plot_grid_prediction

# getting directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_dir = os.path.join(base_dir, 'code')
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
best_params_dir = os.path.join(code_dir, 'best_params')
os.makedirs(best_params_dir, exist_ok=True)
pl.seed_everything(42)

# parsing arguments (args.job_index and args.total_jobs)
parser = argparse.ArgumentParser()
parser.add_argument('--job-index', type=int, required=True, help='Job index number')
parser.add_argument('--total-jobs', type=int, required=True, help='Total number of jobs')
args = parser.parse_args()

# tuning parameters
batch_size = 64
n_trials = 200
max_epochs = 60
gradient_clip_val_range=(0.1, 1.0)
hidden_size_range=(50, 250)
hidden_continuous_size_range=(50, 250)
attention_head_size_range=(1, 4)
learning_rate_range=(0.001, 0.1)
dropout_range=(0.1, 0.5)

# loading datasets
X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name="Hourly",
                                                               directory=os.path.join(data_dir, 'M4'),
                                                               num_obs=414)

unique_ids = y_train_df['unique_id'].unique()
all_forecasts = {}

results_train_dir = os.path.join(results_dir, 'm4', 'base_model_train_set')
# base model forecasts used during training
df_arima_train = pd.read_csv(os.path.join(results_train_dir, 'y_hat_df_arima_ts.csv'))
df_theta_train = pd.read_csv(os.path.join(results_train_dir, 'y_hat_df_theta_ts.csv'))
df_xgb_train = pd.read_csv(os.path.join(results_train_dir, 'y_hat_df_xgb_ts.csv'))
df_gru_train = pd.read_csv(os.path.join(results_train_dir, 'y_hat_df_gru_ts.csv'))
df_lstm_train = pd.read_csv(os.path.join(results_train_dir, 'y_hat_df_lstm_ts.csv'))

results_test_dir = os.path.join(results_dir, 'm4', 'base_model_test_set')
# base model forecasts used during testing
df_arima_test = pd.read_csv(os.path.join(results_test_dir, 'y_hat_df_arima.csv'))
df_theta_test = pd.read_csv(os.path.join(results_test_dir, 'y_hat_df_theta.csv'))
df_xgb_test = pd.read_csv(os.path.join(results_test_dir, 'y_hat_df_xgb.csv'))
df_gru_test = pd.read_csv(os.path.join(results_test_dir, 'y_hat_df_gru.csv'))
df_lstm_test = pd.read_csv(os.path.join(results_test_dir, 'y_hat_df_lstm.csv'))


# getting job indexes
total_datasets = len(unique_ids)
chunk_size = total_datasets // args.total_jobs
start_index = (args.job_index - 1) * chunk_size
end_index = start_index + chunk_size

if args.job_index == args.total_jobs:
    end_index = total_datasets

print(f'start job {args.job_index}: {start_index}')
print(f'end job {args.job_index}: {end_index}')

for unique_id in unique_ids[start_index:end_index]:

    print(f'Currently training: {unique_id}')

    df_base_models_train= pd.DataFrame({
    'unique_id' : df_arima_train.unique_id,
    'y_arima' : df_arima_train.y_hat,
    'y_theta' : df_theta_train.y_hat,
    'y_xgb' : df_xgb_train.y,
    'y_gru' : df_gru_train.y_hat,
    'y_lstm' : df_lstm_train.y_hat
    })

    df_base_models_test= pd.DataFrame({
    'unique_id' : df_arima_test.unique_id,
    'y_arima' : df_arima_test.y_hat,
    'y_theta' : df_theta_test.y_hat,
    'y_xgb' : df_xgb_test.y_hat,
    'y_gru' : df_gru_test.y_hat,
    'y_lstm' : df_lstm_test.y_hat
    })

    # Filter data for the current series (train and val data)
    df = y_train_df[y_train_df['unique_id'] == unique_id].copy()
    df['ds'] = (df['ds'] - df['ds'].min()).dt.total_seconds() // 3600
    df['ds'] = df['ds'].astype(int)
    df_train_val = pd.concat([df.iloc[-24*7:].reset_index(drop=True).drop(columns=['unique_id']), 
                              df_base_models_train[df_base_models_train['unique_id']==unique_id].reset_index(drop=True)], axis=1)
    
    # Test data
    df = y_test_df.drop(columns=['y_hat_naive2'])[y_test_df['unique_id'] == unique_id].copy()
    df['ds'] = (df['ds'] - df['ds'].min()).dt.total_seconds() // 3600 + df_train_val['ds'].max() + 1  #700
    df['ds'] = df['ds'].astype(int)
    df_test = pd.concat([df.reset_index(drop=True).drop(columns=['unique_id']), 
                         df_base_models_test[df_base_models_test['unique_id']==unique_id].reset_index(drop=True)], axis=1)

    # Create the TimeSeriesDataSet for training
    max_encoder_length = 24*7
    max_prediction_length = 48

    training = TimeSeriesDataSet(
        df_train_val.iloc[:-max_prediction_length],
        time_idx="ds",
        target="y",
        group_ids=['unique_id'],
        max_encoder_length=max_encoder_length,
        # min_encoder_length=max_encoder_length // 2,
        min_encoder_length=1,
        max_prediction_length=max_prediction_length,
        # min_prediction_length=max_prediction_length // 2,
        min_prediction_length=1,
        time_varying_known_reals=['y_arima', 'y_theta', 'y_xgb', 'y_gru', 'y_lstm'],  # Base model forecasts
        target_normalizer=GroupNormalizer(
            groups=["unique_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # allow_missing_timesteps=True
        )
    
    validation = TimeSeriesDataSet.from_dataset(training, df_train_val, predict=True, stop_randomization=True)

    # creating the test data that includes the encoder and decoder data
    encoder_data = df_train_val[lambda x: x.ds > x.ds.max() - max_encoder_length]
    df_test.y = df_train_val.y[df_train_val.ds == df_train_val.ds.max()].values[0]
    decoder_data = df_test
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

    batch_size = batch_size  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    ### TUNING
    best_params = {}
    print(f'Tuning has started for {unique_id}...')
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
        trainer_kwargs=dict(limit_train_batches=50,
                            enable_checkpointing=False,
                            callbacks=[]),
        reduce_on_plateau_patience=6,
        use_learning_rate_finder=False,
    )

    with open(os.path.join(best_params_dir, f"study_{unique_id}.pkl"), "wb") as fout:
        pickle.dump(study, fout)

    print(f'Tuning done! Best params for {unique_id}: ')
    print(study.best_trial.params)

    ### END TUNING



