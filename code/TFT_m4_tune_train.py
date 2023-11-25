import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
import torch
import numpy as np
import pandas as pd
import pickle

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
pl.seed_everything(42)


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

print(df_arima_train.head())
print(df_arima_test.head())

