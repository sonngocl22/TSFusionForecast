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

# setting parameters
batch_size = 64
patience = 100
max_epochs = 200

# loading datasets
X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name="Hourly",
                                                               directory=os.path.join(data_dir, 'M4'),
                                                               num_obs=414)

unique_ids = y_train_df['unique_id'].unique()
all_forecasts = {}

for unique_id in unique_ids:

    print(f'Currently training: {unique_id}')

    # Filter data for the current series (train and val data)
    df = y_train_df[y_train_df['unique_id'] == unique_id].copy()
    df['ds'] = (df['ds'] - df['ds'].min()).dt.total_seconds() // 3600
    df['ds'] = df['ds'].astype(int)
    df_train_val = df.iloc[-24*7:].reset_index(drop=True)
    
    # Test data
    df = y_test_df.drop(columns=['y_hat_naive2'])[y_test_df['unique_id'] == unique_id].copy()
    df['ds'] = (df['ds'] - df['ds'].min()).dt.total_seconds() // 3600 + df_train_val['ds'].max() + 1  #700
    df['ds'] = df['ds'].astype(int)
    df_test = df.reset_index(drop=True)

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
        target_normalizer=GroupNormalizer(
            groups=["unique_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # allow_missing_timesteps=Truex
        )
    
    validation = TimeSeriesDataSet.from_dataset(training, df_train_val, predict=True, stop_randomization=True)

    # creating the test data that includes the encoder and decoder data
    encoder_data = df_train_val[lambda x: x.ds > x.ds.max() - max_encoder_length]
    df_test.y = df_train_val.y[df_train_val.ds == df_train_val.ds.max()].values[0]
    decoder_data = df_test
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

    batch_size = 64  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # loading the best parameters from best_params folder
    with open(os.path.join(best_params_dir, f"study_{unique_id}.pkl"), 'rb') as fin:
        study = pickle.load(fin)
    best_params = study.best_trial.params

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=patience, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        gradient_clip_val=best_params['gradient_clip_val'],
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
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
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,

    )

    # predictions = tft.predict(test_dataloader, return_y=False, trainer_kwargs=dict(accelerator="gpu"))
    new_raw_predictions = tft.predict(new_prediction_data, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu"))
    all_forecasts[unique_id] = new_raw_predictions.output.prediction.cpu().numpy().flatten()

results_save_dir = os.path.join(results_dir, 'm4', 'TFT', 'test')
df_save = pd.DataFrame(all_forecasts).melt()
df_save.rename(columns={'variable' : 'unique_id', 'value': 'y_hat'}, inplace=True)
df_save.to_csv(os.path.join(results_save_dir, 'y_hat_df_tft_standalone_test.csv'), index=False)
