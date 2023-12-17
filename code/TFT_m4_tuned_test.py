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
best_params_dir = os.path.join(code_dir, 'best_params_bm14')
# os.makedirs(best_params_dir, exist_ok=True)
pl.seed_everything(22)

# setting parameters
batch_size = 64
patience = 50
max_epochs = 200

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

# getting the most relevant training size
train_window = len(df_arima_train[df_arima_train['unique_id']=='H1']) # 48*7

for unique_id in unique_ids:
# for unique_id in ['H1','H10']:

    print(f'Currently training: {unique_id}')

    df_base_models_train= pd.DataFrame({
    'unique_id' : df_arima_train.unique_id,
    'y_arima' : df_arima_train.y_hat,
    'y_theta' : df_theta_train.y_hat,
    'y_xgb' : df_xgb_train.y_hat,
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
    df_train_val = pd.concat([df.iloc[-train_window:].reset_index(drop=True).drop(columns=['unique_id']), 
                              df_base_models_train[df_base_models_train['unique_id']==unique_id].reset_index(drop=True)], axis=1)
    
    # Test data
    df = y_test_df.drop(columns=['y_hat_naive2'])[y_test_df['unique_id'] == unique_id].copy()
    df['ds'] = (df['ds'] - df['ds'].min()).dt.total_seconds() // 3600 + df_train_val['ds'].max() + 1  #700
    df['ds'] = df['ds'].astype(int)
    df_test = pd.concat([df.reset_index(drop=True).drop(columns=['unique_id']), 
                         df_base_models_test[df_base_models_test['unique_id']==unique_id].reset_index(drop=True)], axis=1)

    # Create the TimeSeriesDataSet for training
    max_encoder_length = 48
    # min_encoder_length = 48
    max_prediction_length = 48

    training = TimeSeriesDataSet(
        df_train_val.iloc[:-max_prediction_length],
        time_idx="ds",
        target="y",
        group_ids=['unique_id'],
        max_encoder_length=max_encoder_length,
        # min_encoder_length=min_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        # min_encoder_length=1,
        max_prediction_length=max_prediction_length,
        min_prediction_length=max_prediction_length // 2,
        # min_prediction_length=1,
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
        reduce_on_plateau_patience=20,
    )

    print(best_params)

    trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    )

    new_raw_predictions = tft.predict(new_prediction_data, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu"))
    prediction_array = new_raw_predictions.output.prediction.cpu().numpy().flatten()
    all_forecasts[unique_id] = prediction_array

    y = y_test_df[y_test_df['unique_id'] == unique_id].y.to_numpy()
    print(np.sqrt(((y - prediction_array)**2).sum()))

#print(all_forecasts)

results_save_dir = os.path.join(results_dir, 'm4', 'TFT', 'test')
df_save = pd.DataFrame(all_forecasts).melt()
df_save.rename(columns={'variable' : 'unique_id', 'value': 'y_hat'}, inplace=True)
df_save['ds'] = X_test_df['ds']
df_save.to_csv(os.path.join(results_save_dir, 'y_hat_df_tft_bm14_tuned_final.csv'), index=False)
