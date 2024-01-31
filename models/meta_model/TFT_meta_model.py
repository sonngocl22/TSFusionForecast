import os
import pickle
from utilities import *
from configs import configs
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import Trainer

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

class TFTMetaModel():
    def __init__(self, configs, with_covariates=True, bm_length = 14):
        """
        Initialize the TFT meta model
        """
    
        self.configs = configs
        self.with_covariates = with_covariates
        self.bm_length = bm_length
        self.seq_length = self.bm_length * self.configs.GLOBAL_PARAMS["target_sequence_length"]
        pl.seed_everything(self.configs.SEED)

    def load_data(self, file_path="data/electricity/", meta_data_path=f"models/meta_data/", meta_models = ['xgb', 'lgb', 'lstm', 'gru']):

        """
        Loading the base learner forecasts as the past and future meta-data for training the TFT meta-model

        :param file_path: path to load the data
        """
        self.train_df = pd.read_csv(os.path.join(file_path, 'train_df.csv'))
        self.test_df = pd.read_csv(os.path.join(file_path, 'test_df.csv'))
        self.meta_models = meta_models

        assert os.path.isdir(os.path.join(meta_data_path, f"bm{self.bm_length}")), f"The specific meta-data with the length {self.bm_length} does not exist. Please generate it by adjusting the 'meta_data_len' and retraining the base models."

        # past covariates and future covariates dataframes of the meta-data
        self.train_df_ts = self.train_df[[self.configs.GLOBAL_PARAMS["timestemp_col"]]].iloc[-self.seq_length:].copy()
        self.test_df_ts = self.test_df[[self.configs.GLOBAL_PARAMS["timestemp_col"]]].copy()
        for meta_model in self.meta_models:
            y_hat_df = pd.read_csv(os.path.join(meta_data_path, f"bm{self.bm_length}", f"y_hat_df_{meta_model}_bm{self.bm_length}.csv"))

            self.train_df_ts[meta_model] = y_hat_df.iloc[:-self.configs.GLOBAL_PARAMS["target_sequence_length"]].values
            self.test_df_ts[meta_model] = y_hat_df.iloc[-self.configs.GLOBAL_PARAMS["target_sequence_length"]:].values

        self.train_df_ts['y'] = self.train_df[self.configs.GLOBAL_PARAMS["target_col"]].iloc[-self.seq_length:].values
        self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = pd.to_datetime(self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]])
        self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = (self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] - self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].min()).dt.total_seconds() // 3600 + 1 #df_train_val['ds'].max() + 1
        self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].astype(int)
        self.train_df_ts['unique_id'] = 'H1'


        # adding additional covariates into the past covariates meta-data if specified
        if self.with_covariates:
            unknown_cov = self.train_df.drop(columns=[self.configs.GLOBAL_PARAMS["timestemp_col"],self.configs.GLOBAL_PARAMS["target_col"]]).iloc[-self.seq_length:]
            self.unknown_cov_cols = list(unknown_cov.columns)
            self.train_df_ts = pd.concat([self.train_df_ts, unknown_cov], axis = 1)
        else:
            self.unknown_cov_cols = []

        self.test_df_ts['y'] = self.test_df[self.configs.GLOBAL_PARAMS["target_col"]].values
        self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = pd.to_datetime(self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]])
        self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = (self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] - self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].min()).dt.total_seconds() // 3600 + self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].max() + 1
        self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] = self.test_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].astype(int)
        self.test_df_ts['unique_id'] = 'H1'

        return self.train_df_ts, self.test_df_ts

        
    def train(self, retune=False, retrain=True, file_path="models/meta_model", best_params_path="models/meta_model/best_params"):
        """
        Traines the model on loaded data. If retune is set to True, the model will run the tuning and save the best parameters.
        Otherwise it will attempt to load the best parameters from the best_params_path.

        Traines the TFT meta-model
        """

        training = TimeSeriesDataSet(
            self.train_df_ts.iloc[:-configs.GLOBAL_PARAMS["target_sequence_length"]],
            time_idx=self.configs.GLOBAL_PARAMS["timestemp_col"],
            target="y",
            group_ids=['unique_id'],
            max_encoder_length=self.configs.TFT_TUNING_PARAMS["max_encoder_length"],
            max_prediction_length=self.configs.TFT_TUNING_PARAMS["max_prediction_length"],
            min_prediction_length=self.configs.TFT_TUNING_PARAMS["max_prediction_length"] // 2,
            time_varying_known_reals=self.meta_models,  # Base model forecasts
            time_varying_unknown_reals=self.unknown_cov_cols,   # Additional covariates
            target_normalizer=GroupNormalizer(
                groups=["unique_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
            )
        
        validation = TimeSeriesDataSet.from_dataset(training, self.train_df_ts, predict=True, stop_randomization=True)

        # creating the test data that includes the encoder and decoder data
        encoder_data = self.train_df_ts[lambda x: x[self.configs.GLOBAL_PARAMS["timestemp_col"]] > x[self.configs.GLOBAL_PARAMS["timestemp_col"]].max() - self.configs.TFT_TUNING_PARAMS["max_encoder_length"]]
        # self.test_df_ts.y = self.train_df_ts.y[self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] == self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].max()].values[0]
        # self.test_df_ts[list(self.unknown_cov_cols)] = self.train_df_ts[list(self.unknown_cov_cols)][self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]] == self.train_df_ts[self.configs.GLOBAL_PARAMS["timestemp_col"]].max()].values[0]
        self.test_df_ts[list(self.unknown_cov_cols)] = self.test_df[list(self.unknown_cov_cols)]
        decoder_data = self.test_df_ts
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

        batch_size = self.configs.TFT_TUNING_PARAMS['batch_size']  # set this between 32 to 128
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=self.configs.TFT_TUNING_PARAMS["patience"], verbose=False, mode="min")

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), 'models/meta_model/my_checkpoints'),  # Change directory path
            save_top_k=1,  # Keep top 3 models
            verbose=True,
            monitor='val_loss',
            mode='min'
        )

        ### Tuning if specified
        if not retune:
            assert os.path.isdir(best_params_path), f"Directory {best_params_path} does not exist."
            assert os.path.isfile(os.path.join(best_params_path, 'best_params_meta_tft.pkl')), "No best params file found. Please run tuning first."
            with open(os.path.join(best_params_path, 'best_params_meta_tft.pkl'), 'rb') as fin:
                study = pickle.load(fin)
                best_params = study.best_trial.params
        else:
            print('Tuning TFT meta model hyperparameters...')

            study = optimize_hyperparameters(
                    train_dataloader,
                    val_dataloader,
                    model_path="optuna_tune_test",
                    n_trials = self.configs.TFT_TUNING_PARAMS["n_trials"],
                    max_epochs = self.configs.TFT_TUNING_PARAMS["max_epochs"],
                    gradient_clip_val_range=self.configs.TFT_TUNING_PARAMS["gradient_clip_val"],
                    hidden_size_range=self.configs.TFT_TUNING_PARAMS["hidden_size_range"],
                    hidden_continuous_size_range=self.configs.TFT_TUNING_PARAMS["hidden_continuous_size"],
                    attention_head_size_range=self.configs.TFT_TUNING_PARAMS["attention_head_size_range"],
                    learning_rate_range=self.configs.TFT_TUNING_PARAMS["learning_rate_range"],
                    dropout_range=self.configs.TFT_TUNING_PARAMS["dropout_range"],
                    trainer_kwargs=dict(#limit_train_batches=50,
                                        enable_checkpointing=False,
                                        callbacks=[early_stop_callback]),
                    reduce_on_plateau_patience=10,
                    use_learning_rate_finder=False,
                )            
            
            best_params = study.best_trial.params

            with open(os.path.join(best_params_path, 'best_params_meta_tft.pkl'), 'wb') as fout:
                pickle.dump(study, fout)


        ### Training the model on best hyperparameters if specified. If retune is set to True, the model will always be retrained.
                
        if retune or retrain:
            print('Training TFT meta model...')
        
            trainer = pl.Trainer(
                max_epochs=self.configs.TFT_TUNING_PARAMS["max_epochs"],
                accelerator="gpu",
                gradient_clip_val=best_params['gradient_clip_val'],
                # limit_train_batches=50,  # coment in for training, running valiation every 30 batches
                # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
                # callbacks=[early_stop_callback, checkpoint_callback],
                callbacks=[checkpoint_callback],
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=True
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
                reduce_on_plateau_patience=self.configs.TFT_TUNING_PARAMS["reduce_on_plateau_patience"],
            )

            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            with open(os.path.join(file_path, 'best_model_path.txt'), 'w') as file:
                file.write(trainer.checkpoint_callback.best_model_path)

            best_model_path = trainer.checkpoint_callback.best_model_path
        
        else:

            with open(os.path.join(file_path, 'best_model_path.txt'), 'r') as file: 
                best_model_path = file.read()


        ### load the best model according to the validation loss
        print(f"best trial: {best_model_path}")
        self.best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        self.new_raw_predictions = self.best_tft.predict(new_prediction_data, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

        print(f"Predicted resuts: {self.new_raw_predictions.output.prediction.cpu().numpy().flatten()}")

        print(f"Test sMAPE: {smape_loss(self.new_raw_predictions.output.prediction.cpu().numpy().flatten(), self.test_df['price_de'].values)}")


    def plot(self):
        """
        Plots the predictions of the model
        """

        self.best_tft.plot_prediction(self.new_raw_predictions.x, 
                                      self.new_raw_predictions.output, 
                                      idx=0, 
                                      add_loss_to_title=True, 
                                      show_future_observed = True)

        




