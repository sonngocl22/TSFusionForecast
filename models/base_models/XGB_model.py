from .base_model import BaseModel
import os
import pickle
from utilities import *
from configs import configs

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.multioutput import MultiOutputRegressor


class XGBoostModel(BaseModel):
    def __init__ (self, configs):
        super().__init__(configs)

        """
        Initialize the XGBoost model
        """

        self.configs = configs

    def _tuning_objective(self, trial, train_data, validation_data):

        """
        Objective function for tuning the XGBoost model. The hyperparameters are tuned using Optuna.

        """
    
        n_estimators = trial.suggest_int('n_estimators', *self.configs.XGB_TUNING_PARAMS['n_estimators'])
        max_depth = trial.suggest_int('max_depth', *self.configs.XGB_TUNING_PARAMS['max_depth'])
        learning_rate = trial.suggest_float('learning_rate', *self.configs.XGB_TUNING_PARAMS['learning_rate'], log=True)
        subsample = trial.suggest_float('subsample', *self.configs.XGB_TUNING_PARAMS['subsample'])
        min_child_weight = trial.suggest_int('min_child_weight', *self.configs.XGB_TUNING_PARAMS['min_child_weight'])
        gamma = trial.suggest_float('gamma', *self.configs.XGB_TUNING_PARAMS['gamma'])
        colsample_bytree = trial.suggest_float('colsample_bytree', *self.configs.XGB_TUNING_PARAMS['colsample_bytree'])
        reg_alpha = trial.suggest_float('reg_alpha', *self.configs.XGB_TUNING_PARAMS['reg_alpha'])
        reg_lambda = trial.suggest_float('reg_lambda', *self.configs.XGB_TUNING_PARAMS['reg_lambda'])

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
            seed=self.configs.SEED
        )

        trained_model = MultiOutputRegressor(model).fit(self.x_train, self.y_train)


        x_test = train_data[self.feature_variable].iloc[-self.configs.GLOBAL_PARAMS["in_length"]:].to_numpy().reshape(1, -1)

        y_hat = trained_model.predict(x_test).reshape(-1, 1)

        y = validation_data.to_numpy().reshape(-1, 1)

        return smape_loss(y, y_hat)


    def load_data(self, file_path="data/electricity/"):
        """
        Load training data from a file and slicing into episodic train and test sets.
        Obtaining the x_train and y_train used for model training from the training set.

        :param file_path: path to load the data
        :return:
        """
        self.train_df = pd.read_csv(os.path.join(file_path, 'train_df.csv'))
        self.test_df = pd.read_csv(os.path.join(file_path, 'test_df.csv'))

        self.feature_variable = self.train_df.drop(columns=self.configs.GLOBAL_PARAMS["timestemp_col"]).columns
        self.target_variable = self.configs.GLOBAL_PARAMS["target_col"]

        self.train_df_slices, self.test_df_slices = get_data_slices(train_df=self.train_df, test_df=self.test_df, n_slices=self.configs.GLOBAL_PARAMS["meta_data_len"])

        training_indices = get_indices_entire_sequence(
            data=self.train_df_slices[0],
            hyperparameters=self.configs.GLOBAL_PARAMS)

        self.x_train, self.y_train = get_x_y( 
            indices=training_indices, 
            data=self.train_df_slices[0],
            target_variable=self.target_variable,
            feature_variable=self.feature_variable,
            target_sequence_length=self.configs.GLOBAL_PARAMS["target_sequence_length"],
            input_seq_len=self.configs.GLOBAL_PARAMS["in_length"]
            )

    def train(self, retune=False, best_params_path="models/base_models/best_params"):
        """
        Training the model on loaded data. If retune is set to True, the model will run the tuning and save the best parameters.
        Otherwise it will attempt to load the best parameters from the best_params_path.

        Generates the meta data for the training and test sets and saves them to the specified path.

        :param retune: whether to retune the model
        :param best_params_path: path to save/load the best parameters
        """
        
        if not retune:
            assert os.path.isdir(best_params_path), f"Directory {best_params_path} does not exist."
            assert os.path.isfile(os.path.join(best_params_path, 'best_params_xgb.pkl')), "No best params file found. Please run tuning first."
            with open(os.path.join(best_params_path, 'best_params_xgb.pkl'), 'rb') as fin:
                study = pickle.load(fin)
                best_params = study.best_trial.params
        else:
            print('Tuning XGBoost hyperparameters...')
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self._tuning_objective(trial, 
                                                                train_data=self.train_df_slices[0], 
                                                                validation_data=self.test_df_slices[0]), 
                                         n_trials=self.configs.XGB_TUNING_PARAMS["n_trials"], 
                                         show_progress_bar=True)
            best_params = study.best_trial.params
            with open(os.path.join(best_params_path, 'best_params_xgb.pkl'), 'wb') as fout:
                pickle.dump(study, fout)


        model = xgb.XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            min_child_weight=best_params["min_child_weight"],
            gamma=best_params["gamma"],
            colsample_bytree=best_params["colsample_bytree"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            objective="reg:squarederror",
            tree_method="hist",
            seed=self.configs.SEED
        )

        print(f"Training the XGBoost model...")
        self.trained_model = MultiOutputRegressor(model).fit(self.x_train, self.y_train)

    def generate_meta(self, meta_data_path="models/meta_data"):
        """
        Generating the meta data for the training and test sets and saving them to the specified path.
        """
    
        y_hat_full = np.empty((0, 1))
        for i, train_df_slice in enumerate(self.train_df_slices):

            x_test = train_df_slice[self.feature_variable].iloc[-self.configs.GLOBAL_PARAMS["in_length"]:].to_numpy().reshape(1, -1)
            y_hat = self.trained_model.predict(x_test).reshape(-1, 1)
            y = self.test_df_slices[i].to_numpy().reshape(-1, 1)

            print(f'The XGBoost SMAPE loss for {i} slice: {smape_loss(y, y_hat)}')

            y_hat_full = np.vstack((y_hat_full, y_hat))

        y_hat_df = pd.DataFrame({'y_hat_xgb': y_hat_full.flatten()})
        if not os.path.isdir(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}')):
            os.makedirs(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}'))

        y_hat_df.to_csv(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}',f'y_hat_df_xgb_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.csv'), index=False)


    def save_model(self, file_path="models/base_models/saved_models"):
        """
        Saving the trained model to a file.
        """

        assert os.path.isdir(file_path), f"Directory {file_path} does not exist."

        with open(os.path.join(file_path, f'xgb_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pkl'), 'wb') as fout:
            pickle.dump(self.trained_model, fout)

    def load_model(self, file_path="models/base_models/saved_models"):
        """
        Loading the trained model from a file.
        """

        assert os.path.isdir(file_path), f"Directory {file_path} does not exist."
        assert os.path.isfile(os.path.join(file_path, f'xgb_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pkl')), "No model file found. Please run training first."

        with open(os.path.join(file_path, f'xgb_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pkl'), 'rb') as fin:
            self.trained_model = pickle.load(fin)
        











        




