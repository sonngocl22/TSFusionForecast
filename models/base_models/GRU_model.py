from .base_model import BaseModel
import os
import pickle
from utilities import *
from configs import configs

import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(configs.SEED)
torch.manual_seed(configs.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the architecture of the gru model
class _GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        gru_out, _ = self.gru(x, h0)
        out = self.fc(gru_out[:, -1, :])
        return out


class GRUModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        """
        Initialize the GRU model
        """

        self.configs = configs


    def _train_model(self, model, criterion, optimizer, X_train, y_train, batch_size, epochs):

        """
        Train the GRU model
        """
        dataset = TensorDataset(X_train, y_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()

            for batch_idx, (sequences, targets) in enumerate(data_loader):
                sequences, targets = sequences.to(device), targets.to(device)

                optimizer.zero_grad()
                pred = model(sequences)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

        # generating forecasts
        model.eval()
        last_sequence = X_train[-1:].to(device) # [1, 168, 16]
        forecast_seq = model(last_sequence)
        
        return model, forecast_seq


    def _tuning_objective(self, trial, train_data, validation_data):

        """
        Objective function for tuning the GRU model. The hyperparameters are tuned using Optuna.
        """

        learning_rate = trial.suggest_float("learning_rate", *self.configs.GRU_TUNING_PARAMS['learning_rate'], log=True)
        hidden_size = trial.suggest_int("hidden_size", *self.configs.GRU_TUNING_PARAMS['hidden_size'], step=5)
        num_layers = trial.suggest_int("num_layers", *self.configs.GRU_TUNING_PARAMS['num_layers'])
        batch_size = trial.suggest_categorical("batch_size", self.configs.GRU_TUNING_PARAMS['batch_size'])
        dropout = trial.suggest_float("dropout", *self.configs.GRU_TUNING_PARAMS['dropout'])

        sequences_dict = create_sequences(train_data[self.feature_variable], self.config.GLOBAL_PARAMS['in_length'], self.config.GLOBAL_PARAMS['step_size'])

        X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))
        y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))


        model = _GRU_Model(len(self.feature_variable), hidden_size, num_layers, self.config.GLOBAL_PARAMS['target_sequence_length'], dropout)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        _, forecast_seq = self._train_model(model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            X_train=X_train,
                                            y_train=y_train,
                                            batch_size=batch_size,
                                            epochs=self.configs.GRU_TUNING_PARAMS['train_epochs'],)
        
        scaler = sequences_dict['scaler']
        forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[self.target_index] + scaler.mean_[self.target_index]
        loss = smape_loss(forecast_seq_descaled.flatten(), validation_data.values)

        return loss.item()


    def load_data(self, file_path="data/electricity/"):

        """
        Load training data from a file and slicing into episodic train and test sets.
        Obtaining the x_train and y_train used for model training from the training set.

        :param file_path: path to load the data
        :return:
        """

        self.train_df = pd.read_csv(os.path.join(file_path, 'train_df.csv'))
        self.test_df = pd.read_csv(os.path.join(file_path, 'test_df.csv'))

        self.feature_variable = self.train_df.drop(columns=self.configs.GLOBAL_PARAMS['timestemp_col']).columns
        self.target_variable = self.configs.GLOBAL_PARAMS['target_col']

        self.target_index = self.train_df[self.feature_variable].columns.get_loc(self.target_variable)


        self.train_df_slices, self.test_df_slices = get_data_slices(train_df=self.train_df, test_df=self.test_df, n_slices=self.configs.GLOBAL_PARAMS["meta_data_len"])

        # sequences_dict = create_sequences(self.train_df_slices[self.feature_variable], self.config.GLOBAL_PARAMS['in_length'], self.config.GLOBAL_PARAMS['step_size'])

        # self.X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))
        # self.y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))
        # self.scaler = sequences_dict['scaler']


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
            assert os.path.isfile(os.path.join(best_params_path, 'best_params_gru.pkl')), "No best params file found. Please run tuning first."
            with open(os.path.join(best_params_path, 'best_params_gru.pkl'), 'rb') as fin:
                study = pickle.load(fin)
                best_params = study.best_trial.params
        else:
            print('Tuning GRU hyperparameters...')
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self._tuning_objective(trial, 
                                                                train_data=self.train_df_slices[0], 
                                                                validation_data=self.test_df_slices[0]), 
                                         n_trials=self.configs.GRU_TUNING_PARAMS["n_trials"], 
                                         show_progress_bar=True)
            best_params = study.best_trial.params
            with open(os.path.join(best_params_path, 'best_params_gru.pkl'), 'wb') as fout:
                pickle.dump(study, fout)

        sequences_dict = create_sequences(self.train_df_slices[0][self.feature_variable], self.config.GLOBAL_PARAMS['in_length'], self.config.GLOBAL_PARAMS['step_size'])
        X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))
        y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))

        model = _GRU_Model(len(self.feature_variable), best_params['hidden_size'], best_params['num_layers'], self.config.GLOBAL_PARAMS['target_sequence_length'], best_params['dropout'])
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        self.model_gru, forecast_seq = self._train_model(model,
                                                         criterion=criterion,
                                                         optimizer=optimizer,
                                                         X_train=X_train,
                                                         y_train=y_train,
                                                         batch_size=best_params['batch_size'],
                                                         epochs=configs.GRU_TUNING_PARAMS['train_epochs'],)
        

    def generate_meta(self, meta_data_path="models/meta_data"):
        """
        Generating the meta data for the training and test sets and saving them to the specified path.
        """
        
        y_hat_full = np.empty((0, 1))
        for i, train_df_slice in enumerate(self.train_df_slices):

            sequences_dict = create_sequences(train_df_slice[self.feature_variable], self.configs.GLOBAL_PARAMS['in_length'], self.configs.GLOBAL_PARAMS['step_size'])

            X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
            y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))#.unsqueeze(1)

            model = self.model_gru
            model.eval()
            last_sequence = X_train[-1:].to(device) # [1, 168, 16]
            forecast_seq = model(last_sequence)

            scaler = sequences_dict['scaler']
            forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[self.target_index] + scaler.mean_[self.target_index]

            loss_smape = smape_loss(forecast_seq_descaled.flatten(), self.test_df_slices[i].to_numpy().flatten())

            print(f'The GRU SMAPE loss for {i} slice: {loss_smape.item()}')

            y_hat_full = np.vstack((y_hat_full, forecast_seq_descaled.reshape(-1, 1)))

        if not os.path.isdir(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}')):
            os.makedirs(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}'))

        y_hat_df = pd.DataFrame({'y_hat_gru': y_hat_full.flatten()})
        y_hat_df.to_csv(os.path.join(meta_data_path, f'bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}', f'y_hat_df_gru_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.csv'), index=False)


    def save_model(self, file_path="models/base_models/saved_models"):
        """
        Saving the trained model to a file.
        """

        assert os.path.isdir(file_path), f"Directory {file_path} does not exist."

        torch.save(self.model_gru.state_dict(), os.path.join(file_path, f'gru_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pth'))


    def load_model(self, file_path="models/base_models/saved_models", best_params_path="models/base_models/best_params"):
        """
        Loading the trained model from a file.
        """

        assert os.path.isdir(file_path), f"Directory {file_path} does not exist."
        assert os.path.isfile(os.path.join(file_path, f'gru_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pt')), "No model file found. Please run training first."

        assert os.path.isfile(os.path.join(best_params_path, 'best_params_gru.pkl')), "No best params file found. Please run tuning first."
        with open(os.path.join(best_params_path, 'best_params_gru.pkl'), 'rb') as fin:
            study = pickle.load(fin)
            best_params = study.best_trial.params        

        self.model_gru = _GRU_Model(len(self.feature_variable), best_params['hidden_size'], best_params['num_layers'], self.config.GLOBAL_PARAMS['target_sequence_length'], best_params['dropout'])
        self.model_gru.load_state_dict(torch.load(os.path.join(file_path, f'gru_model_bm{self.configs.GLOBAL_PARAMS["meta_data_len"]}.pt')))



