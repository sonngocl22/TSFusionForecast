import os
import pickle
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# getting directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code_dir = os.path.join(base_dir, 'code')
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
elec_dir = os.path.join(base_dir, 'notebooks', 'Elec_model')

assert os.path.isdir(os.path.join(elec_dir, 'best_params')), "Directory best_params does not exist"
assert os.path.isdir(os.path.join(elec_dir, 'base_models_ts')), "Directory base_models_ts does not exist"


# loading datasets
train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'train_df.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'test_df.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_train_df.csv'))
y_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'y_test_df.csv'))
X_train_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_train_df.csv'))
X_test_df = pd.read_csv(os.path.join(data_dir, 'electricity', 'X_test_df.csv'))

# setting seed and defining all parameters
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# normalizing data and output scaler
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)#(data.reshape(-1, 1))
    # scaled_data = scaled_data.flatten()

    return scaled_data, scaler

# generating sequences from data for training
def create_sequences(df, seq_length, target_seq_length):

    data = df.values
    data, scaler = normalize_data(data)
    X, Y = [], []
    sequences_dict = {}

    # price_de_idx = df.columns.get_loc('price_de')

    for i in range(len(data) - seq_length - target_seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        # y = data[(i + seq_length):(i + seq_length+target_seq_length)]

        X.append(x)
        Y.append(y)

    sequences_dict = {'X' : np.array(X), 'y': np.array(Y), 'scaler' : scaler}

    return sequences_dict

# defining the architecture of the LSTM model
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out
    

def smape_loss(y_true, y_pred):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
    y_true (torch.Tensor): The true values.
    y_pred (torch.Tensor): The predicted values.

    Returns:
    torch.Tensor: The SMAPE value.
    """
    epsilon = torch.finfo(y_true.dtype).eps
    denominator = torch.max(torch.abs(y_true) + torch.abs(y_pred) + epsilon, torch.tensor(0.5 + epsilon).to(y_true.device))

    diff = 2 * torch.abs(y_pred - y_true) / denominator
    smape_value = 100 / len(y_true) * torch.sum(diff)
    return smape_value


def train_model(model, 
                criterion, 
                optimizer, 
                X_train, 
                y_train, 
                batch_size,
                epochs):
    
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()

        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            pred = model(sequences)
            loss = criterion(pred, targets.squeeze(1)) # squeeze to match dimensions
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # generating forecasts
    model.eval()
    last_sequence = X_train[-1:].to(device) # [1, 168, 16]
    forecast_seq = torch.Tensor().to(device)
    
    for _ in range(params["target_seq_length"]):
        with torch.no_grad():
            next_step_forecast = model(last_sequence) # [1, 16]
            forecast_seq = torch.cat((forecast_seq, next_step_forecast), dim=0) # [1, 16, 1]
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_step_forecast.unsqueeze(1)), dim=1)
    
    return model, forecast_seq



### Tuner

train_val_dict = {'train_set' : train_df.iloc[:-8*24], 
                  'val_set' : train_df.iloc[-8*24:-8*24+24]}

sequences_dict = create_sequences(train_val_dict['train_set'][feature_variable], params["seq_length"], params["target_seq_length"])

def objective(trial):

    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 5, 50, step=5)
    num_layers = trial.suggest_int("num_layers", 1, 12)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    epochs = trial.suggest_int("epochs", 50, 150, step=10)

    X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
    y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32)).unsqueeze(1)

    model = LSTM_Model(params['input_size'], hidden_size, num_layers, params['output_size'], dropout)
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_lstm, forecast_seq = train_model(model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        X_train=X_train,
                                        y_train=y_train,
                                        batch_size=batch_size,
                                        epochs=epochs)

    forecast_seq_descaled = sequences_dict['scaler'].inverse_transform(forecast_seq.cpu().numpy())

    print(forecast_seq_descaled[:,-1])
    print(train_val_dict['val_set'].iloc[:,-1].values)

    # loss = smape_loss(torch.from_numpy(forecast_seq_descaled[:,-1]), torch.from_numpy(train_val_dict['val_set'].iloc[:,-1].values))
    loss = criterion(torch.from_numpy(forecast_seq_descaled[:,-1]), torch.from_numpy(train_val_dict['val_set'].iloc[:,-1].values))

    return loss.item()

if os.path.exists(os.path.join(elec_dir, 'best_params', 'best_params_lstm.pkl')):
    print('Loading best params...')
    with open(os.path.join(elec_dir, 'best_params', 'best_params_lstm.pkl'), 'rb') as fin:
        study = pickle.load(fin)
        best_params = study.best_trial.params

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

else:
    print('Tuning hyperparameters...')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_value))
    study.optimize(objective, n_trials=30)
    best_params = study.best_trial.params

    with open(os.path.join(elec_dir, 'best_params', 'best_params_lstm.pkl'), 'wb') as fout:
        pickle.dump(study, fout)

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

### Finished tuning hyperparameters

# creating data slices to generate forecasts for the next 8 days
# index_cutoffs = [24*i for i in range(7, -1, -1)]
# train_df_list = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
# index_ceiling = [x.index.stop for x in train_df_list]
# test_df_list = [train_df['price_de'].iloc[idx:idx+step_size] if idx!=index_ceiling[-1] else test_df['price_de'] for idx in index_ceiling]
    
# creating data slices to generate forecasts for the next 15 days
index_cutoffs = [24*i for i in range(14, -1, -1)]
train_df_list = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
index_ceiling = [x.index.stop for x in train_df_list]
test_df_list = [train_df['price_de'].iloc[idx:idx+step_size] if idx!=index_ceiling[-1] else test_df['price_de'] for idx in index_ceiling]

y_hat_full = np.empty((0, 1))

for i, train_df_slice in enumerate(train_df_list):

    sequences_dict = create_sequences(train_df_slice[feature_variable], params["seq_length"], params["target_seq_length"])

    all_forecast_seq_descaled = []

    X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
    y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32)).unsqueeze(1)

    model = LSTM_Model(params['input_size'], best_params['hidden_size'], best_params['num_layers'], params['output_size'], best_params['dropout'])
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    model_lstm, forecast_seq = train_model(model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        X_train=X_train,
                                        y_train=y_train,
                                        batch_size=best_params['batch_size'],
                                        epochs=200)

    forecast_seq_descaled = sequences_dict['scaler'].inverse_transform(forecast_seq.cpu().numpy())

    loss_smape = smape_loss(torch.from_numpy(forecast_seq_descaled[:,-1].reshape(-1, 1)), torch.from_numpy(test_df_list[i].to_numpy().reshape(-1, 1)))
    loss_huber = criterion(torch.from_numpy(forecast_seq_descaled[:,-1].reshape(-1, 1)), torch.from_numpy(test_df_list[i].to_numpy().reshape(-1, 1)))

    print(f'The SMAPE loss for {i}: {loss_smape.item()}')
    print(f'The Huber loss for {i}: {loss_huber.item()}')

    y_hat_full = np.vstack((y_hat_full, forecast_seq_descaled[:,-1].reshape(-1, 1)))

# save the forecasts
y_hat_df = pd.DataFrame({'y_hat_lstm': y_hat_full.flatten()})
y_hat_df.to_csv(os.path.join(elec_dir, 'base_models_ts', 'y_hat_df_lstm_bm14.csv'), index=False)

# asserting that the length of the forecast is correct
print(f'y_hat length: {len(y_hat_full)} vs correct length: {params["seq_length"] + params["target_seq_length"]}')

