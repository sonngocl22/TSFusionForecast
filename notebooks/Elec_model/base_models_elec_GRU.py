import os
import pickle
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    "output_size": 24,                 # Output size
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)#(data.reshape(-1, 1))
    # scaled_data = scaled_data.flatten()

    return scaled_data, scaler


def create_sequences(df, seq_length, target_seq_length):

    data = df.values
    data, scaler = normalize_data(data)
    X, Y = [], []
    sequences_dict = {}

    for i in range(len(data) - seq_length - target_seq_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length+target_seq_length)][:,-1].flatten()

        X.append(x)
        Y.append(y)

    sequences_dict = {'X' : np.array(X), 'y': np.array(Y), 'scaler' : scaler}

    return sequences_dict

# defining the architecture of the gru model
class GRU_Model(nn.Module):
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


def train_model(model, 
                criterion, 
                optimizer, 
                X_train, 
                y_train,
                batch_size,
                epochs):
    
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # droping the last batch to avoid errors

    for epoch in range(epochs):
        model.train()

        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            pred = model(sequences)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # generating forecasts
    model.eval()
    last_sequence = X_train[-1:].to(device) # [1, 168, 16]
    forecast_seq = model(last_sequence)
        
    return model, forecast_seq

### Tuner

train_val_dict = {'train_set' : train_df.iloc[:-8*24], 
                  'val_set' : train_df.iloc[-8*24:-8*24+24]}

sequences_dict = create_sequences(train_val_dict['train_set'][feature_variable], params["seq_length"], params["target_seq_length"])

def objective(trial):

    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 10, 65, step=5)
    num_layers = trial.suggest_int("num_layers", 2, 12)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # epochs = trial.suggest_int("epochs", 50, 200, step=10)

    X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
    y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))#.unsqueeze(1)

    model = GRU_Model(params['input_size'], hidden_size, num_layers, params['output_size'], dropout)
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.5)
    # criterion = smape_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_gru, forecast_seq = train_model(model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        X_train=X_train,
                                        y_train=y_train,
                                        batch_size=batch_size,
                                        epochs=200)

    # forecast_seq_descaled = sequences_dict['scaler_y'].inverse_transform(forecast_seq.detach().cpu().numpy())
    scaler = sequences_dict['scaler']
    forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[-1] + scaler.mean_[-1]

    print(forecast_seq_descaled.flatten())
    print(train_val_dict['val_set'].iloc[:,-1].values)

    # loss = criterion(torch.from_numpy(forecast_seq_descaled.flatten()), torch.from_numpy(train_val_dict['val_set'].iloc[:,-1].values))
    loss = smape_loss(forecast_seq_descaled.flatten(), train_val_dict['val_set'].iloc[:,-1].values)
    print(criterion(torch.from_numpy(forecast_seq_descaled.flatten()), torch.from_numpy(train_val_dict['val_set'].iloc[:,-1].values)))

    return loss.item()

if os.path.exists(os.path.join(elec_dir, 'best_params', 'best_params_gru.pkl')):
    print('Loading best params...')
    with open(os.path.join(elec_dir, 'best_params', 'best_params_gru.pkl'), 'rb') as fin:
        study = pickle.load(fin)
        best_params = study.best_trial.params

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

else:
    print('Tuning hyperparameters...')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_value))
    study.optimize(objective, n_trials=30)
    best_params = study.best_trial.params

    with open(os.path.join(elec_dir, 'best_params', 'best_params_gru.pkl'), 'wb') as fout:
        pickle.dump(study, fout)

    print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

### Finished tuning hyperparameters

# creating data slices to generate forecasts for the next 8 days
index_cutoffs = [24*i for i in range(7, -1, -1)]
train_df_list = [train_df.iloc[:-idx] if idx != 0 else train_df for idx in index_cutoffs]
index_ceiling = [x.index.stop for x in train_df_list]
test_df_list = [train_df['price_de'].iloc[idx:idx+step_size] if idx!=index_ceiling[-1] else test_df['price_de'] for idx in index_ceiling]
y_hat_full = np.empty((0, 1))

### Trial Best Parameters ###
# best_params = {'learning_rate': 0.00031489116479568613, 'hidden_size': 50, 'num_layers': 9, 'batch_size': 32, 'dropout': 0.12323344486727979}
#############################

# for i, train_df_slice in enumerate(train_df_list):

# # for train_df_slice in [train_df_list[-1]]:

#     sequences_dict = create_sequences(train_df_slice[feature_variable], params["seq_length"], params["target_seq_length"])

#     all_forecast_seq_descaled = []

#     X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
#     y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))#.unsqueeze(1)

#     model = GRU_Model(params['input_size'], best_params['hidden_size'], best_params['num_layers'], params['output_size'], best_params['dropout'])
#     model.to(device)

#     # criterion = nn.MSELoss()
#     criterion = nn.SmoothL1Loss(beta=0.5)
#     # criterion = smape_loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

#     model_gru, forecast_seq = train_model(model,
#                                         criterion=criterion,
#                                         optimizer=optimizer,
#                                         X_train=X_train,
#                                         y_train=y_train,
#                                         # y_val=sequences_dict['scaler_y'].transform(test_df_list[i].to_numpy().reshape(-1, 1)).flatten(),
#                                         batch_size=best_params['batch_size'],
#                                         epochs=50)

#     # forecast_seq_descaled = sequences_dict['scaler'].inverse_transform(forecast_seq.cpu().numpy())
#     # forecast_seq_descaled = sequences_dict['scaler_y'].inverse_transform(forecast_seq.detach().cpu().numpy().reshape(-1, 1))
#     scaler = sequences_dict['scaler']
#     # forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) / scaler.scale_[-1] + scaler.min_[-1]
#     forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[-1] + scaler.mean_[-1]

#     # loss_smape = smape_loss(torch.from_numpy(forecast_seq_descaled[:,-1].reshape(-1, 1)), torch.from_numpy(test_df_list[i].to_numpy().reshape(-1, 1)))
#     loss_smape = smape_loss(torch.from_numpy(forecast_seq_descaled.flatten()), torch.from_numpy(test_df_list[i].to_numpy().flatten()))

#     print(forecast_seq_descaled.flatten())
#     print(test_df_list[i].to_numpy())

#     print(f'The SMAPE loss for {i}: {loss_smape.item()}')
#     # print(f'The Huber loss for {i}: {loss_huber.item()}')

#     y_hat_full = np.vstack((y_hat_full, forecast_seq_descaled.reshape(-1, 1)))

train_df_slice = train_df_list[0]

sequences_dict = create_sequences(train_df_slice[feature_variable], params["seq_length"], params["target_seq_length"])

X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))#.unsqueeze(1)

# best_params['hidden_size'] = 70
model = GRU_Model(params['input_size'], best_params['hidden_size'], best_params['num_layers'], params['output_size'], best_params['dropout'])
model.to(device)

criterion = nn.SmoothL1Loss(beta=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

model_gru, forecast_seq = train_model(model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    X_train=X_train,
                                    y_train=y_train,
                                    batch_size=best_params['batch_size'],
                                    epochs=700)

scaler = sequences_dict['scaler']
forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[-1] + scaler.mean_[-1]

print(forecast_seq_descaled.flatten())
print(test_df_list[0].to_numpy())

loss_smape = smape_loss(forecast_seq_descaled.flatten(), test_df_list[0].to_numpy().flatten())

print(f'The SMAPE loss for the testing phase: {loss_smape.item()}')

for i, train_df_slice in enumerate(train_df_list):

    sequences_dict = create_sequences(train_df_slice[feature_variable], params["seq_length"], params["target_seq_length"])

    X_train = torch.from_numpy(sequences_dict['X'].astype(np.float32))#.unsqueeze(-1)
    y_train = torch.from_numpy(sequences_dict['y'].astype(np.float32))#.unsqueeze(1)

    model = model_gru
    model.eval()
    last_sequence = X_train[-1:].to(device) # [1, 168, 16]
    forecast_seq = model(last_sequence)

    scaler = sequences_dict['scaler']
    forecast_seq_descaled = forecast_seq.detach().cpu().numpy().reshape(-1, 1) * scaler.scale_[-1] + scaler.mean_[-1]

    print(forecast_seq_descaled.flatten())
    print(test_df_list[i].to_numpy())

    loss_smape = smape_loss(forecast_seq_descaled.flatten(), test_df_list[i].to_numpy().flatten())

    y_hat_full = np.vstack((y_hat_full, forecast_seq_descaled.reshape(-1, 1)))

    print(f'The SMAPE loss for {i}: {loss_smape.item()}')


# save the forecasts
y_hat_df = pd.DataFrame({'y_hat_gru': y_hat_full.flatten()})
y_hat_df.to_csv(os.path.join(elec_dir, 'base_models_ts', 'y_hat_df_gru_test.csv'), index=False)

# asserting that the length of the forecast is correct
print(f'y_hat length: {len(y_hat_full)} vs correct length: {params["seq_length"] + params["target_seq_length"]}')

