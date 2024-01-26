import pandas as pd
from config import ROOT_DIR

df_h1 = pd.read_csv('./data/ETT/ETTh1.csv')

# num_train = 8760
# num_valid = 4380

dataset_configs = {
    "ETTh1" : {
        "num_train" : 8720,
        "num_valid" : 4340
    }
}

train_indices = list(range(dataset_configs['ETTh1']['num_train']))
valid_indices = list(range(dataset_configs['ETTh1']['num_train'], dataset_configs['ETTh1']['num_train'] + dataset_configs['ETTh1']['num_valid']))
test_indices = list(range(dataset_configs['ETTh1']['num_train'] + dataset_configs['ETTh1']['num_valid'], len(df_h1)))

df_h1['split'] = 'test'
df_h1.loc[train_indices, 'split'] = 'train'
df_h1.loc[valid_indices, 'split'] = 'valid'

df_h1.to_csv('./data/df_h1.csv', index=False)