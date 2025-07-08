import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import ParameterSampler
import pandas as pd
from ClassesML.AudioTrainer import AudioTrainer
from ClassesML.AudioModel import AudioModel
from ClassesData.AnimalSoundDataset import AnimalSoundDataset

import json


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
df = pd.read_csv('data/Animal_Sound_processed.csv')

dataset_train = AnimalSoundDataset(df, split='train', split_ratio=0.8, seed=42)
dataset_val = AnimalSoundDataset(df, split='val', split_ratio=0.8, seed=42)

# Hyperparameter tunning
input_dim = dataset_train[0][0].shape[0]
n_classes = len(dataset_train.classes)
 

hyperparameters = dict(
    input_dim=input_dim,
    output_dim=n_classes,
    hidden_layers_size=5000,
    activation='relu',
    kernel_size_conv=[(57, 6), (1, 3)],
    kernel_size_pool=[(4, 3), (1, 3)],
    stride_conv=[(1, 1), (1, 1)],
    stride_pool=[(1, 3), (1, 3)],
    filters=[80, 80],
    batch_normalization=False,
    dropout_rate=0.5,

    # trainer hyperparameters
    learning_rate=0.002,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,

    # questionable hyperparameters
    #batch_size=batch_size,
    max_epoch=100,

    #Early stopping and sceduler
    patience_lr=5,
    early_stopping=True
)

hyperparameter_choices = {}
for k in hyperparameters.keys():
    hyperparameter_choices[k] = [hyperparameters[k]]

hyperparameter_choices['learning_rate'] = [0.001, 0.002, 0.005]
hyperparameter_choices['batch_size'] = [32, 64, 128, 256]
hyperparameter_choices['max_epoch'] = [2]
hyperparameter_choices['hidden_layers_size']=[1000, 5000]
hyperparameter_choices['patience_lr'] = [5, 10, 15]
hyperparameter_choices['momentum'] = [0.9, 0.95, 0.85]
hyperparameter_choices['weight_decay'] = [0.001, 0.002]


hyperparameter_try = list(ParameterSampler(hyperparameter_choices, n_iter=20))

metric_list = []

for hyperparam in hyperparameter_try:

    model = AudioModel(hyperparam).to(device)
    
    train_loader = DataLoader(dataset_train, batch_size=hyperparam['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=hyperparam['batch_size'], shuffle=False)

    trainer = AudioTrainer(model, train_loader, val_loader, hyperparam, device=device)

    train_accuracy_list, valid_accuracy_list = trainer.train()
    metric_list.append(valid_accuracy_list[-1])
    hyperparam['val_accuracy'] = valid_accuracy_list[-1]
    hyperparam['train_accuracy'] = train_accuracy_list[-1]

idx = np.argsort(metric_list)
hyperparameter_sorted = np.array(hyperparameter_try)[idx].tolist()
df = pd.DataFrame.from_dict(hyperparameter_sorted)

best_hyperparams = hyperparameter_sorted[-1]

# Save to JSON
with open("configs/audio_model_processed_data_best.json", "w") as f:
    json.dump(best_hyperparams, f, indent=4)

print("Best hyperparameters saved to best_hyperparameters.json")