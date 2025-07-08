import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random
import json

from ClassesData.AnimalSoundDataset import AnimalSoundDataset
from ClassesML.AudioModel import AudioModel
from ClassesML.AudioTrainer import AudioTrainer

from Utilities.Utilities import Utilities

import pandas as pd
from sklearn.model_selection import KFold

# load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('data/Animal_Sound_processed.csv')

# load best hyperparameters
with open("configs/audio_model_processed_data_best.json", "r") as f:
    hyperparameters = json.load(f)


# Prepare 5-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_list = []
val_loader_list = []
train_accuracy_per_epoch_list = []
val_accuracy_per_epoch_list = []

# track the best model
best_accuracy = 0
best_model_path = ""
results = []  # to store fold results


for fold, (train_val_idx, val_idx) in enumerate(kf.split(df)):
    # split the dataset
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[val_idx].reset_index(drop=True)

    print(f"Fold {fold + 1}")
    print(f"Train/Val samples: {len(df_train_val)}")
    print(f"Test samples: {len(df_test)}")

    # Create datasets for this fold
    dataset_train = AnimalSoundDataset(df_train_val, split='train', split_ratio=0.75, seed=42)
    dataset_val = AnimalSoundDataset(df_train_val, split='val', split_ratio=0.75, seed=42)
    train_loader = DataLoader(dataset_train, batch_size=hyperparameters['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=hyperparameters['batch_size'], shuffle=False)
    val_loader_list.append(val_loader)

    # Initialize model and trainer
    model = AudioModel(hyperparameters).to(device)
    trainer_final = AudioTrainer(model, train_loader, val_loader, hyperparameters, device=device)
    train_acc_list, val_acc_list = trainer_final.train()
    train_accuracy_per_epoch_list.append(train_acc_list)
    val_accuracy_per_epoch_list.append(val_acc_list)

    # Save the model for this fold
    model_save_path = f'models/model_fold_{fold + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}\n")

    # Evaluate on test set
    total_predictions = 0
    correct_predictions = 0
    for path, label in zip(df_test['path'], df_test['name']):

        test_segments = Utilities.extract_segments_with_deltas(path, variant='short')
        predicted_sound = Utilities.predict_clip(model, torch.tensor(test_segments, dtype=torch.float32).to(device), device, method='prob')
        
        predicted_label = dataset_train.classes[predicted_sound]
        
        if predicted_label == label:
            correct_predictions += 1
        total_predictions += 1
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    accuracy_list.append(accuracy)
    print(f"Fold {fold + 1} - Test Accuracy: {accuracy:.4f}\n")

        # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_save_path

    # Save results for CSV
    results.append({
        "fold": fold + 1,
        "test_accuracy": accuracy,
        "avg_train_accuracy": sum(train_acc_list) / len(train_acc_list),
        "avg_val_accuracy": sum(val_acc_list) / len(val_acc_list),
        "model_path": model_save_path
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results/kfold_results.csv", index=False)
print("Saved k-fold results to results/kfold_results.csv")

# Save best model info
print(f"Best model: {best_model_path} with accuracy: {best_accuracy:.2f}%")

best_model_target = "models/best_model.pth"
if os.path.exists(best_model_path):
    import shutil
    shutil.copy(best_model_path, best_model_target)
    print(f"Best model copied to {best_model_target}")

print("Done")

   