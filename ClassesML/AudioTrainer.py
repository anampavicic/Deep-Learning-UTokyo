import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utilities.Utilities import Utilities

class AudioTrainer:
    def __init__(self, model, train_dataset, val_dataset, hyperparameters, device='cpu'):
        self.hyperparameters = hyperparameters
        self.model = model
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=False)
        self.max_epoch = self.hyperparameters['max_epoch']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparameters['learning_rate'])
        self.model.to(device)

    def run(self):
        train_accuracy_dict = {}
        valid_accuracy_dict = {}
        for epoch in range(self.max_epoch):
            self.model.train()
            total_loss = 0.0 #Running loss
            total_accuracy = 0.0
            n_batch = self.hyperparameters['batch_size']

            for x_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epoch}"):
                x, y = x_batch.to(self.device), y_batch.to(self.device)
                # Forward
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # Compute accuracy
                batch_accuracy = Utilities.compute_accuracy(y, y_hat)
                total_accuracy += batch_accuracy

            train_loss = total_loss / n_batch
            train_accuracy = total_accuracy / n_batch

            val_acc, val_loss = self.evaluate()

            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss/len(self.train_loader):.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
            
            train_accuracy_dict[epoch] = train_accuracy
            valid_accuracy_dict[epoch] = val_acc
        train_accuracy_list = [train_accuracy_dict[e] for e in train_accuracy_dict.keys()]
        valid_accuracy_list = [valid_accuracy_dict[e] for e in valid_accuracy_dict.keys()]
        return train_accuracy_list, valid_accuracy_list

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batch = self.hyperparameters['batch_size']

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x, y = x_batch.to(self.device), y_batch.to(self.device)
                # Forward
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                total_loss += loss.item()
                # Compute accuracy
                batch_accuracy = Utilities.compute_accuracy(y, y_hat)
                total_accuracy += batch_accuracy
            valid_loss = total_loss / n_batch
            valid_accuracy = total_accuracy / n_batch

        return valid_accuracy, valid_loss
