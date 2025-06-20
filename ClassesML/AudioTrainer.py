import torch
import torch.nn as nn
from tqdm import tqdm
from Utilities.Utilities import Utilities
import torch.optim as optim
from ClassesML.EarlyStopper import EarlyStopper
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, hyperparameters, device='cpu'):
        # Initialize hyperparameters and model
        self.hyperparameters = hyperparameters
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.max_epoch = self.hyperparameters['max_epoch']

        # Set up loss function and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameters['learning_rate'],
            momentum=hyperparameters['momentum'],  # Default momentum
            weight_decay=hyperparameters['weight_decay'],  # Default weight decay
            nesterov=hyperparameters['nesterov']  # Nesterov momentum
        )


        # Sceduler and early stopper
        if 'patience_lr' in hyperparameters:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max',
                                               patience=hyperparameters['patience_lr'],
                                               factor=0.1)
        else:
            self.scheduler = None

        if 'early_stopping' in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None


        self.model.to(device)

    def train(self):
        train_accuracy_dict = {}
        valid_accuracy_dict = {}
        for epoch in range(self.max_epoch):
            # Set model to training mode
            self.model.train()
            train_loss = 0.0 #Running loss
            
            batch_accuracy = 0.0  # Reset batch accuracy for each epoch

            for x_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epoch}"):
                x, y = x_batch.to(self.device), y_batch.to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                
                # Backward
                loss.backward()
                self.optimizer.step()

                # TO-DO: Decide how to compute loss
                train_loss += loss.item()

                # Compute accuracy
                batch_accuracy += Utilities.compute_accuracy(y, y_hat)
                
            train_loss = train_loss / len(self.train_loader)
            train_accuracy = batch_accuracy / len(self.train_loader) 

            val_accuracy, val_loss = self.evaluate()

            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")
            
            # Learning rate scheduler
            if self.scheduler:
                validation_metric = val_accuracy
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(validation_metric)
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f'Learning rate changed from {old_lr} to {new_lr} at epoch {epoch}')
            
            # Early stopping scheduler
            if self.early_stopper:
                validation_metric = val_accuracy
                keep_training = self.early_stopper.set(model=self.model, epoch = epoch, metric_epoch = validation_metric)
                if not keep_training:
                    break

            train_accuracy_dict[epoch] = train_accuracy
            valid_accuracy_dict[epoch] = val_accuracy

            

            
        train_accuracy_list = [train_accuracy_dict[e] for e in train_accuracy_dict.keys()]
        valid_accuracy_list = [valid_accuracy_dict[e] for e in valid_accuracy_dict.keys()]
        return train_accuracy_list, valid_accuracy_list
            
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

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

            valid_loss = total_loss / len(self.val_loader)
            valid_accuracy = total_accuracy / len(self.val_loader)

        return valid_accuracy, valid_loss
