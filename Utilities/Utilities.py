import os
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torchvision.utils import make_grid

import warnings
import librosa

import torch.nn.functional as F


class Utilities:

    @staticmethod
    def compute_accuracy(y, y_hat):

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = torch.tensor(y_hat)

        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0) * 100

        return accuracy

    @staticmethod
    def get_activation(activation_str: str or None):

        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == "linear":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")

    
    @staticmethod
    def extract_segments_with_deltas(file_path, variant='short', silence_threshold=-80):
        """
        Extracts 2-channel (log-mel + delta) spectrogram segments from an audio file.

        Parameters:
        - file_path (str): Path to audio file.
        - variant (str): 'short' (41 frames, 50% overlap) or 'long' (101 frames, 90% overlap).
        - silence_threshold (float): dB threshold for discarding low-energy segments.

        Returns:
        - np.ndarray: Array of shape (n_segments, 2, 60, frames_per_segment)
        """
        # Config
        sr = 22050
        n_fft = 1024
        hop_length = 512
        n_mels = 60

        if variant == 'short':
            frames_per_segment = 41
            overlap = 0.5
        elif variant == 'long':
            frames_per_segment = 101
            overlap = 0.9
        else:
            raise ValueError("variant must be 'short' or 'long'")

        try:
            # Load audio in mono
            y, _ = librosa.load(file_path, sr=sr, mono=True)

            # Skip empty or very short files
            if len(y) < n_fft:
                warnings.warn(f"File too short to process: {file_path}")
                return np.empty((0, 2, 60, frames_per_segment))

            # Compute log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize safely
            mean = np.mean(log_mel_spec)
            std = np.std(log_mel_spec)
            if std == 0:
                warnings.warn(f"Zero std encountered in file: {file_path}")
                return np.empty((0, 2, 60, frames_per_segment))

            log_mel_spec = (log_mel_spec - mean) / std

            # Compute deltas
            delta_spec = librosa.feature.delta(log_mel_spec)

            # Segmenting
            step = int(frames_per_segment * (1 - overlap))
            segments = []

            for start in range(0, log_mel_spec.shape[1] - frames_per_segment + 1, step):
                seg = log_mel_spec[:, start:start + frames_per_segment]
                delta = delta_spec[:, start:start + frames_per_segment]

                # Skip silent segments
                if np.mean(seg) < silence_threshold:
                    continue

                stacked = np.stack([seg, delta], axis=0)
                segments.append(stacked)

            return np.stack(segments) if segments else np.empty((0, 2, 60, frames_per_segment))

        except Exception as e:
            warnings.warn(f"Failed to process {file_path}: {e}")
            return np.empty((0, 2, 60, frames_per_segment))
    
    @staticmethod
    def predict_clip(model, segment_tensors, device, method='prob'):
        """
        Generate prediction for a whole clip given its segments.
        
        segment_tensors: tensor of shape (n_segments, channels, mel_bands, frames)
        method: 'majority' or 'prob'
        """
        model.eval()
        segment_tensors = segment_tensors.to(device)
        with torch.no_grad():
            outputs = model(segment_tensors)  # (n_segments, num_classes)
            probs = F.softmax(outputs, dim=1)

        if method == 'majority':
            preds = torch.argmax(probs, dim=1)
            counts = torch.bincount(preds)
            clip_pred = torch.argmax(counts).item()
        elif method == 'prob':
            avg_probs = probs.mean(dim=0)
            clip_pred = torch.argmax(avg_probs).item()
        else:
            raise ValueError("method must be 'majority' or 'prob'")

        return clip_pred

    @staticmethod
    def plot_confusion_matrix_animals(y_true, y_pred, label_to_index, fig_path = "results/confusion_matrix_modified_best.png"):
        # y_pred is already class indices, so compute accuracy directly
        accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        index_to_label = {v: k.capitalize() for k, v in label_to_index.items()}
        labels = [index_to_label[i] for i in sorted(index_to_label)]
        
        plt.figure()
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix - Accuracy: {:.2f}%".format(accuracy))
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved the confusion matrix to f{fig_path}")
    
    @staticmethod
    def plot_graphs(max_train_acc_list, max_val_acc_list, fig_path = "results/graphs_modified.png"):
        plt.figure()
        plt.plot(max_train_acc_list, 'b', label='Train accuracy')
        plt.plot(max_val_acc_list, 'r', label='Valid accuracy')
        plt.title('Train and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')  
        plt.legend()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved the graphs to {fig_path}")
        return 


