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
    def plot_confusion_matrix_fashion(y, y_hat):

        accuracy = Utilities.compute_accuracy(y, y_hat)

        y_hat = np.argmax(y_hat, 1)

        cm = confusion_matrix(y, y_hat)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        plt.figure()
        plt.subplot(1, 1, 1)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[label_map[i] for i in range(10)],
                    yticklabels=[label_map[i] for i in range(10)])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix - Accuracy: " + str(accuracy))
        plt.tight_layout()

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
    def images_as_canvas(images, title: str = ""):

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def images_2_as_canvas(images, images2, title: str = ""):

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        canvas2 = make_grid(images2.cpu(), padding=10, nrow=10, normalize=True)
        canvas2 = canvas2.permute(1, 2, 0).numpy() * 255
        canvas2 = canvas2.astype("uint8")

        canvas = np.concatenate((canvas, canvas2), axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def plot_latent_space(z_fit, y_fit):

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1)

        cmap = plt.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, 10))
        colors = dict(zip(label_map.keys(), colors))

        for y in label_map.keys():
            index = np.where(y_fit == y)
            ax.scatter(z_fit[index, 0], z_fit[index, 1], color=colors[y],
                       marker='o', s=30, alpha=0.5,
                       label=label_map[y])

        ax.legend()
        plt.show()
    
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
