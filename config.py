"""
Zentrale Konfiguration für das Emotion Recognition Projekt.
"""

import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


CLASS_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "trained_model_rafdb.pth"
DATASET_DIR = "RafDB-Dataset"
IMAGE_SIZE = 64

import numpy as np

# Gabor-Filter
GABOR_KSIZE = 31
GABOR_SIGMA = 4.0
GABOR_THETA = np.pi / 4
GABOR_LAMBDA = 9.0
GABOR_GAMMA = 2.0
GABOR_PSI = np.pi / 2
