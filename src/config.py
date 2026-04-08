"""
Central configuration for the live-emotion-recognition project.
"""

import os
import torch
import numpy as np


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Definition of frequently used values and paths
CLASS_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "trained_model_rafdb.pth")
DATASET_DIR = os.path.join(_PROJECT_ROOT, "RafDB-Dataset")
IMAGE_SIZE = 64


# Definition of Gabor-Filter values
GABOR_KSIZE = 31
GABOR_SIGMA = 4.0
GABOR_THETA = np.pi / 4
GABOR_LAMBDA = 9.0
GABOR_GAMMA = 2.0
GABOR_PSI = np.pi / 2
