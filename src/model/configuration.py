# Install segmentation-models-pytorch
!pip install -q segmentation-models-pytorch

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)



# ============================================================
# Configuration
# ============================================================
VERSION = 'vR.P.7'
CHANGE = 'Extended training (50 epochs, patience 10) — P.3 was still improving at epoch 25'
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3  # ELA is 3-channel RGB
NUM_CLASSES = 1
LEARNING_RATE = 1e-3  # Single LR (decoder + BN only)
ELA_QUALITY = 90  # NEW: ELA JPEG recompression quality
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4          # Parallel data loading (was 2 in P.3)
PREFETCH_FACTOR = 2      # Explicit prefetching
CHECKPOINT_PATH = f'{VERSION}_checkpoint.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')