# config.py

import os
import torch

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory of the project
DATA_DIR = os.path.join(BASE_DIR, "data")                               # Main data directory
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")                            # Raw MRI images from TCIA
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")                # Processed images directory
TRAIN_DIR = os.path.join(DATA_DIR, "train")                             # Directory for training data
VAL_DIR = os.path.join(DATA_DIR, "val")                                 # Directory for validation data
TEST_DIR = os.path.join(DATA_DIR, "test")                               # Directory for test data
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model Parameters
MODEL_NAME = "resnet50"            # Model type (e.g., "resnet50", "efficientnet_b0")
NUM_CLASSES = 2                    # Number of output classes (e.g., 2 for binary classification)
PRETRAINED = True                  # Use a pretrained model

# Training Hyperparameters
BATCH_SIZE = 32                    # Batch size
LEARNING_RATE = 0.001              # Learning rate
NUM_EPOCHS = 10                    # Number of training epochs
MOMENTUM = 0.9                     # Momentum for optimizer

# Image Processing
IMG_SIZE = (256, 256)              # Size to resize images for input to model
NORMALIZE_MEAN = [0.5]             # Mean for normalization (modify if using 3-channel data)
NORMALIZE_STD = [0.5]              # Standard deviation for normalization (modify if using 3-channel data)
AUGMENTATION_PROB = 0.5            # Probability for data augmentation (e.g., horizontal flip)

# Device Configuration
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"  # Set to "cuda" if a GPU is available

# Checkpoint and Model Saving
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")  # Directory for model checkpoints
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pth")  # Path to save the final model

# Logging and Results
RESULTS_DIR = os.path.join(BASE_DIR, "results")                  # Directory to save results (e.g., plots)
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.txt")         # File to save training logs

# Create directories if they don’t exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# config.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # Binary classification: tumor vs. no tumor

