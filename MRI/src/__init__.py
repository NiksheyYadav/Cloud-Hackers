# src/__init__.py

# Importing all necessary modules for easy access
from .config import DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS  # Adjust the variable names to match config.py

from .data_preprocessing import load_data, preprocess_images, split_data  # Adjust function names as needed
from .model_architecture import build_model                          # Ensure build_model function is defined in model_architecture.py
from .train import train_model
from .evaluate import evaluate_model
from .inference import predict                                      # Define predict function in inference.py
from .utils import plot_confusion_matrix, calculate_metrics         # Adjust function names as needed

__all__ = [
    "DATA_DIR",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "load_data",
    "preprocess_images",
    "split_data",
    "build_model",
    "train_model",
    "evaluate_model",
    "predict",
    "plot_confusion_matrix",
    "calculate_metrics",
]