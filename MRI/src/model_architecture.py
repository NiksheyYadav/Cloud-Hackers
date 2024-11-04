# model_architecture.py

import torch
import torch.nn as nn
from torchvision import models

from src.config import DEVICE, NUM_CLASSES


def build_model(pretrained=True):
    """
    Builds a ResNet-50 model with a modified final layer for binary classification.
    
    Args:
        pretrained (bool): If True, loads a ResNet model pretrained on ImageNet.
    
    Returns:
        model (nn.Module): The modified ResNet model.
    """
    # Load a pretrained ResNet-50 model
    model = models.resnet50(pretrained=pretrained)
    
    # Get the number of features in the last layer
    num_features = model.fc.in_features
    
    # Replace the final fully connected layer with a custom one (for NUM_CLASSES outputs)
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    # Move model to the specified device (CPU or GPU)
    model = model.to(DEVICE)
    
    return model

def get_optimizer(model, learning_rate=0.001):
    """
    Sets up the optimizer for model training.
    
    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        optimizer: Configured optimizer (SGD).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return optimizer

def get_loss_function():
    """
    Returns the loss function for binary classification.
    
    Returns:
        criterion: CrossEntropyLoss.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion
