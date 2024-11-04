import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.config import LOG_FILE, RESULTS_DIR


def save_checkpoint(model, path):
    """
    Saves a model checkpoint.
    """
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved at {path}")

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap('Blues'))
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")

def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculates accuracy, ROC AUC, and generates an ROC curve.
    """
    accuracy = (y_true == y_pred).sum() / len(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()
    print("ROC curve saved.")

    return accuracy, roc_auc

def log_training(epoch, loss):
    """
    Logs training loss per epoch.
    """
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss:.4f}\n")
    print(f"Logged epoch {epoch + 1} loss.")
