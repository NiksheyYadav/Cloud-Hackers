import torch
from torch.utils.data import DataLoader
from src.utils import plot_confusion_matrix, calculate_metrics
from src.config import TEST_DIR, BATCH_SIZE, DEVICE
from src.data_preprocessing import load_data
from src.model_architecture import build_model

def evaluate_model(model_path):
    """
    Evaluates the trained model on the test dataset.
    """
    # Load the model
    model = build_model(pretrained=False)  # pretrained=False as we're loading trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval().to(DEVICE)

    # Load test dataset
    test_dataset = load_data(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy, roc_auc = calculate_metrics(y_true, y_pred, y_probs)
    print(f"Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=["No Tumor", "Tumor"])
