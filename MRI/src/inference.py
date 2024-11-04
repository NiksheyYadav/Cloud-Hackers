import torch
from torchvision import transforms
from PIL import Image
from src.model_architecture import build_model
from src.config import DEVICE, IMG_SIZE, FINAL_MODEL_PATH

def preprocess_image(image_path):
    """
    Preprocesses an image for model input.
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model_path=FINAL_MODEL_PATH):
    """
    Loads a model and makes a prediction on an image.
    """
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    image_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.softmax(output, dim=1)[0, predicted].item()

    classes = ["No Tumor", "Tumor"]
    result = classes[predicted.item()]
    print(f"Prediction: {result} (Probability: {probability:.2f})")
    return result, probability
