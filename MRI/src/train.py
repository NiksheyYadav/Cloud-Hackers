import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_preprocessing import Dataset  # Assuming you have a dataset class
from src.model_architecture import build_model # Your model class
from src.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_DIR, FINAL_MODEL_PATH

def train_model():
    # Set device
    device = DEVICE

        # Define transformations for the training set
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
        ])

        # Load datasets
    train_dataset = Dataset(root='data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Initialize model, loss function, and optimizer
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()  # Adjust based on your task
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0
            
        for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], 
                        Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

            # Save the model checkpoint
        if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch + 1}.pth'))

        # Save the final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print('Training complete! Model saved.')

    if __name__ == '__main__':
        train_model()