# main.py

from src import (build_model, evaluate_model, load_data, predict, preprocess_images, split_data, train_model)
from src.config import BATCH_SIZE, DATA_DIR, DEVICE, FINAL_MODEL_PATH

# Step 1: Load and Preprocess Data
print("Loading and preprocessing data...")
data = load_data(DATA_DIR)            # Load the data from the specified directory
processed_data = preprocess_images(data)   # Preprocess images as per the project's requirements
train_data, val_data = split_data(processed_data)   # Split the data into training and validation sets

# Step 2: Initialize DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}

# Step 3: Build Model
print("Building model...")
model = build_model()     # Build the model architecture (e.g., ResNet50)
model = model.to(DEVICE)  # Move model to GPU if available

# Step 4: Train Model
print("Training model...")
trained_model = train_model(model, dataloaders, DEVICE)   # Train the model on the data

# Step 5: Evaluate Model
print("Evaluating model...")
evaluate_model(trained_model, dataloaders['val'], DEVICE)   # Evaluate the model on the validation set

# Step 6: Save the Model
torch.save(trained_model.state_dict(), FINAL_MODEL_PATH)    # Save the trained model to the specified path
print(f"Model saved to {FINAL_MODEL_PATH}")

# Step 7: Optional - Make Predictions on Test Data
test_data = load_data(os.path.join(DATA_DIR, "test"))    # Load test data if available
predictions = predict(trained_model, test_data, DEVICE)   # Use the model to make predictions on new data
print("Predictions:", predictions)
