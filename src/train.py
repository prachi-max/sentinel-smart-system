import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess_data
from model import SentinelLSTM, SentinelGRU
import numpy as np
import os

# 1. THE WINDOWING FUNCTION (Crucial for Real-Time logic)
def create_sequences(data, seq_length, sensors):
    x, y = [], []
    for engine_id in data['engine_id'].unique():
        engine_data = data[data['engine_id'] == engine_id]
        sensor_values = engine_data[sensors].values
        rul_values = engine_data['RUL'].values
        
        for i in range(len(engine_data) - seq_length):
            x.append(sensor_values[i:i+seq_length])
            y.append(rul_values[i+seq_length])
            
    return np.array(x), np.array(y)

# 🔥 Hyperparameters (ADD THIS AT TOP)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 50


def train_model():
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'train_FD001.txt')
    
    # Load Data
    df, sensors = load_and_preprocess_data(file_path)
    SEQUENCE_LENGTH = 50
    
    print("🔄 Creating sequences...")
    X_raw, y_raw = create_sequences(df, SEQUENCE_LENGTH, sensors)
    
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 🔥 MODEL SELECTION (LSTM / GRU)
    use_gru = False  # change True to test GRU

    if use_gru:
        model = SentinelGRU(input_size=len(sensors), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
        print("🚀 Using GRU Model")
    else:
        model = SentinelLSTM(input_size=len(sensors), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
        print("🚀 Using LSTM Model")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 🔥 TRAINING LOOP
    print(" Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📅 Epoch {epoch+1}/{EPOCHS} | 📉 Loss: {total_loss/len(loader):.4f}")

        # 🔥 SAVE MODEL AT LAST EPOCH
        if epoch == EPOCHS - 1:
            model_path = os.path.join(base_dir, 'sentinel_model.pth')
            torch.save(model.state_dict(), model_path)
            print("✅ Model saved successfully!")
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'train_FD001.txt')
    
    # 2. Load and Sequence Data
    df, sensors = load_and_preprocess_data(file_path)
    SEQUENCE_LENGTH = 50 # The AI looks at 50 cycles at a time
    
    print("🔄 Creating sequences for the LSTM...")
    X_raw, y_raw = create_sequences(df, SEQUENCE_LENGTH, sensors)
    
    # Convert to PyTorch Tensors
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Initialize the Brain
    model = SentinelLSTM(input_size=len(sensors), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epochs = EPOCHS
    criterion = nn.MSELoss() # Measures how far the prediction is from the truth


    # 4. The Training Loop (The "Study" Session)
    print(" Starting Training (The AI is learning)...")
    epochs = 150 # Increase this to 50+ for better accuracy later
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"📅 Epoch {epoch+1}/{epochs} | 📉 Loss: {total_loss/len(loader):.4f}")

    # 5. Save the trained brain
    model_path = os.path.join(base_dir, 'sentinel_model.pth')
    if epoch == epochs - 1:
        model_path = os.path.join(base_dir, 'sentinel_model.pth')
        torch.save(model.state_dict(), model_path)
        print(" Best model saved!")
        print(f"Training Complete! Model saved as: {model_path}")

if __name__ == "__main__":
    train_model()