import torch
import numpy as np
import pandas as pd
import os
import time
from data_loader import load_and_preprocess_data
from model import SentinelLSTM

def simulate_real_time_inference():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file_path = os.path.join(base_dir, 'data', 'test_FD001.txt')
    model_path = os.path.join(base_dir, 'sentinel_model.pth')
    
    SEQUENCE_LENGTH = 50
    
    # 1. Load Data
    df, sensors = load_and_preprocess_data(test_file_path)
    
    # 2. Find an engine with enough data (>= 50 cycles)
    counts = df.groupby('engine_id').size()
    valid_engines = counts[counts >= SEQUENCE_LENGTH].index.tolist()
    
    if not valid_engines:
        print(f"❌ No engines found with at least {SEQUENCE_LENGTH} cycles.")
        return

    engine_id = valid_engines[0] # Pick the first valid engine
    print(f"✅ Selected Engine #{engine_id} for simulation (Total cycles: {counts[engine_id]})")

    # 3. Load Model
    model = SentinelLSTM(input_size=len(sensors), hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\n📡 Starting Real-Time Sensor Stream Simulation...")
    print("-" * 60)

    engine_data = df[df['engine_id'] == engine_id][sensors].values
    
    for i in range(SEQUENCE_LENGTH, len(engine_data)):
        input_window = engine_data[i-SEQUENCE_LENGTH:i]
        input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        # Simple Logic for Anomaly/Status
        if prediction < 30:
            status = "🔴 CRITICAL"
        elif prediction < 70:
            status = "🟡 WARNING"
        else:
            status = "🟢 NORMAL"
            
        print(f"Cycle: {i} | Predicted RUL: {prediction:>6.2f} | Status: {status}")
        time.sleep(0.3) # Slightly faster for testing

if __name__ == "__main__":
    simulate_real_time_inference()