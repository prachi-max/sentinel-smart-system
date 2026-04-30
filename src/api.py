
from fastapi import FastAPI
import torch
import os
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from src.data_loader import  load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler
from src.model import SentinelLSTM
app = FastAPI()


# ==============================
# 🔥 CORS (Allow Angular)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🔥 LOAD MODEL
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "sentinel_model.pth")

model = SentinelLSTM(input_size=21, hidden_size=64, num_layers=2)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", e)

# ==============================
# 🔥 LOAD DATA
# ==============================
DATA_PATH = os.path.join(BASE_DIR, "Data", "train_FD001.txt")

df, sensors = load_and_preprocess_data(DATA_PATH)

# Normalize sensor values
scaler = MinMaxScaler()
df[sensors] = scaler.fit_transform(df[sensors])

SEQUENCE_LENGTH = 50

# ==============================
# 🔥 REAL-TIME ENGINE TRACKING
# ==============================
current_index_map = {}

def get_engine_sequence(engine_id: int):
    global current_index_map

    engine_data = df[df['engine_id'] == engine_id]

    if len(engine_data) < SEQUENCE_LENGTH:
        return None, None

    # Initialize index for each engine
    import random

    if engine_id not in current_index_map:
      max_start = len(engine_data) - SEQUENCE_LENGTH
      current_index_map[engine_id] = random.randint(0, max_start)

    current_index = current_index_map[engine_id]

    # Reset when reaching end
    if current_index + SEQUENCE_LENGTH >= len(engine_data):
        current_index = 0

    # Get sequence window
    sequence = engine_data[sensors].values[
        current_index : current_index + SEQUENCE_LENGTH
    ]

    # Get corresponding cycle
    cycle = int(
        engine_data['cycle'].values[current_index + SEQUENCE_LENGTH - 1]
    )

    # Move forward (simulate real-time)
    current_index_map[engine_id] = current_index + 1

    # Convert to tensor
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = sequence.unsqueeze(0)

    return sequence, cycle

# ==============================
# 🚀 API ENDPOINT
# ==============================

@app.get("https://sentinel-smart-system-4.onrender.com/api/v1/health-check/{engine_id}")
async def get_engine_status(engine_id: int):
    try:
        input_data, current_cycle = get_engine_sequence(engine_id)

        if input_data is None:
            return {
                "engine_id": engine_id,
                "cycle": 0,
                "predicted_rul": 0,
                "status": "OFFLINE",
                "timestamp": str(pd.Timestamp.now())
            }

        with torch.no_grad():
            prediction = model(input_data).item()
            prediction = max(0, min(prediction, 200))

        if prediction < 20:
            status = "CRITICAL"
        elif prediction < 80:
            status = "WARNING"
        else:
            status = "OPERATIONAL"

        return {
            "engine_id": engine_id,
            "cycle": current_cycle,
            "predicted_rul": round(prediction, 2),
            "status": status,
            "timestamp": str(pd.Timestamp.now())
        }

    except Exception as e:
        print("API ERROR:", e)
        return {
            "engine_id": engine_id,
            "cycle": 0,
            "predicted_rul": 0,
            "status": "ERROR",
            "timestamp": str(pd.Timestamp.now())
        }
    