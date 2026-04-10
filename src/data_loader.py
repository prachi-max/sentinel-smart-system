import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # 1. Define column names for the NASA dataset
    id_col = ['engine_id']
    time_col = ['cycle']
    setting_cols = ['setting1', 'setting2', 'setting3']
    sensor_cols = [f's{i}' for i in range(1, 22)]
    columns = id_col + time_col + setting_cols + sensor_cols
    
    # 2. Read the raw text file
    df = pd.read_csv(file_path, sep=' ', header=None, names=columns, index_col=False)
    
    # Drop empty columns if any
    df = df.dropna(axis=1, how='all')
    
    print(f" Successfully loaded data with shape: {df.shape}")
    
    # 3. Calculate Remaining Useful Life (RUL) - This is what our AI will predict!
    # We find the max cycle for each engine and subtract the current cycle
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    # 4. Normalize the sensor data (Scale between 0 and 1)
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    
    return df, sensor_cols

# This allows us to test the file directly
if __name__ == "__main__":
    # Test loading the file (Make sure the path matches where you put your data)
    try:
        data, sensors = load_and_preprocess_data('data/train_FD001.txt')
        print(data.head())
    except Exception as e:
        print(f" Error: Could not find the file. Please check your path. Details: {e}")