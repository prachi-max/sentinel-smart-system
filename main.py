import os
import sys
import uvicorn
from fastapi import FastAPI

# Ensure the 'src' directory is in the system path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from train import train_model
    # Note: We won't call simulate_real_time_inference directly here 
    # because the API will handle the data flow to Angular.
except ImportError as e:
    print(f" Import Error: {e}. Ensure your 'src' folder has an __init__.py file.")

def start_api():
    print("\n" + "="*50)
    print(" STARTING SENTINEL-AI API SERVER")
    print("   Listening on: http://127.0.0.1:8000")
    print("="*50)
    
    # This connects to the 'app' variable inside your 'src/api.py'
    # 'reload=True' is great for development (auto-restarts on code changes)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

def main():
    while True:
        print("\n" + "="*50)
        print("    SENTINEL-AI: PREDICTIVE MAINTENANCE SYSTEM    ")
        print("==================================================")
        print("1. [TRAIN]    - Start Deep Learning Training (LSTM)")
        print("2. [SIMULATE] - Start API & Dashboard Stream")
        print("3. [EXIT]     - Close the program")
        print("==================================================")
        
        choice = input("\nSelect an option (1/2/3): ").strip()
        
        if choice == '1':
            print("\n Initializing Training Pipeline...")
            train_model()
            
        elif choice == '2':
            # Option 2 now launches the "Bridge" to your Angular Frontend
            try:
                start_api()
            except Exception as e:
                print(f" Failed to start API: {e}")
                 
        elif choice == '3':
            print("\nShutting down. Good luck with your exams, Prachi! 👋")
            sys.exit()
            
        else:
            print("\n Invalid input. Please select 1, 2, or 3.")
        
if __name__ == "__main__":
    main()