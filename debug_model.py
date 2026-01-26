import pickle
import pandas as pd
import numpy as np

try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    print("Model loaded successfully.")
    
    # Create a sample input matching the app logic
    input_df = pd.DataFrame({
        'batting_team': ['Mumbai Indians'],
        'bowling_team': ['Chennai Super Kings'],
        'city': ['Mumbai'],
        'runs_left': [50],
        'balls_left': [30],
        'wickets': [7],
        'total_runs_x': [200],
        'crr': [10.0],
        'rrr': [10.0]
    })
    
    print("Input DataFrame:")
    print(input_df)
    
    print("Predicting...")
    result = pipe.predict_proba(input_df)
    print("Result:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
