import numpy as np
from scipy import stats
import pandas as pd

window_size = 200 
step_size = 100

def create_frames(data, window_size, step_size):
    frames = []
    labels = []
    
    for activity in data['label'].unique():
        activity_df = data[data['label'] == activity]
        
        for i in range(0, len(activity_df) - window_size, step_size):
            window = activity_df.iloc[i : i + window_size]
            frames.append(window['magnitude'].values)
            labels.append(activity)
            
    return np.array(frames), np.array(labels)

def extract_features(frames):
    feature_list = []
    for frame in frames:
        features = [
            np.mean(frame),
            np.std(frame),
            np.max(frame) - np.min(frame),
            stats.skew(frame),          
            np.sum(frame**2) / len(frame)
        ]
        feature_list.append(features)
    return np.array(feature_list)