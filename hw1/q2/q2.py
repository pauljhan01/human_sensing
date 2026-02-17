import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def count_steps(file_path):
    df = pd.read_csv(file_path)
    
    acc_x = df['accelerometerAccelerationX(G)']
    acc_y = df['accelerometerAccelerationY(G)']
    acc_z = df['accelerometerAccelerationZ(G)']
    
    # acc_x = df['x']
    # acc_y = df['y']
    # acc_z = df['z']

    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Removes gravity (high-pass) and high-frequency noise (low-pass)
    fs = 30.0  
    lowcut = 0.5
    highcut = 3.0
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    filtered_mag = filtfilt(b, a, mag)
    
    peaks, _ = find_peaks(filtered_mag, height=0.20, distance=fs/4)

    return len(peaks)

print(count_steps("steps.csv"))