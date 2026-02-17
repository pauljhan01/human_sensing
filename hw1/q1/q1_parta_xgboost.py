import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import q1_extract_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB          
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

with open("paulhan-q1-data.csv", "w") as f:
    with open("../Car/Accelerometer.csv") as f1:
        for line in f1:
            if "time" in line:
                line = line.strip("\n")
                line += ",label\n"
                f.write(line)
                continue

            line = line.strip("\n")
            line += ",car\n"
            f.write(line)
    
    with open("../Stairs/Accelerometer.csv") as f1:
        for line in f1:
            if "time" in line:
                continue

            line = line.strip("\n")
            line += ",stairs\n"
            f.write(line)
    
    with open("../Walk/Accelerometer.csv") as f1:
        for line in f1:
            if "time" in line:
                continue
        
            line = line.strip("\n")
            line += ",walk\n"
            f.write(line)

    with open("../Dancing/Accelerometer.csv") as f1:
        for line in f1:
            if "time" in line:
                continue
        
            line = line.strip("\n")
            line += ",dance\n"
            f.write(line)

    with open("../Running/Accelerometer.csv") as f1:
        for line in f1:
            if "time" in line:
                continue
        
            line = line.strip("\n")
            line += ",run\n"
            f.write(line)

path = "paulhan-q1-data.csv"
df = pd.read_csv(path)

df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

fs = 100.0 #0.1 kHz = 100 Hz
lowcut = 20.0
highcut = 0.5

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

df['magnitude_smooth'] = butter_lowpass_filter(df['magnitude'], lowcut, fs)

df['magnitude_no_gravity'] = butter_highpass_filter(df['magnitude_smooth'], highcut, fs)

frames, labels = q1_extract_data.create_frames(df, 200, 100)

X = q1_extract_data.extract_features(frames)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

boost_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
boost_model.fit(X_train, y_train_encoded)
y_pred_boost = boost_model.predict(X_test)

print("XGBoost")
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_boost):.2f}")
print(classification_report(y_test_encoded, y_pred_boost))



