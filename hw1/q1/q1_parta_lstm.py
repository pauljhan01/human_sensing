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

scaler = StandardScaler()
df[['x', 'y', 'z']] = scaler.fit_transform(df[['x', 'y', 'z']])

def create_sequences(data, window_size=200, step_size=100):
    X, y = [], []
    for activity in data['label'].unique():
        subset = data[data['label'] == activity]
        # Use raw x, y, z as features (3 channels)
        features = subset[['x', 'y', 'z']].values
        labels = subset['label'].values
        
        for i in range(0, len(features) - window_size, step_size):
            X.append(features[i : i + window_size])
            y.append(labels[i]) # Assuming label is constant in the window
            
    return np.array(X), np.array(y)

X, y = create_sequences(df)

# sample_plot = df.iloc[500:1000]
# plt.figure(figsize=(12, 6))
# plt.plot(sample_plot['seconds_elapsed'], sample_plot['magnitude'], label='Raw Magnitude', alpha=0.5)
# plt.plot(sample_plot['seconds_elapsed'], sample_plot['magnitude_no_gravity'], label='Filtered (No Gravity)', linewidth=2)
# plt.title('Preprocessing: Raw vs. Filtered Signal')
# plt.legend()
# plt.savefig("fig.png")

# window_size = 200
# step_size = 100
# frames, labels = q1_extract_data.create_frames(df, window_size, step_size)

# X = q1_extract_data.extract_features(frames)
# y = labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_one_hot = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

model = Sequential([
    # Input shape: (Time Steps=200, Features=3)
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(len(le.classes_), activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training LSTM model...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLSTM Test Accuracy: {accuracy*100:.2f}%")

