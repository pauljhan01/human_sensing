import tensorflow as tf
import tensorflow_hub as hub
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from scipy.stats import skew, kurtosis

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

animals = ["Bird", "Cat", "Chicken", "Cow", "Dog", "Donkey", "Frog", "Lion", "Monkey", "Sheep"]
dir = "../q2_data/animal-sounds-dataset/"

vocal_path = "../q2_data/cat-vocalization-dataset/"
vocal_label_dict = {"B":"brushing", "I":"isolated", "F":"waiting"}
vocal_features = []
vocal_labels = []

def get_stats(feat):
    return np.hstack([
        np.mean(feat, axis=1), 
        np.std(feat, axis=1),
        skew(feat, axis=1),
        np.max(feat, axis=1)
    ])

for filename in os.listdir(vocal_path):
    name = filename.split("_")[0]
    label = vocal_label_dict[name]

    path = vocal_path + filename
    y, sr = librosa.load(path)
    intervals = librosa.effects.split(y, top_db=25)
    y = np.concatenate([y[start:end] for start, end in intervals])
    # y, _ = librosa.effects.trim(y)

    y = librosa.util.normalize(y)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    y_harm, y_perc = librosa.effects.hpss(y)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.hstack([
        get_stats(mfcc), 
        get_stats(centroid), 
        get_stats(flatness),
        get_stats(rolloff),
        np.mean(zcr)
    ])

    vocal_labels.append(label)
    vocal_features.append(features)

x = np.array(vocal_features)
le = LabelEncoder()
y = le.fit_transform(vocal_labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2001)
vocal_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    #('knn', KNeighborsClassifier(10))
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])

vocal_pipeline.fit(x_train, y_train)
y_pred = vocal_pipeline.predict(x_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))





