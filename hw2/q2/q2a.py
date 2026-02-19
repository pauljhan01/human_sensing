import tensorflow as tf
import tensorflow_hub as hub
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

animals = ["Bird", "Cat", "Chicken", "Cow", "Dog", "Donkey", "Frog", "Lion", "Monkey", "Sheep"]
dir = "../q2_data/animal-sounds-dataset/"

vocal_path = "../q2_data/cat-vocalization-dataset/"
vocal_labels = {"B":"brushing", "I":"isolated", "F":"waiting"}
vocal_features = []
vocal_labels = []

for filename in os.listdir(vocal_path):
    name = filename.split("_")
    label = vocal_labels[name[0]]

    path = vocal_path + filename
    y, sr = librosa.load(path)

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features = np.hstack([mfccs, centroid, rolloff])

    vocal_labels.append(label)
    vocal_features.append(features)

x = np.array(vocal_features)
le = LabelEncoder()
y = le.fit_transform(vocal_labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2001)
vocal_classifier = RandomForestClassifier(n_estimators=100)
vocal_classifier.fit(x_train, y_train)
y_pred = vocal_classifier.predict(x_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

for animal in animals:
    for filename in os.listdir(dir + animal):
        path = dir + animal + filename
        audio, sr = librosa.load(path)

        scores, embeddings, spectrogram = yamnet_model(audio)
