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

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

animals = ["Bird", "Cat", "Chicken", "Cow", "Dog", "Donkey", "Frog", "Lion", "Monkey", "Sheep"]
dir = "../q2_data/animal-sounds-dataset/"

vocal_path = "../q2_data/cat-vocalization-dataset/"
vocal_label_dict = {"B":"brushing", "I":"isolated", "F":"waiting"}
vocal_features = []
vocal_labels = []

for filename in os.listdir(vocal_path):
    name = filename.split("_")[0]
    label = vocal_label_dict[name]
    print(filename)
    print(label)

    path = vocal_path + filename
    y, sr = librosa.load(path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.hstack([np.mean(mfccs), np.mean(delta_mfccs), np.mean(delta2_mfccs)])

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
#vocal_classifier = RandomForestClassifier(n_estimators=100)
#vocal_classifier.fit(x_train, y_train)
#y_pred = vocal_classifier.predict(x_test)
vocal_pipeline.fit(x_train, y_train)
y_pred = vocal_pipeline.predict(x_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

preds = []
avg_cat_score = []
for animal in animals:
    for filename in os.listdir(dir + animal):
        path = dir + animal + "/" + filename
        audio, sr = librosa.load(path, sr=16000)

        scores, embeddings, spectrogram = yamnet_model(audio)
        cat_score = np.mean(scores, axis=0)[79]

        if cat_score > 0.005:
            print("Cat detected")
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
# 
            audio_features = np.hstack([np.mean(mfccs), np.mean(delta_mfccs), np.mean(delta2_mfccs)])
            audio_features = audio_features.reshape(1, -1)
            prediction = vocal_pipeline.predict(audio_features)[0]
#
            print(f"Cat behavior: {prediction}")
#

