import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import librosa
import os
import numpy as np

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

animals = ["Bird", "Cat", "Chicken", "Cow", "Dog", "Donkey", "Frog", "Lion", "Monkey", "Sheep"]
dir = "../q2_data/animal-sounds-dataset/"

labels = []
animal_feat = []
for i in range(len(animals)):
    for filename in os.listdir(dir + animals[i]):
        path = dir + animals[i] + "/" + filename
        audio, sr = librosa.load(path, sr=16000)

        scores, embeddings, spectrogram = yamnet_model(audio)
        features = np.mean(embeddings, axis=0)
        animal_feat.append(features)

        if "cat" in filename:
            labels.append(1)
        else:
            labels.append(0)

x = np.array(animal_feat)
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2001)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(classification_report(y_test, y_pred, target_names=["Not Cat", "Cat"]))