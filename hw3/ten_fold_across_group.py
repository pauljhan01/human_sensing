import skimage as ski
import pandas as pd
import sklearn as skl
from PIL import Image
import imageio.v3 as iio
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, StratifiedKFold, StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow import keras
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import backend as K

class Gestures():
    def __init__(self, user, gesture, path):
        self.user = user
        self.gesture = gesture
        self.path = path
        self.load_image()
        self.apply_preprocessing()
    
    def load_image(self):
        image = Image.open(self.path)
        image = np.array(image)
        self.image = image

    def apply_preprocessing(self):
        # self.image = ski.exposure.equalize_adapthist(self.image)
        # self.image_resized = ski.transform.resize(self.image, (64, 64), anti_aliasing=True)
        # self.image_hist = ski.feature.hog(self.image, orientations=9, pixels_per_cell=(8, 8),
        #                        cells_per_block=(2, 2), visualize=False)
        # self.hog_image = ski.exposure.rescale_intensity(self.image_hist, in_range=(0,10))

        threshold = ski.filters.threshold_otsu(self.image)
        binary_mask = self.image > threshold

        coordinates = np.column_stack(np.where(binary_mask))

        if coordinates.size > 0:
            min_r, min_c = coordinates.min(axis=0)
            max_r, max_c = coordinates.max(axis=0)

            cropped_hand = self.image[min_r:max_r, min_c:max_c]

            self.processed_image = ski.transform.resize(cropped_hand, (128, 128), anti_aliasing=True)

        else:
            self.processed_image = ski.transform.resize(self.image, (128, 128))
        
        self.features = ski.feature.hog(self.processed_image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2,2), visualize=False)
data = []
groups = []
labels = []
subjects = ["00", "01", "02", "03", "04"]
gestures = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
# subjects = ["00", "01"]
# gestures = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index"]
def load_data():

    for subject in subjects:
        for gesture in gestures:
            path = "gestures_database/" + f"{subject}/" + f"{gesture}/"
            for file in os.listdir(path):
                name = "".join(gesture.split("_")[1:])
                full_path = path + f"{file}"
                image = Gestures(subject, name, full_path)
                data.append(image.processed_image)
                groups.append(int(subject[1]))
                labels.append(name)

load_data()
groups = np.array(groups)
X = data
y = labels
labeler = LabelEncoder()
y = labeler.fit_transform(y)
y = np.array(y)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
X = np.array(X)

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(128,128,1)),
            keras.layers.Conv2D(64, (3,3), activation="relu"),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(32, (3,3), activation="relu"),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation="relu"),
            keras.layers.MaxPooling2D((2,2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10)
        ]
    )

    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model

# pipeline = Pipeline([
#     # ('label', LabelEncoder()),
#     ('scale', StandardScaler()),
#     ('svc', SVC(kernel='rbf', C=10, gamma='scale'))
# ])

# svc = SVC(kernel='rbf', C=10, gamma='scale', class_weight="balanced")

model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)

# logo = LeaveOneGroupOut()
# y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo)
# print(classification_report(y, y_pred=y_pred, zero_division=1))

# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)
# y_pred = cross_val_predict(model, X, y, cv=cv)
# print(classification_report(y, y_pred))

for subject in subjects:
    X_subject = X[groups == int(subject[1])]
    y_subject = y[groups == int(subject[1])]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)

    y_pred = cross_val_predict(model, X_subject, y_subject, cv=cv)
    print(classification_report(y_subject, y_pred))

    K.clear_session()

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2001)

# clf = SVC(kernel='rbf', C=10, gamma='scale')
# clf.fit(x_train, y_train)

# y_pred = clf.predict(x_test)
# print(classification_report(y_test, y_pred))
