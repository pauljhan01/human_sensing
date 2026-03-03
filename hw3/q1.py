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
        self.image = ski.exposure.equalize_adapthist(self.image)
        self.image_resized = ski.transform.resize(self.image, (64, 64), anti_aliasing=True)
        self.image_hist = ski.feature.hog(self.image_resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False)

data = []
groups = []
labels = []
subjects = ["00", "01", "02", "03", "04"]
gestures = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]

def load_data():
    # subjects = ["00"]
    # gestures = ["01_palm"]

    for subject in subjects:
        for gesture in gestures:
            path = "gestures_database/" + f"{subject}/" + f"{gesture}/"
            for file in os.listdir(path):
                name = "".join(gesture.split("_")[1:])
                full_path = path + f"{file}"
                image = Gestures(subject, name, full_path)
                data.append(image.image_hist)
                groups.append(int(subject[1]))
                labels.append(name)

load_data()
groups = np.array(groups)
X = data
y = labels
labeler = LabelEncoder()
y = labeler.fit_transform(y)
y = np.array(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.array(X)

# pipeline = Pipeline([
#     # ('label', LabelEncoder()),
#     ('scale', StandardScaler()),
#     ('svc', SVC(kernel='rbf', C=10, gamma='scale'))
# ])

svc = SVC(kernel='rbf', C=10, gamma='scale')

logo = LeaveOneGroupOut()
y_pred = cross_val_predict(svc, X, y, groups=groups, cv=logo)
print(classification_report(y, y_pred=y_pred))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)
y_pred = cross_val_predict(svc, X, y, cv=cv)
print(classification_report(y, y_pred))

for subject in subjects:
    X_subject = X[groups == int(subject[1])]
    y_subject = y[groups == int(subject[1])]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)

    cv_score = cross_val_score(svc, X_subject, y_subject, cv=cv)
    print(cv_score)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2001)

# clf = SVC(kernel='rbf', C=10, gamma='scale')
# clf.fit(x_train, y_train)

# y_pred = clf.predict(x_test)
# print(classification_report(y_test, y_pred))
