import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# path = "../q1_data/"

# coffee_path = path + "coffee.wav"
# kitchen_path = path + "kitchen.wav"
# party_path = path + "party.mp3"
# soccer_path = path + "soccer.wav"

sr=22050

# coffee_data, _ = librosa.load(path=coffee_path, sr=sr)
# kitchen_data, _ = librosa.load(path=kitchen_path, sr=sr)
# party_path, _ = librosa.load(path=party_path, sr=sr)
# soccer_path, _ = librosa.load(path=soccer_path, sr=sr)

def extract_features(file_path, duration=30, frame_length=0.025, hop_length=0.010):
    y, sr = librosa.load(file_path, duration=duration)

    n_fft = int(frame_length * sr)
    hop_size = int(hop_length * sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_size)
    return mfccs.T

dir = "../q1_data/"
labels_names = ["coffee", "kitchen", "soccer", "party"]
filenames = ["coffee.wav", "kitchen.wav", "soccer.wav", "party.mp3"]

features = []
labels = []

for i in range(len(labels_names)):
    path = dir + filenames[i]
    print(len(labels_names))
    frames = extract_features(path)

    features.append(frames)
    labels.append([labels_names[i]] * len(frames))

X = np.vstack(features)
y = np.concatenate(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=67)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

print("Random Forest")
print(classification_report(y_true=y_test, y_pred=rf_pred, target_names=le.classes_))

print("SVM")
print(classification_report(y_true=y_test, y_pred=svm_pred, target_names=le.classes_))

print("K Nearest Neighbors")
print(classification_report(y_true=y_test, y_pred=knn_pred, target_names=le.classes_))