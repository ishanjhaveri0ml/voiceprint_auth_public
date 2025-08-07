import os
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATASET_PATH = "path_to_extracted_ravdess_audio_files"  # Set this to your dataset path
EMOTION_LABELS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path: str) -> np.ndarray:
    [Fs, x] = audioBasicIO.read_audio_file(file_path)
    x = audioBasicIO.stereo_to_mono(x)
    features = audioFeatureExtraction.st_feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    return np.mean(features, axis=1)

def get_emotion_from_filename(filename: str) -> str:
    emotion_code = filename.split("-")[2]
    return EMOTION_LABELS.get(emotion_code, "unknown")

def load_data(dataset_path: str):
    X, y = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                emotion = get_emotion_from_filename(file)
                if emotion == "unknown":
                    continue
                features = extract_features(filepath)
                X.append(features)
                y.append(emotion)
    return np.array(X), np.array(y)

def main():
    print("Loading data and extracting features...")
    X, y = load_data(DATASET_PATH)
    print(f"Extracted features from {len(X)} files.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, "emotion_svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Model, scaler, and label encoder saved.")

if __name__ == "__main__":
    main()
