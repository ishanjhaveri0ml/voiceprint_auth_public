import os
import numpy as np
import torchaudio
import sounddevice as sd
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load Speaker Recognition model and Emotion Classifier once
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)
emotion_model = joblib.load("emotion_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

def record_audio(filename: str = "input.wav", duration: int = 5, sr: int = 16000):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)

def extract_embedding(audio_path: str) -> np.ndarray:
    embedding = speaker_model.encode_batch(audio_path)
    return embedding.squeeze().detach().numpy()

def save_voiceprint(user_id: str, embedding: np.ndarray, db_path: str = "voice_db"):
    os.makedirs(db_path, exist_ok=True)
    np.save(os.path.join(db_path, f"{user_id}.npy"), embedding)

def load_voiceprints(db_path: str = "voice_db") -> dict:
    data = {}
    if not os.path.exists(db_path):
        return data
    for file in os.listdir(db_path):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            emb = np.load(os.path.join(db_path, file))
            data[name] = emb
    return data

def verify_user(claimed_name: str, input_embedding: np.ndarray, db_path: str = "voice_db", threshold: float = 0.75):
    voice_db = load_voiceprints(db_path)
    if claimed_name not in voice_db:
        return False, "User not found"
    similarity = 1 - cosine(input_embedding, voice_db[claimed_name])
    return similarity > threshold, f"Similarity: {similarity:.2f}"

def recognize_user(input_embedding: np.ndarray, db_path: str = "voice_db"):
    voice_db = load_voiceprints(db_path)
    best_match, best_score = None, -1
    for name, emb in voice_db.items():
        similarity = 1 - cosine(input_embedding, emb)
        if similarity > best_score:
            best_match, best_score = name, similarity
    return best_match, best_score

def extract_features(file_name: str) -> np.ndarray:
    [Fs, x] = audioBasicIO.read_audio_file(file_name)
    x = audioBasicIO.stereo_to_mono(x)
    features = audioFeatureExtraction.st_feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    return np.mean(features, axis=1)

def detect_emotion(audio_path: str) -> str:
    features = extract_features(audio_path)
    features = scaler.transform([features])
    label_idx = emotion_model.predict(features)[0]
    emotion_label = le.inverse_transform([label_idx])[0]
    return emotion_label
