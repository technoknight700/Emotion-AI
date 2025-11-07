# ============================================
# 🎙️ Emotion AI Server (app.py)
# ============================================
import os
import numpy as np
import librosa
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pydub import AudioSegment

# ============================================
# 1️⃣ Config
# ============================================
N_MFCC = 13
MAX_LEN = 100
TEMP_FILE = "temp_audio_file.wav"

# ============================================
# 2️⃣ Initialize Flask App
# ============================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# ============================================
# 3️⃣ Load Model and Artifacts
# ============================================
try:
    model_path = "best_voice_emotion_model.keras"  # use the correct file name
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)
    le = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    print("✅ Model, label encoder, and scaler loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model or artifacts: {e}")
    print("Ensure 'best_voice_emotion_model.keras', 'label_encoder.pkl', and 'scaler.pkl' exist.")
    model, le, scaler = None, None, None

# ============================================
# 4️⃣ Feature Extraction Function
# ============================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate to fixed length
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc.T
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# ============================================
# 5️⃣ API Endpoint
# ============================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or le is None or scaler is None:
        return jsonify({"error": "Model not loaded. Restart the server with valid files."}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    file = request.files["audio"]

    try:
        # Convert webm/opus to wav using pydub
        audio = AudioSegment.from_file(file)
        audio.export(TEMP_FILE, format="wav")

        # Extract features
        features = extract_features(TEMP_FILE)
        if features is None:
            return jsonify({"error": "Failed to process audio. Try again with a clearer clip."}), 400

        # Scale and reshape
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
        features_ready = features_scaled.reshape(1, MAX_LEN, N_MFCC)

        # Predict emotion
        prediction = model.predict(features_ready)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_emotion = le.inverse_transform(predicted_index)[0]

        print(f"🎤 Predicted Emotion: {predicted_emotion}")
        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        print(f"❌ Server error during prediction: {e}")
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500

    finally:
        # Cleanup temp file
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

# ============================================
# 6️⃣ Run Server
# ============================================
if __name__ == "__main__":
    print("🚀 Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
