# ============================================
# 🎙️ Voice Emotion Recognition – Final Version
# ============================================
import os
import numpy as np
import librosa
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================
# 1️⃣ Config
# ============================================
DATASET_PATH = r"C:\Users\nitaa\Downloads\FOML_project"  # change this
N_MFCC = 13
MAX_LEN = 100  # number of time frames to pad/truncate MFCCs to

# RAVDESS emotion labels
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# ============================================
# 2️⃣ Feature Extraction
# ============================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # Pad or truncate to MAX_LEN
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc.T  # shape -> (time_steps, n_mfcc)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# ============================================
# 3️⃣ Load Dataset
# ============================================
data, labels = [], []

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            try:
                emotion_code = file.split("-")[2]
                if emotion_code in emotion_dict:
                    feature = extract_features(os.path.join(root, file))
                    if feature is not None:
                        data.append(feature)
                        labels.append(emotion_dict[emotion_code])
            except Exception as e:
                print("Error:", e)

print(f"✅ Extracted features: {len(data)} samples")
print("🎭 Label distribution:", Counter(labels))

# ============================================
# 4️⃣ Preprocessing
# ============================================
X = np.array(data)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Flatten for scaling (we’ll reshape back)
X_reshaped = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)

# ============================================
# 5️⃣ Build Model
# ============================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(MAX_LEN, N_MFCC)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ============================================
# 6️⃣ Train
# ============================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # *** FIX: Changed file extension from .h5 to .keras ***
    ModelCheckpoint('best_voice_emotion_model.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 7️⃣ Evaluate
# ============================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Final Test Accuracy: {acc * 100:.2f}%")

# ============================================
# 8️⃣ Save Artifacts
# ============================================
# *** FIX: Changed file extension from .h5 to .keras ***
model.save("voice_emotion_model.keras")
print("✅ Model saved as voice_emotion_model.keras")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("✅ Label encoder saved as label_encoder.pkl")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved as scaler.pkl")

# ============================================
# 9️⃣ Optional: Classification Report
# ============================================
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))
