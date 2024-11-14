import numpy as np
import tensorflow as tf
from pipeline import extract_features

# Definisikan path ke model dan file .wav
model_path = 'hiroop-env\\model\\modelv3.keras'
audio_path = 'hiroop-env\\data\\uji_healthy.wav'

# Definisikan label kelas
label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

# Load model
model = tf.keras.models.load_model(model_path)

# Ekstrak fitur dari file audio
features = extract_features(audio_path)

if features is not None:
    # Ubah bentuk data agar sesuai dengan input model (None, 162, 1)
    features = np.expand_dims(features, axis=(0, 2))
    
    # Prediksi
    prediction = model.predict(features)
    
    # Mengonversi hasil prediksi ke rentang probabilitas (0-1) dan menggabungkannya dengan label
    prediction_probabilities = {label: float(prob) for label, prob in zip(label_classes, prediction[0])}
    
    # Menampilkan label beserta skor probabilitas
    for label, prob in prediction_probabilities.items():
        print(f"{label}: {prob * 100:.2f}%")
else:
    print("Failed to extract features for prediction.")
