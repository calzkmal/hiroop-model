from predictor import extract_features
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model_path = 'hiroop-env\model\modelv3.keras'
model = tf.keras.models.load_model(model_path)

label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

@app.route("/predict-audio", methods=['POST'])
def predict_audio():
    data = request.get_json()
    
    if 'audio_path' not in data:
        return jsonify({"error": "No audio data provided"}), 400
    
    audio_path = data['audio_path']
    features = extract_features(audio_path)
    
    if features is not None:
        features = np.expand_dims(features, axis=(0, 2))
        prediction = model.predict(features)
        prediction_probabilities = {label: float(prob) for label, prob in zip(label_classes, prediction[0])}
        return jsonify(prediction_probabilities)
    else:
        return jsonify({"error": "Failed to extract features for prediction"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
