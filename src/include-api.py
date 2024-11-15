from pipeline import FeatureExtractor  # Import the FeatureExtractor class
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
model_path = 'model\\model.keras'
model = tf.keras.models.load_model(model_path)

# Define the class labels
label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

# Initialize the FeatureExtractor
extractor = FeatureExtractor()

@app.route("/predict-audio", methods=['POST'])
def predict_audio():
    data = request.get_json()
    
    if 'audio_path' not in data:
        return jsonify({"error": "No audio data provided"}), 400
    
    audio_path = data['audio_path']
    
    # Extract features from the audio file
    features = extractor.get_features(audio_path)
    
    if features is not None:
        # Reshape the data to match the model input shape (None, 162, 1)
        features = np.expand_dims(features, axis=(0, 2))
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Convert prediction to a dictionary with probabilities and labels
        prediction_probabilities = {label: float(prob) for label, prob in zip(label_classes, prediction[0])}
        
        return jsonify(prediction_probabilities)
    else:
        return jsonify({"error": "Failed to extract features for prediction"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
