import numpy as np
import tensorflow as tf
from pipeline import FeatureExtractor  # Import the FeatureExtractor class

# Define the paths for the model and audio file
model_path = 'model\\modelv3.keras'
audio_path = 'data\\uji_healthy.wav'

# Define the class labels
label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialize the FeatureExtractor
extractor = FeatureExtractor()

# Extract features from the audio file
features = extractor.get_features(audio_path)

if features is not None:
    # Reshape the data to match the model input shape (None, 162, 1)
    features = np.expand_dims(features, axis=(0, 2))
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Convert prediction to a dictionary with probabilities and labels
    prediction_probabilities = {label: float(prob) for label, prob in zip(label_classes, prediction[0])}
    
    # Display the labels and their associated probability scores
    for label, prob in prediction_probabilities.items():
        print(f"{label}: {prob * 100:.2f}%")
else:
    print("Failed to extract features for prediction.")
