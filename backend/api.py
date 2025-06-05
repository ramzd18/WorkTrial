# Create real time MOS prediction API
from flask import Flask, request, jsonify
import torch
import numpy as np
from inference.simple_feature_extraction import AudioFeatureExtractor
from inference.run_inference_deep_network import load_model
import torchaudio
import io
import soundfile as sf
import tempfile
import os

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model()
feature_extractor = AudioFeatureExtractor()

def process_audio(audio_data, sample_rate):
    # Process audio data and return prediction

    try:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        features = feature_extractor.extract_all_features(audio_tensor, sample_rate)
        features_tensor = feature_extractor.features_to_tensor(features).unsqueeze(0)
        with torch.no_grad():
            prediction = model(audio_tensor, features_tensor)
        
        return float(prediction.item())
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    # Endpoint for MOS prediction
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            audio_data, sample_rate = sf.read(temp_file.name)
            os.unlink(temp_file.name)
        mos_score = process_audio(audio_data, sample_rate)
        
        return jsonify({
            'mos_score': mos_score,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Health check endpoint
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)