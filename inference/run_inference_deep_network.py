import torch
import torchaudio
import argparse
from pathlib import Path
from deep_network import MultiModalMOSPredictor
from simple_feature_extraction import AudioFeatureExtractor

def load_model(model_path='inference/mos_deep_network.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load the saved model
    model = MultiModalMOSPredictor(feature_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def process_audio(audio_path, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Process a single audio file and get MOS prediction
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    feature_extractor = AudioFeatureExtractor()
    features = feature_extractor.extract_all_features(audio, sr)
    features_tensor = feature_extractor.features_to_tensor(features).unsqueeze(0)
    print("FEATURES TENSOR SHAPE", features_tensor.shape)
    audio = audio.to(device)
    features_tensor = features_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(audio, features_tensor)
    
    return prediction.item()

def main():
    parser = argparse.ArgumentParser(description='Run MOS prediction on audio files')
    parser.add_argument('--model_path', type=str, default='inference/mos_deep_network.pth',
                      help='Path to the saved model')
    parser.add_argument('--audio_path', type=str, default="data/somos/somos_16.wav",
                      help='Path to the audio file or directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.device)
    print("Model loaded successfully!")
    
    # Process audio
    audio_path = Path(args.audio_path)
    if audio_path.is_file():
        print(f"\nProcessing {audio_path}...")
        mos_score = process_audio(str(audio_path), model, args.device)
        print(f"Predicted MOS Score: {mos_score:.2f}")
    elif audio_path.is_dir():
        # Directory of files
        print(f"\nProcessing all audio files in {audio_path}...")
        for audio_file in audio_path.glob('*.wav'):
            print(f"\nProcessing {audio_file.name}...")
            mos_score = process_audio(str(audio_file), model, args.device)
            print(f"Predicted MOS Score: {mos_score:.2f}")
    else:
        print(f"Error: {args.audio_path} is not a valid file or directory")

if __name__ == "__main__":
    main()
