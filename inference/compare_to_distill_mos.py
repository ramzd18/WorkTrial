import distillmos
from datasets import load_dataset
import torch
import numpy as np
from rule_predictor import RulesBasedMOSPredictor
from simple_feature_extraction import AudioFeatureExtractor
from sklearn.metrics import mean_squared_error
import joblib
from simple_neural_model import SimpleNeuralMOSPredictor
from run_inference_deep_network import load_model

def pad_or_truncate(audio, target_length=16000): 
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio

def predict_distillmos_on_commonvoice(num_samples=50):
    model = distillmos.ConvTransformerSQAModel()
    model.eval()
    
    rules_predictor = RulesBasedMOSPredictor()
    feature_extractor = AudioFeatureExtractor()
    saved = joblib.load("inference/simple_neural_mos_model.joblib")
    nn_predictor = SimpleNeuralMOSPredictor()
    nn_predictor.scaler = saved['scaler']
    nn_predictor.model = saved['model']
    nn_predictor.feature_names = saved['feature_names']
    deep_network_predictor = load_model()

    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True)
    dataset = dataset.take(num_samples)
    dataset = list(dataset)  

    processed_audio = [pad_or_truncate(sample["audio"]["array"]) for sample in dataset]
    audio_arrays = torch.tensor(processed_audio, dtype=torch.float32)

    sampling_rate = 16000

    with torch.no_grad():
        distillmos_predictions = model(audio_arrays)
    deep_network_predictions = []
    rules_predictions = []
    features_list = []
    valid_indices = []
    
    for i, audio in enumerate(processed_audio):
        try:
            features = feature_extractor.extract_all_features(torch.tensor(audio), sampling_rate)
            feature_values = list(features.values())
            if any(np.isinf(val) for val in feature_values if isinstance(val, (int, float))) or \
               any(np.isnan(val) for val in feature_values if isinstance(val, (int, float))):
                continue
                
            features_list.append(features)
            rules_pred = rules_predictor.predict_single(features)
            rules_predictions.append(rules_pred)
            valid_indices.append(i)

            features_squeezed = feature_extractor.features_to_tensor(features).unsqueeze(0)
            audio_arr = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            deep_network_pred = deep_network_predictor(audio_arr, features_squeezed)
            deep_network_predictions.append(deep_network_pred)
        except Exception as e:
            print(f"Skipping sample {i} due to error: {str(e)}")
            continue

    rules_predictions = np.array(rules_predictions)
    distillmos_predictions = distillmos_predictions[valid_indices]
    small_network_predictions = nn_predictor.predict(features_list)
    
    distillmos_np = distillmos_predictions.detach().cpu().numpy().reshape(-1)
    rules_mse = mean_squared_error(distillmos_np, rules_predictions)
    deep_network_mse = mean_squared_error(distillmos_np, 
                                        np.array([pred.detach().cpu().numpy().item() for pred in deep_network_predictions]))
    small_network_mse = mean_squared_error(distillmos_np, small_network_predictions)
    
    return {
        'distillmos_predictions': distillmos_predictions,
        'rules_predictions': rules_predictions,
        'deep_network_predictions': deep_network_predictions,
        'small_network_predictions': small_network_predictions,
        'rules_mse': rules_mse,
        'deep_network_mse': deep_network_mse,
        'small_network_mse': small_network_mse
    }

if __name__ == "__main__":
    results = predict_distillmos_on_commonvoice(500)
    print("SMALL NETWORK PREDICTIONS", results['small_network_predictions'])
    print("\nMSE between  Rule predictions:", results['rules_mse'])
    print("\nMSE between  Deep Network predictions:", results['deep_network_mse'])
    print("\nMSE between  Small Network predictions:", results['small_network_mse'])
