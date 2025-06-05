from simple_feature_extraction import AudioFeatureExtractor
from simple_neural_model import SimpleNeuralMOSPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import load_dataset
from rule_predictor import RulesBasedMOSPredictor
import joblib
import torch
from run_inference_deep_network import load_model
import numpy as np

def main():
    """Main function to demonstrate both approaches"""
    print("Loading dataset...")
    data = load_dataset("urgent-challenge/urgent2024_mos")
    
    subset_size = 50
    test_data = data["test"].shuffle(seed=90).select(range(subset_size))
    print(f"Processing {len(test_data)} samples...")
    extractor = AudioFeatureExtractor()    
    features_list = []
    mos_scores = []
    audio_list = []
    for i, sample in enumerate(test_data):
        print(f"Processing sample {i+1}/{len(test_data)}")
        audio = sample['audio']['array']
        audio_list.append(audio)
        mos = sample['mos']
        sample_rate = sample['sampling_rate']
        
        try:
            features = extractor.extract_all_features(audio,sample_rate)
            features_list.append(features)
            mos_scores.append(mos)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {len(features_list)} samples")
    
    if len(features_list) < 10:
        print("Not enough samples for meaningful evaluation")
        return
    

    
    # Rules-based approach
    print("\n=== Rules-Based Approach ===")
    rules_predictor = RulesBasedMOSPredictor()
    rules_predictions = rules_predictor.predict(features_list)
    print(rules_predictions)
    rules_mse = mean_squared_error(mos_scores, rules_predictions)
    rules_mae = mean_absolute_error(mos_scores, rules_predictions)
    
    print(f"Rules-based MSE: {rules_mse:.3f}")
    print(f"Rules-based MAE: {rules_mae:.3f}")
    
    # SMALL Neural network approach
    print("\n=== Neural Network Approach ===")
    saved = joblib.load("inference/simple_neural_mos_model.joblib")
    nn_predictor = SimpleNeuralMOSPredictor()
    nn_predictor.scaler = saved['scaler']
    nn_predictor.model = saved['model']
    nn_predictor.feature_names = saved['feature_names']

    nn_predictions = nn_predictor.predict(features_list)
    
    nn_mse = mean_squared_error(mos_scores, nn_predictions)
    nn_mae = mean_absolute_error(mos_scores, nn_predictions)
    print(f"Neural Network MSE: {nn_mse:.3f}")
    print(f"Neural Network MAE: {nn_mae:.3f}")
    
    # DEEP Neural network approach
    print("\n=== Deep Neural Network Approach ===")
    deep_network_predictor = load_model()
    features_list= [extractor.features_to_tensor(features) for features in features_list]
    deep_network_predictions = []
    for i, audio in enumerate(audio_list):
        audio_torch = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        torch_features = features_list[i].unsqueeze(0)
        pred = deep_network_predictor(audio_torch, torch_features).detach().cpu().numpy()
        deep_network_predictions.append(float(pred.item()))

    deep_network_predictions = np.array(deep_network_predictions)
    deep_network_mse = mean_squared_error(mos_scores, deep_network_predictions)
    deep_network_mae = mean_absolute_error(mos_scores, deep_network_predictions)
    print(f"Deep Neural Network MSE: {deep_network_mse:.3f}")
    print(f"Deep Neural Network MAE: {deep_network_mae:.3f}")
    

if __name__ == "__main__":
    main()