from simple_feature_extraction import AudioFeatureExtractor
from simple_neural_model import SimpleNeuralMOSPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import load_dataset
from rule_predictor import RulesBasedMOSPredictor

def main():
    """Main function to demonstrate both approaches"""
    print("Loading dataset...")
    data = load_dataset("urgent-challenge/urgent2024_mos")
    
    subset_size = 200  
    test_data = data["test"].select(range(min(subset_size, len(data["test"]))))
    print(f"Processing {len(test_data)} samples...")
    extractor = AudioFeatureExtractor()    
    features_list = []
    mos_scores = []
    
    for i, sample in enumerate(test_data):
        print(f"Processing sample {i+1}/{len(test_data)}")
        
        audio = sample['audio']['array']
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
    
    train_features, test_features, train_mos, test_mos = train_test_split(
        features_list, mos_scores, test_size=0.3, random_state=42
    )
    
    # Rules-based approach
    print("\n=== Rules-Based Approach ===")
    rules_predictor = RulesBasedMOSPredictor()
    rules_predictions = rules_predictor.predict(test_features)
    print(rules_predictions)
    rules_mse = mean_squared_error(test_mos, rules_predictions)
    rules_mae = mean_absolute_error(test_mos, rules_predictions)
    
    print(f"Rules-based MSE: {rules_mse:.3f}")
    print(f"Rules-based MAE: {rules_mae:.3f}")
    
    # Neural network approach
    print("\n=== Neural Network Approach ===")
    nn_predictor = SimpleNeuralMOSPredictor()
    nn_predictor.train(train_features, train_mos)
    nn_predictions = nn_predictor.predict(test_features)
    
    nn_mse = mean_squared_error(test_mos, nn_predictions)
    nn_mae = mean_absolute_error(test_mos, nn_predictions)
    
    print(f"Neural Network MSE: {nn_mse:.3f}")
    print(f"Neural Network MAE: {nn_mae:.3f}")
    

if __name__ == "__main__":
    main()