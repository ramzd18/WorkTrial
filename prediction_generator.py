from simple_feature_extraction import AudioFeatureExtractor
from simple_neural_model import SimpleNeuralMOSPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import load_dataset
from rule_predictor import RulesBasedMOSPredictor
from data_extraction.load_somos import retrieve_dataset as retrieve_somos_dataset
from data_extraction.load_bvcc import retrieve_dataset as retrieve_bvcc_dataset
import os
import json
import soundfile as sf
import glob
import shutil

def main():
    """Main function to demonstrate both approaches"""
    print("Loading dataset...")
    data = load_dataset("urgent-challenge/urgent2024_mos")
    
    subset_size = 500  
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
    
def complete_data_local_download(): 
    data = load_dataset("urgent-challenge/urgent2024_mos")
    test_data = data["test"]
    urgent_data_list=[]
    for data in test_data:
        mos = data['mos']
        audio = data['audio']['array']
        sample_rate = data['sampling_rate']
        urgent_data_list.append((audio, mos, sample_rate))
    bvcc_data_list=retrieve_bvcc_dataset()
    somos_data_list = retrieve_somos_dataset()
    complete_data_list = urgent_data_list + bvcc_data_list + somos_data_list
    
    os.makedirs('data', exist_ok=True)
    
    print("Saving datasets locally...")
    for idx, (audio, mos, sr) in enumerate(urgent_data_list):
        sf.write(f'data/urgent_{idx}.wav', audio, sr)
        
    for idx, (audio, mos) in enumerate(bvcc_data_list):
        sf.write(f'data/bvcc_{idx}.wav', audio, 16000)
        
    for idx, (audio, mos) in enumerate(somos_data_list):
        sf.write(f'data/somos_{idx}.wav', audio, 16000)
        
    mos_data = {
        'urgent': [(idx, mos,sr) for idx, (_, mos, sr) in enumerate(urgent_data_list)],
        'bvcc': [(idx, mos) for idx, (_, mos) in enumerate(bvcc_data_list)],
        'somos': [(idx, mos) for idx, (_, mos) in enumerate(somos_data_list)]
    }
    
    with open('data/mos_scores.json', 'w') as f:
        json.dump(mos_data, f)
        
def reformat_data():
    os.makedirs('data/urgent', exist_ok=True) 
    os.makedirs('data/bvcc', exist_ok=True)
    os.makedirs('data/somos', exist_ok=True)
    
    for file in glob.glob('data/urgent_*.wav'):
        filename = os.path.basename(file)
        shutil.move(file, os.path.join('data/urgent', filename))
        
    for file in glob.glob('data/bvcc_*.wav'):
        filename = os.path.basename(file)
        shutil.move(file, os.path.join('data/bvcc', filename))
        
    for file in glob.glob('data/somos_*.wav'):
        filename = os.path.basename(file)
        shutil.move(file, os.path.join('data/somos', filename))
    
def data_length(): 
    wav_count = 0
    for folder in ['urgent', 'bvcc', 'somos']:
        folder_path = os.path.join('data', folder)
        if os.path.exists(folder_path):
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            wav_count += len(wav_files)
            
    return wav_count
if __name__ == "__main__":
    main()