import json 
import torch 
import torchaudio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from simple_feature_extraction import AudioFeatureExtractor


# MOS dataset class
# Uses local data files in the data folder. Uses json file to map audio files to their mos scores. 
class MOSDataset(Dataset):
    def __init__(self, data_path='./data/mos_scores.json'):
        self.mos_scores = json.load(open(data_path))
        self.data = []
        
        for key, value in self.mos_scores.items():
            for idx, mos in value:
                audio_path = f'data/{key}/{key}_{idx}.wav'
                # Check if file exists
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                self.data.append({
                    'audio_path': audio_path,
                    'mos_score': mos,
                    'dataset': key
                })
        
        if len(self.data) == 0:
            raise ValueError("No valid audio files found in the dataset!")
        
        print(f"Successfully loaded {len(self.data)} audio files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            audio, sr = torchaudio.load(item['audio_path'])

            feature_extractor = AudioFeatureExtractor()
            features = feature_extractor.extract_all_features(audio, sr)
            features_tensor = feature_extractor.features_to_tensor(features)
            return {
                'audio': audio,
                'mos_score': torch.tensor(item['mos_score'], dtype=torch.float32),
                'dataset': item['dataset'],
                'features': features_tensor,
            }
        except Exception as e:
            print(f"Error loading audio file {item['audio_path']}: {str(e)}")
            # Return a zero tensor as fallback
            return {
                'audio': torch.zeros(1, 16000),  # Assuming 1 second of audio at 16kHz
                'mos_score': torch.tensor(item['mos_score'], dtype=torch.float32),
                'dataset': item['dataset'],
                'features': torch.zeros(1, 64)  
            }

def get_dataloaders(train_ratio=0.8, batch_size=1, num_workers=0):  
    """
    Create train and test dataloaders with limited batches for quick testing.
    
    Args:
        train_ratio (float): Ratio of training data (default: 0.8)
        batch_size (int): Batch size for both dataloaders (default: 1)
        num_workers (int): Number of worker processes (default: 0)
    
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    full_dataset = MOSDataset()
    
   
    train_size = int(0.75 * len(full_dataset))  # 300 samples for training
    test_size = len(full_dataset) - train_size  # 100 samples for testing
    
    print(f"Splitting subset: total={len(full_dataset)}, train={train_size}, test={test_size}")
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  
        pin_memory=True
    )
    
    print(f"Created dataloaders with {len(train_dataloader)} training batches and {len(test_dataloader)} test batches")
    return train_dataloader, test_dataloader
