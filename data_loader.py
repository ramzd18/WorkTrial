import json 
import torch 
import torchaudio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from simple_feature_extraction import AudioFeatureExtractor

class MOSDataset(Dataset):
    def __init__(self, data_path='data/mos_scores.json'):
        self.mos_scores = json.load(open(data_path))
        self.data = []
        
        # Process all datasets
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
            # print("MADE It HERE 2")
            features = feature_extractor.extract_all_features(audio, sr)
            features_tensor = feature_extractor.features_to_tensor(features)
            # print("MADE It HERE")
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
                'features': torch.zeros(1, 128)  # Adjust size based on your feature dimensions
            }

def get_dataloaders(train_ratio=0.8, batch_size=1, num_workers=0):  # Set num_workers to 0
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
    
    # First, take a subset of 400 samples from the full dataset
    total_samples = 400  # We want 400 total samples
    indices = torch.randperm(len(full_dataset))[:total_samples]
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Now split the subset into train and test
    train_size = int(0.75 * total_samples)  # 300 samples for training
    test_size = total_samples - train_size  # 100 samples for testing
    
    print(f"Splitting subset: total={total_samples}, train={train_size}, test={test_size}")
    
    train_dataset, test_dataset = random_split(
        subset_dataset, 
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

# # Example usage:
# if __name__ == "__main__":
#     try:
#         train_dataloader, test_dataloader = get_dataloaders()
        
#         # Print some information about the datasets
#         print(f"Number of training batches: {len(train_dataloader)}")
#         print(f"Number of test batches: {len(test_dataloader)}")
        
#         # Print first batch of training data
#         for batch in train_dataloader:
#             print("\nTraining batch example:")
#             print(f"Audio shape: {batch['audio'].shape}")
#             print(f"MOS scores: {batch['mos_score']}")
#             print(f"Dataset: {batch['dataset']}")
#             print(f"Features shape: {batch['features'].shape}")
#             break
        
#         # Print first batch of test data
#         for batch in test_dataloader:
#             print("\nTest batch example:")
#             print(f"Audio shape: {batch['audio'].shape}")
#             print(f"MOS scores: {batch['mos_score']}")
#             print(f"Dataset: {batch['dataset']}")
#             print(f"Features shape: {batch['features'].shape}")
#             break
#     except Exception as e:
#         print(f"Error during dataloader creation or iteration: {str(e)}")  