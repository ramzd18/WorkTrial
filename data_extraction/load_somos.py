import os
import numpy as np
import librosa
from tqdm import tqdm

DATA_FOLDER = "/Volumes/RAM_DRIVE/somos/training_files/split1/full"
INITIAL_FOLDER = "/Volumes/RAM_DRIVE/somos/"
TRAIN_TXT_PATH = os.path.join(DATA_FOLDER, "train_mos_list.txt")
TEST_TXT_PATH = os.path.join(DATA_FOLDER, "test_mos_list.txt")
VAL_TXT_PATH = os.path.join(DATA_FOLDER, "valid_mos_list.txt")
WAV_FOLDER = os.path.join(INITIAL_FOLDER, "audios")

def load_mos_scores():
    """
    Load MOS scores from all available files (train, val, test)
    
    Returns:
        dict: Dictionary mapping filenames to their MOS scores
    """
    mos_scores = {}
    txt_paths = {
        'train': TRAIN_TXT_PATH,
        'val': VAL_TXT_PATH,
        'test': TEST_TXT_PATH
    }
    
    for split_name, txt_path in txt_paths.items():
        try:
            with open(txt_path, 'r') as f:
                # Skip the header row which contains 'filename,mean'
                next(f)
                for line in f:
                    try:
                        filename, score = line.strip().split(',')
                        if filename in mos_scores:
                            print(f"Warning: Duplicate score found for {filename} in {split_name}")
                        mos_scores[filename] = float(score)
                    except ValueError as e:
                        print(f"Warning: Skipping malformed line in {split_name}: {line.strip()}")
                        continue
            print(f"Loaded {len(mos_scores)} scores from {split_name} set")
        except FileNotFoundError:
            print(f"Warning: {split_name} file not found at {txt_path}")
        except Exception as e:
            print(f"Error loading {split_name} scores: {e}")
    
    return mos_scores

def load_somos_dataset(sample_size=None, target_sr=16000):
    """
    Load BVCC dataset with specified sample size
    
    Args:
        sample_size (int, optional): Number of samples to load. If None, loads all samples.
        target_sr (int): Target sample rate for audio resampling
    
    Returns:
        list: List of tuples (audio_array, mos_score)
    """
    print("Loading BVCC dataset...")
    
    # First, get all WAV files from the folder
    wav_files = [f for f in os.listdir(WAV_FOLDER) if f.endswith('.wav')]
    if sample_size:
        wav_files = wav_files[:sample_size]
    
    # Load MOS scores from all available files
    mos_scores = load_mos_scores()
    
    audio_mos_pairs = []
    missing_scores = []
    
    print("Processing audio files...")
    for wav_file in tqdm(wav_files):
        try:
            # Get MOS score for this file
            if wav_file not in mos_scores:
                missing_scores.append(wav_file)
                continue
                
            mos_score = mos_scores[wav_file]
            
            # Construct full path to wav file
            wav_path = os.path.join(WAV_FOLDER, wav_file)
            
            # Load and resample audio
            audio, sr = librosa.load(wav_path, sr=target_sr)
            
            audio_mos_pairs.append((audio, mos_score))
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    if missing_scores:
        print(f"\nWarning: {len(missing_scores)} files were missing MOS scores")
        print("First 5 missing files:", missing_scores[:5])
    
    return audio_mos_pairs

def retrieve_dataset(sample_size=None, target_sr=16000):
    sample_size = 40000
    audio_mos_pairs = load_somos_dataset(sample_size=sample_size)
    return audio_mos_pairs









