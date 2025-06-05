from load_somos import retrieve_dataset as retrieve_somos_dataset
from load_bvcc import retrieve_dataset as retrieve_bvcc_dataset
import os
import json
import soundfile as sf
import glob
import shutil
from datasets import load_dataset

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