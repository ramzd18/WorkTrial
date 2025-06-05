import numpy as np
import pandas as pd
import librosa
import webrtcvad
from vosk import Model, KaldiRecognizer
from scipy import signal
from scipy.stats import kurtosis
import warnings
import os
import json
import logging
import torch
from torchmetrics.functional.audio.srmr import (
    speech_reverberation_modulation_energy_ratio,
)
import tempfile
import soundfile as sf
from scipy.special import logsumexp

# Configure logging and warnings
logging.basicConfig(level=logging.ERROR)
for logger_name in ['librosa', 'vosk', 'kaldi', 'numpy', 'scipy']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

class AudioFeatureExtractor:
    """Extract comprehensive audio features for MOS prediction"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  
        
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            import urllib.request
            import zipfile
            url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            urllib.request.urlretrieve(url, "model.zip")
            with zipfile.ZipFile("model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("model.zip")
        
        self.vosk_model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, self.sr)
    
    def vad_based_snr(self, audio, sr=16000):
        """Calculate SNR using Voice Activity Detection"""
        vad = webrtcvad.Vad(3)  
        audio_int16 = (audio * 32767).to(torch.int16)
        frame_length = 480 
        speech_frames = []
        noise_frames = []
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            frame_bytes = frame.numpy().tobytes()
            
            if vad.is_speech(frame_bytes, sr):
                speech_frames.extend(frame.numpy())
            else:
                noise_frames.extend(frame.numpy())
        
        if len(speech_frames) > 0 and len(noise_frames) > 0:
            speech_power = torch.mean(torch.tensor(speech_frames,dtype=torch.float32)**2)
            noise_power = torch.mean(torch.tensor(noise_frames,dtype=torch.float32)**2)
            snr_db = 10 * torch.log10(speech_power / noise_power)
            return snr_db.item()
        else:
            return 0
    
    def extract_silence_percentage(self, audio):
        """Extract percentage of silence using WebRTC VAD"""
        audio_int16 = (audio * 32767).to(torch.int16)
        frame_length = 480
        frames = []
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length].numpy().tobytes()
            is_speech = self.vad.is_speech(frame, self.sr)
            frames.append(is_speech)
        
        if len(frames) == 0:
            return 0.5  
        
        silence_ratio = 1 - (sum(frames) / len(frames))
        return silence_ratio
    
    def extract_speaking_rate(self, audio, sr=16000):
        """Extract speaking rate using Vosk ASR"""
        try:
            audio_int16 = (audio * 32767).to(torch.int16)
            chunk_size = 4000  
            transcript = ""
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only process full chunks
                    if self.recognizer.AcceptWaveform(chunk.numpy().tobytes()):
                        result = self.recognizer.Result()
                        if isinstance(result, dict):
                            transcript += result.get("text", "")
                        elif isinstance(result, str):
                            try:
                                result_dict = json.loads(result)
                                transcript += result_dict.get("text", "")
                            except json.JSONDecodeError:
                                continue
            
            final_result = self.recognizer.FinalResult()
            if isinstance(final_result, dict):
                transcript += final_result.get("text", "")
            elif isinstance(final_result, str):
                try:
                    final_dict = json.loads(final_result)
                    transcript += final_dict.get("text", "")
                except json.JSONDecodeError:
                    pass
            
            word_count = len(transcript.split())
            duration = len(audio) / sr
            
            if duration > 0:
                speaking_rate = word_count / duration
            else:
                speaking_rate = 0
                
            return speaking_rate, word_count, duration
            
        except Exception as e:
            print(f"Error in speaking rate extraction: {e}")
            return 0, 0, len(audio) / sr
    
    def extract_clipping(self, audio):
        """Detect audio clipping"""
        audio_norm = audio / (torch.max(torch.abs(audio)) + 1e-8)
        clipping_ratio = torch.mean((torch.abs(audio_norm) >= 0.99).float())
        return clipping_ratio.item()
    
    def extract_reverberation_proxy(self, audio):
        """Extract reverberation proxy using spectral decay"""
        audio_np = audio.numpy()
        f, psd = signal.welch(audio_np, fs=self.sr, nperseg=1024)        
        psd_db = 10 * np.log10(psd + 1e-8)
        
        if len(psd_db.shape) > 1:
            psd_db = psd_db.squeeze()
        
        slope = np.polyfit(f[1:], psd_db[1:], 1)[0]        
        envelope = torch.abs(torch.tensor(signal.hilbert(audio_np)))
        envelope_smooth = torch.tensor(signal.savgol_filter(envelope.numpy(), 51, 3))        
        env_kurtosis = kurtosis(envelope_smooth.numpy())
        
        return abs(slope), env_kurtosis
    
    def extract_spectral_features(self, audio):
        """Extract MFCC and other spectral features with consistent dimensions"""
        if len(audio.shape) > 1:
            audio = audio.squeeze()
            
        audio_np = audio.numpy()
        
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        
        mfccs = librosa.feature.mfcc(
            y=audio_np, 
            sr=self.sr, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mfccs_tensor = torch.tensor(mfccs)
        
        mfcc_mean = torch.mean(mfccs_tensor, dim=1)  
        mfcc_std = torch.std(mfccs_tensor, dim=1)    
        
        mfcc_delta = torch.tensor(librosa.feature.delta(mfccs))
        mfcc_delta2 = torch.tensor(librosa.feature.delta(mfccs, order=2))
        
        delta_mean = torch.mean(mfcc_delta, dim=1)    
        delta2_mean = torch.mean(mfcc_delta2, dim=1)  
        
        spectral_centroids = torch.tensor(librosa.feature.spectral_centroid(
            y=audio_np, 
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length
        )[0])
        
        spectral_rolloff = torch.tensor(librosa.feature.spectral_rolloff(
            y=audio_np, 
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length
        )[0])
        
        spectral_flux = torch.tensor(librosa.onset.onset_strength(
            y=audio_np, 
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length
        ))
        
        features = {
            'mfcc_mean': mfcc_mean,                    # Shape: (n_mfcc,)
            'mfcc_std': mfcc_std,                      # Shape: (n_mfcc,)
            'mfcc_delta_mean': delta_mean,             # Shape: (n_mfcc,)
            'mfcc_delta2_mean': delta2_mean,           # Shape: (n_mfcc,)
            'spectral_centroid_mean': torch.mean(spectral_centroids),    # Scalar
            'spectral_centroid_std': torch.std(spectral_centroids),      # Scalar
            'spectral_rolloff_mean': torch.mean(spectral_rolloff),       # Scalar
            'spectral_rolloff_std': torch.std(spectral_rolloff),         # Scalar
            'spectral_flux_mean': torch.mean(spectral_flux),            # Scalar
            'spectral_flux_std': torch.std(spectral_flux)               # Scalar
        }
        
        return features
    
    def extract_all_features(self, audio, sr):
        """Extract all features for a single audio sample"""
        features = {}
        
        print("Audio Shape", audio.shape)
        audio = torch.tensor(audio) if not isinstance(audio, torch.Tensor) else audio
        features['snr'] = self.vad_based_snr(audio)
        features['silence_percentage'] = self.extract_silence_percentage(audio)
        speaking_rate, word_count, duration = self.extract_speaking_rate(audio, sr)
        features['speaking_rate'] = speaking_rate
        features['word_count'] = word_count
        features['duration'] = duration
        features['clipping_ratio'] = self.extract_clipping(audio)
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        print("Spectral Features")
        return features

    def features_to_tensor(self, features):
        feature_values = []
        for value in features.values():
            # Flatten everything to a list of numbers, recursively if needed
            def flatten(val):
                if isinstance(val, torch.Tensor):
                    return flatten(val.detach().cpu().numpy())
                elif isinstance(val, np.ndarray):
                    return val.flatten().tolist()
                elif isinstance(val, (list, tuple)):
                    flat = []
                    for v in val:
                        flat.extend(flatten(v))
                    return flat
                elif isinstance(val, (float, int, np.floating, np.integer)):
                    return [float(val)]
                else:
                    try:
                        return [float(val)]
                    except Exception:
                        return []
            feature_values.extend(flatten(value))
        return torch.tensor(feature_values, dtype=torch.float32)


