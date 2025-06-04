import numpy as np
import pandas as pd
import librosa
import webrtcvad
from vosk import Model, KaldiRecognizer
from scipy import signal
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json
warnings.filterwarnings('ignore')

from datasets import load_dataset

class AudioFeatureExtractor:
    """Extract comprehensive audio features for MOS prediction"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  
        
        print("Loading Vosk model...")
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print("Downloading Vosk model...")
            import urllib.request
            import zipfile
            url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            urllib.request.urlretrieve(url, "model.zip")
            with zipfile.ZipFile("model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("model.zip")
        
        self.vosk_model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, self.sr)
    
    def extract_snr(self, audio):
        """Extract Signal-to-Noise Ratio using spectral gating"""
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        noise_floor = np.percentile(audio_db, 10)
        signal_level = np.percentile(audio_db, 90)
        snr = signal_level - noise_floor
        return max(snr, 0)
      
    def vad_based_snr(self, audio, sr=16000):
        """Calculate SNR using Voice Activity Detection"""
        vad = webrtcvad.Vad(3)  
        audio_int16 = (audio * 32767).astype(np.int16)
        frame_length = 480 
        speech_frames = []
        noise_frames = []
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            frame_bytes = frame.tobytes()
            
            if vad.is_speech(frame_bytes, sr):
                speech_frames.extend(frame)
            else:
                noise_frames.extend(frame)
        
        if len(speech_frames) > 0 and len(noise_frames) > 0:
            speech_power = np.mean(np.array(speech_frames)**2)
            noise_power = np.mean(np.array(noise_frames)**2)
            snr_db = 10 * np.log10(speech_power / noise_power)
            return snr_db
        else:
            return 0
    
    def extract_silence_percentage(self, audio):
        """Extract percentage of silence using WebRTC VAD"""
        audio_int16 = (audio * 32767).astype(np.int16)
        frame_length = 480
        frames = []
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length].tobytes()
            is_speech = self.vad.is_speech(frame, self.sr)
            frames.append(is_speech)
        
        if len(frames) == 0:
            return 0.5  
        
        silence_ratio = 1 - (sum(frames) / len(frames))
        return silence_ratio
    
    def extract_speaking_rate(self, audio, sr=16000):
        """Extract speaking rate using Vosk ASR"""
        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            chunk_size = 4000  
            transcript = ""
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only process full chunks
                    if self.recognizer.AcceptWaveform(chunk.tobytes()):
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
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        clipping_ratio = np.mean(np.abs(audio_norm) >= 0.99)
        return clipping_ratio
    
    def extract_reverberation_proxy(self, audio):
        """Extract reverberation proxy using spectral decay"""
        # Compute power spectral density
        f, psd = signal.welch(audio, fs=self.sr, nperseg=1024)        
        psd_db = 10 * np.log10(psd + 1e-8)        
        slope = np.polyfit(f[1:], psd_db[1:], 1)[0]        
        envelope = np.abs(signal.hilbert(audio))
        envelope_smooth = signal.savgol_filter(envelope, 51, 3)        
        env_kurtosis = kurtosis(envelope_smooth)
        
        return abs(slope), env_kurtosis
    
    def extract_spectral_features(self, audio):
        """Extract MFCC and other spectral features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        features = {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_delta_mean': delta_mean,
            'mfcc_delta2_mean': delta2_mean,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_flux_mean': np.mean(spectral_flux),
            'spectral_flux_std': np.std(spectral_flux)
        }
        
        return features
    
    def extract_all_features(self, audio,sr):
        """Extract all features for a single audio sample"""
        features = {}
        
        # Core quality cues
        features['snr'] = self.extract_snr(audio)
        features['silence_percentage'] = self.extract_silence_percentage(audio)
        
        speaking_rate, word_count, duration = self.extract_speaking_rate(audio,sr)
        features['speaking_rate'] = speaking_rate
        features['word_count'] = word_count
        features['duration'] = duration
        
        features['clipping_ratio'] = self.extract_clipping(audio)
        
        slope, env_kurtosis = self.extract_reverberation_proxy(audio)
        features['reverberation_slope'] = slope
        features['envelope_kurtosis'] = env_kurtosis
        
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        return features


