import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class SimpleNeuralMOSPredictor:
    # Simple neural network for MOS prediction
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        self.feature_names = None
    
    def features_to_array(self, features_list):
        # Convert list of feature dictionaries to numpy array
        if not features_list:
            return np.array([])
        
        if self.feature_names is None:
            self.feature_names = []
            
            scalar_features = ['snr', 'silence_percentage', 'speaking_rate', 'word_count', 
                             'duration', 'clipping_ratio', 'reverberation_slope', 
                             'envelope_kurtosis', 'spectral_centroid_mean', 'spectral_centroid_std',
                             'spectral_rolloff_mean', 'spectral_rolloff_std', 
                             'spectral_flux_mean', 'spectral_flux_std']
            
            self.feature_names.extend(scalar_features)
            
            for i in range(13):
                self.feature_names.extend([f'mfcc_mean_{i}', f'mfcc_std_{i}', 
                                         f'mfcc_delta_mean_{i}', f'mfcc_delta2_mean_{i}'])
        
        feature_arrays = []
        for features in features_list:
            feature_vector = []
            
            scalar_features = ['snr', 'silence_percentage', 'speaking_rate', 'word_count', 
                             'duration', 'clipping_ratio', 'reverberation_slope', 
                             'envelope_kurtosis', 'spectral_centroid_mean', 'spectral_centroid_std',
                             'spectral_rolloff_mean', 'spectral_rolloff_std', 
                             'spectral_flux_mean', 'spectral_flux_std'
                             ]
            
            for feat_name in scalar_features:
                feature_vector.append(features.get(feat_name, 0))
            
            for i in range(13):
                feature_vector.append(features.get('mfcc_mean', np.zeros(13))[i])
                feature_vector.append(features.get('mfcc_std', np.zeros(13))[i])
                feature_vector.append(features.get('mfcc_delta_mean', np.zeros(13))[i])
                feature_vector.append(features.get('mfcc_delta2_mean', np.zeros(13))[i])
            
            feature_arrays.append(feature_vector)
        
        return np.array(feature_arrays)
    
    def train(self, features_list, mos_scores):
        # Train the neural network
        X = self.features_to_array(features_list)
        y = np.array(mos_scores)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, features_list):
        # Predict MOS scores
        X = self.features_to_array(features_list)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)        
        return np.clip(predictions, 1.0, 5.0)