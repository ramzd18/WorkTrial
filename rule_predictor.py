class RulesBasedMOSPredictor:
    """Rules-based MOS prediction using heuristics"""
    
    def __init__(self):
        # Empirically determined thresholds
        self.snr_thresholds = {'excellent': 25, 'good': 15, 'fair': 5, 'poor': 0}
        self.silence_thresholds = {'excellent': 0.1, 'good': 0.2, 'fair': 0.4, 'poor': 1.0}
        self.speaking_rate_range = {'min': 1.0, 'max': 4.0}  # words per second
        self.clipping_threshold = 0.01  # 1% clipping is bad
        
    def predict_single(self, features):
        """Predict MOS for a single sample using rules"""
        score = 5.0  # Start with perfect score
        
        # SNR penalty
        snr = features.get('snr', 0)
        if snr < self.snr_thresholds['poor']:
            score -= 2.0
        elif snr < self.snr_thresholds['fair']:
            score -= 1.5
        elif snr < self.snr_thresholds['good']:
            score -= 0.8
        elif snr < self.snr_thresholds['excellent']:
            score -= 0.3
        
        # Silence penalty
        silence = features.get('silence_percentage', 0)
        if silence > self.silence_thresholds['poor']:
            score -= 2.0
        elif silence > self.silence_thresholds['fair']:
            score -= 1.0
        elif silence > self.silence_thresholds['good']:
            score -= 0.5
        
        # Speaking rate penalty
        speaking_rate = features.get('speaking_rate', 2.0)
        if speaking_rate < self.speaking_rate_range['min'] or speaking_rate > self.speaking_rate_range['max']:
            score -= 0.8
        
        # Clipping penalty
        clipping = features.get('clipping_ratio', 0)
        if clipping > self.clipping_threshold:
            score -= 1.5
        
        # env_kurtosis = features.get('envelope_kurtosis', 0)
        # if env_kurtosis > 10:
        #     score -= 1.0
        # elif env_kurtosis > 5:
        #     score -= 0.5
        
        # Spectral quality check
        spectral_centroid_mean = features.get('spectral_centroid_mean', 1000)
        if spectral_centroid_mean < 500 or spectral_centroid_mean > 4000:
            score -= 0.5
        # srmr = features.get('srmr', 0)
        # if srmr < 0.5:
        #     score -= 1.0
        # elif srmr < 0.7:
        #     score -= 0.5
        
        # Clip to valid MOS range
        return max(1.0, min(5.0, score))
    
    def predict(self, features_list):
        """Predict MOS for multiple samples"""
        return [self.predict_single(features) for features in features_list]