# Rules-based MOS prediction using heuristics
class RulesBasedMOSPredictor:
    
    def __init__(self):
        self.snr_thresholds = {'excellent': 25, 'good': 15, 'fair': 5, 'poor': 0}
        self.silence_thresholds = {'excellent': 0.1, 'good': 0.2, 'fair': 0.4, 'poor': 1.0}
        self.speaking_rate_range = {'min': 1.0, 'max': 4.0}  
        self.clipping_threshold = 0.01  

    # Predict MOS for a single sample using rules
    def predict_single(self, features):
        score = 5.0 
        
        snr = features.get('snr', 0)
        if snr < self.snr_thresholds['poor']:
            score -= 2.0
        elif snr < self.snr_thresholds['fair']:
            score -= 1.5
        elif snr < self.snr_thresholds['good']:
            score -= 0.8
        elif snr < self.snr_thresholds['excellent']:
            score -= 0.3
        
        silence = features.get('silence_percentage', 0)
        if silence > self.silence_thresholds['poor']:
            score -= 2.0
        elif silence > self.silence_thresholds['fair']:
            score -= 1.0
        elif silence > self.silence_thresholds['good']:
            score -= 0.5
        
        speaking_rate = features.get('speaking_rate', 2.0)
        if speaking_rate < self.speaking_rate_range['min'] or speaking_rate > self.speaking_rate_range['max']:
            score -= 0.8
        
        clipping = features.get('clipping_ratio', 0)
        if clipping > self.clipping_threshold:
            score -= 1.5
        
        spectral_centroid_mean = features.get('spectral_centroid_mean', 1000)
        if spectral_centroid_mean < 500 or spectral_centroid_mean > 4000:
            score -= 0.5   
        return max(1.0, min(5.0, score))
    
    # Predict MOS for multiple samples
    def predict(self, features_list):
        return [self.predict_single(features) for features in features_list]