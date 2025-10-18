"""
Signal preprocessing and feature extraction
"""

import numpy as np
from scipy import signal
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SignalPreprocessor:
    """Preprocess seismic signals"""
    
    def __init__(self, sample_rate: int, lowcut: float, highcut: float):
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        
        # Design bandpass filter
        self.sos = signal.butter(
            4,
            [lowcut, highcut],
            btype='band',
            fs=sample_rate,
            output='sos'
        )
        
        logger.info(f"Initialized preprocessor: {lowcut}-{highcut} Hz bandpass")
    
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove noise"""
        if len(data) < 10:
            return data
        return signal.sosfilt(self.sos, data)
    
    def detrend(self, data: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal"""
        return signal.detrend(data)
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-10:
            return (data - mean) / std
        return data - mean


class FeatureExtractor:
    """Extract features from seismic waveforms"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def extract_all_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive feature set"""
        
        features = {}
        
        # Time domain features
        features.update(self._time_domain_features(waveform))
        
        # Frequency domain features
        features.update(self._frequency_domain_features(waveform))
        
        # Statistical features
        features.update(self._statistical_features(waveform))
        
        return features
    
    def _time_domain_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract time domain features"""
        
        # Peak ground acceleration
        pga = np.max(np.abs(waveform))
        
        # Mean amplitude
        mean_amplitude = np.mean(np.abs(waveform))
        
        # Standard deviation
        std_amplitude = np.std(waveform)
        
        # Energy
        energy = np.sum(waveform ** 2)
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(waveform)) != 0)
        
        # Duration
        duration = len(waveform) / self.sample_rate
        
        return {
            'pga': pga,
            'mean_amplitude': mean_amplitude,
            'std_amplitude': std_amplitude,
            'energy': energy,
            'zero_crossings': zero_crossings,
            'duration': duration
        }
    
    def _frequency_domain_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        
        # FFT
        fft = np.fft.fft(waveform)
        freqs = np.fft.fftfreq(len(waveform), 1/self.sample_rate)
        
        # Only positive frequencies
        positive_idx = freqs > 0
        magnitude = np.abs(fft[positive_idx])
        positive_freqs = freqs[positive_idx]
        
        if np.sum(magnitude) > 0:
            # Mean frequency
            mean_freq = np.sum(positive_freqs * magnitude) / np.sum(magnitude)
            
            # Dominant frequency
            dominant_freq = positive_freqs[np.argmax(magnitude)]
        else:
            mean_freq = 0.0
            dominant_freq = 0.0
        
        return {
            'mean_freq': mean_freq,
            'dominant_freq': dominant_freq
        }
    
    def _statistical_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        
        mean = np.mean(waveform)
        std = np.std(waveform)
        
        # Kurtosis (measure of peakedness)
        if std > 1e-10:
            kurtosis = np.mean(((waveform - mean) / std) ** 4) - 3
        else:
            kurtosis = 0.0
        
        # Skewness (measure of asymmetry)
        if std > 1e-10:
            skewness = np.mean(((waveform - mean) / std) ** 3)
        else:
            skewness = 0.0
        
        return {
            'kurtosis': kurtosis,
            'skewness': skewness
        }
    
    def features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to ordered array for ML model"""
        ordered_keys = [
            'pga', 'mean_amplitude', 'std_amplitude', 'energy',
            'kurtosis', 'skewness', 'mean_freq', 'dominant_freq',
            'zero_crossings', 'duration'
        ]
        
        return np.array([features.get(key, 0.0) for key in ordered_keys])


class STALTACalculator:
    """Calculate Short-Term Average / Long-Term Average ratio"""
    
    def __init__(self, sta_window: float, lta_window: float, sample_rate: int):
        self.sta_samples = int(sta_window * sample_rate)
        self.lta_samples = int(lta_window * sample_rate)
        self.sample_rate = sample_rate
        
        logger.info(f"STA/LTA: {sta_window}s / {lta_window}s")
    
    def calculate(self, data: np.ndarray) -> tuple:
        """
        Calculate STA/LTA ratio
        
        Returns:
            (sta, lta, ratio)
        """
        if len(data) < self.lta_samples:
            return 0.0, 0.0, 0.0
        
        abs_data = np.abs(data)
        
        # Short-term average (recent activity)
        sta = np.mean(abs_data[-self.sta_samples:])
        
        # Long-term average (background level)
        lta = np.mean(abs_data)
        
        # Ratio
        if lta > 1e-10:
            ratio = sta / lta
        else:
            ratio = 0.0
        
        return sta, lta, ratio