"""
Earthquake detection with ML and rule-based methods
"""

import numpy as np
import joblib
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

from preprocessing import SignalPreprocessor, FeatureExtractor, STALTACalculator
from location_estimator import LocationEstimator

logger = logging.getLogger(__name__)


class EarthquakeDetector:
    """
    Integrated earthquake detection system
    Combines STA/LTA triggering with ML validation and location estimation
    """
    
    def __init__(self, config: dict, model_path: Optional[str] = None, location_config: dict = None):
        self.config = config
        
        # Initialize components
        self.sample_rate = config['sample_rate']
        
        self.preprocessor = SignalPreprocessor(
            sample_rate=self.sample_rate,
            lowcut=config['bandpass_low'],
            highcut=config['bandpass_high']
        )
        
        self.feature_extractor = FeatureExtractor(self.sample_rate)
        
        self.sta_lta = STALTACalculator(
            sta_window=config['sta_window'],
            lta_window=config['lta_window'],
            sample_rate=self.sample_rate
        )
        
        # Initialize location estimator
        self.location_estimator = None
        if location_config and location_config.get('enable_location', False):
            station_location = (
                location_config['station_latitude'],
                location_config['station_longitude']
            )
            self.location_estimator = LocationEstimator(station_location)
            logger.info("Location estimation enabled")
        
        # Data buffer (for LTA calculation)
        lta_samples = int(config['lta_window'] * self.sample_rate)
        self.buffer = deque(maxlen=lta_samples)
        
        # Detection state
        self.is_triggered = False
        self.event_start_time = None
        self.event_data = []
        self.detected_events = []
        
        # Thresholds
        self.trigger_ratio = config['trigger_ratio']
        self.detrigger_ratio = config['detrigger_ratio']
        self.ml_threshold = config['ml_confidence_threshold']
        
        # Load ML model
        self.ml_model = None
        self.ml_scaler = None
        if model_path:
            self.load_ml_model(model_path)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'triggers': 0,
            'confirmed_events': 0,
            'false_positives': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Earthquake detector initialized")
    
    def load_ml_model(self, model_path: str):
        """Load trained ML model"""
        try:
            model_package = joblib.load(model_path)
            self.ml_model = model_package['model']
            self.ml_scaler = model_package['scaler']
            logger.info(f"ML model loaded: {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found: {model_path}")
            logger.warning("Using rule-based detection only")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def process_sample(self, sample: float, timestamp: datetime) -> Dict:
        """
        Process a single sensor sample
        
        Returns:
            Dictionary with detection results
        """
        self.stats['total_samples'] += 1
        self.buffer.append(sample)
        
        # Calculate STA/LTA
        sta, lta, ratio = self.sta_lta.calculate(np.array(self.buffer))
        
        result = {
            'timestamp': timestamp.isoformat(),
            'sample': sample,
            'sta': sta,
            'lta': lta,
            'ratio': ratio,
            'triggered': False,
            'event_detected': False,
            'event': None
        }
        
        # State machine for event detection
        if not self.is_triggered:
            # Check for trigger
            if ratio > self.trigger_ratio:
                self._trigger_event(sample, timestamp, ratio)
                result['triggered'] = True
        
        else:
            # Event is ongoing
            self.event_data.append(sample)
            
            # Check for detrigger
            if ratio < self.detrigger_ratio:
                event = self._process_event()
                
                if event:
                    result['event_detected'] = True
                    result['event'] = event
        
        return result
    
    def _trigger_event(self, sample: float, timestamp: datetime, ratio: float):
        """Trigger a new event"""
        self.is_triggered = True
        self.event_start_time = timestamp
        self.event_data = [sample]
        self.stats['triggers'] += 1
        
        logger.info(f"🔴 TRIGGER: STA/LTA = {ratio:.2f}")
    
    def _process_event(self) -> Optional[Dict]:
        """Process and validate a triggered event"""
        self.is_triggered = False
        
        # Convert to numpy array
        event_waveform = np.array(self.event_data)
        
        # Preprocess
        filtered = self.preprocessor.bandpass_filter(event_waveform)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(filtered)
        
        # ML validation
        is_earthquake, confidence = self._ml_validate(features)
        
        event = None
        
        if is_earthquake and confidence > self.ml_threshold:
            # Confirmed earthquake
            magnitude = self._estimate_magnitude(features)
            
            event = {
                'timestamp': self.event_start_time.isoformat(),
                'duration': features['duration'],
                'magnitude': magnitude,
                'confidence': confidence,
                'pga': features['pga'],
                'mean_freq': features['mean_freq'],
                'dominant_freq': features['dominant_freq'],
                'energy': features['energy']
            }
            
            # Estimate location if enabled
            if self.location_estimator:
                location_info = self.location_estimator.estimate_location(
                    features={'magnitude': magnitude, **features},
                    waveform=filtered,
                    sample_rate=self.sample_rate
                )
                event['location'] = location_info
            
            self.detected_events.append(event)
            self.stats['confirmed_events'] += 1
            
            self._log_earthquake(event)
        
        else:
            # False positive
            self.stats['false_positives'] += 1
            logger.info(f"⚪ False alarm (confidence: {confidence:.2%})")
        
        # Reset event data
        self.event_data = []
        self.event_start_time = None
        
        return event
    
    def _ml_validate(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Validate event using ML model or fallback rules"""
        
        if self.ml_model is not None:
            try:
                # Convert features to array
                feature_array = self.feature_extractor.features_to_array(features)
                feature_array = feature_array.reshape(1, -1)
                
                # Scale and predict
                feature_scaled = self.ml_scaler.transform(feature_array)
                prediction = self.ml_model.predict(feature_scaled)[0]
                confidence = self.ml_model.predict_proba(feature_scaled)[0, 1]
                
                return bool(prediction), float(confidence)
            
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                # Fall through to rule-based
        
        # Rule-based fallback
        return self._rule_based_validate(features)
    
    def _rule_based_validate(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Simple rule-based validation"""
        score = 0.0
        
        # Check PGA range
        if 0.1 < features['pga'] < 100:
            score += 0.25
        
        # Check duration
        if 5 < features['duration'] < 60:
            score += 0.25
        
        # Check frequency
        if 0.5 < features['mean_freq'] < 10:
            score += 0.25
        
        # Check energy
        if features['energy'] > 1.0:
            score += 0.15
        
        # Check kurtosis
        if -2 < features['kurtosis'] < 5:
            score += 0.10
        
        is_earthquake = score > 0.6
        return is_earthquake, score
    
    def _estimate_magnitude(self, features: Dict[str, float]) -> float:
        """Estimate local magnitude from PGA"""
        pga = features['pga']
        magnitude = np.log10(pga * 1000) + 2.0
        return np.clip(magnitude, 1.0, 9.0)
    
    def _log_earthquake(self, event: Dict):
        """Log detected earthquake"""
        logger.info("=" * 70)
        logger.info("🚨 EARTHQUAKE DETECTED")
        logger.info("=" * 70)
        logger.info(f"Time:       {event['timestamp']}")
        logger.info(f"Magnitude:  {event['magnitude']:.1f}")
        logger.info(f"Duration:   {event['duration']:.1f}s")
        logger.info(f"Confidence: {event['confidence']:.1%}")
        logger.info(f"PGA:        {event['pga']:.3f} m/s²")
        
        # Log location if available
        if 'location' in event:
            loc = event['location']
            logger.info(f"Distance:   {loc['distance_km']:.1f} km")
            logger.info(f"Location:   {loc['epicenter_lat']:.4f}°, {loc['epicenter_lon']:.4f}°")
            logger.info(f"Depth:      {loc['depth_km']:.1f} km")
            logger.info(f"Probability: {loc['probability']:.1%}")
            if loc.get('nearest_zone'):
                logger.info(f"Near:       {loc['nearest_zone']}")
        
        logger.info("=" * 70)
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        detection_rate = 0.0
        if self.stats['triggers'] > 0:
            detection_rate = self.stats['confirmed_events'] / self.stats['triggers']
        
        return {
            'runtime_seconds': runtime,
            'total_samples': self.stats['total_samples'],
            'triggers': self.stats['triggers'],
            'confirmed_events': self.stats['confirmed_events'],
            'false_positives': self.stats['false_positives'],
            'detection_rate': detection_rate,
            'samples_per_second': self.stats['total_samples'] / max(1, runtime)
        }
    
    def get_recent_events(self, limit: int = 10) -> list:
        """Get most recent detected events"""
        return self.detected_events[-limit:]