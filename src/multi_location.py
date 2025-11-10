"""
Multi-Location Earthquake Prediction
Predicts earthquake probability across multiple geographic locations
"""

import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MultiLocationPredictor:
    """
    Predict earthquake probability across multiple locations
    Uses historical patterns, stress accumulation, and regional analysis
    """
    
    def __init__(self):
        # Define monitoring regions
        self.monitoring_regions = self._initialize_regions()
        
        # Historical earthquake data (simplified model)
        self.historical_data = self._load_historical_patterns()
        
        # Stress accumulation model
        self.stress_model = StressAccumulationModel()
        
        logger.info(f"Multi-location predictor initialized with {len(self.monitoring_regions)} regions")
    
    def _initialize_regions(self) -> List[Dict]:
        """
        Define geographic regions to monitor
        Can be customized based on area of interest
        """
        regions = [
            # California regions
            {
                'id': 'southern_california',
                'name': 'Southern California',
                'center': (34.0522, -118.2437),
                'radius_km': 150,
                'fault_zones': ['San Andreas', 'San Jacinto', 'Elsinore'],
                'historical_rate': 2.5,  # events per year (M>3.0)
                'tectonic_setting': 'strike-slip'
            },
            {
                'id': 'bay_area',
                'name': 'San Francisco Bay Area',
                'center': (37.7749, -122.4194),
                'radius_km': 100,
                'fault_zones': ['San Andreas', 'Hayward', 'Calaveras'],
                'historical_rate': 43.0,
                'tectonic_setting': 'strike-slip'
            },
            
            # Pacific Northwest
            {
                'id': 'cascadia',
                'name': 'Cascadia Subduction Zone',
                'center': (44.0, -124.0),
                'radius_km': 300,
                'fault_zones': ['Cascadia Megathrust'],
                'historical_rate': 60.0,
                'tectonic_setting': 'subduction'
            },
            
            # Central US
            {
                'id': 'new_madrid',
                'name': 'New Madrid Seismic Zone',
                'center': (36.5, -89.5),
                'radius_km': 150,
                'fault_zones': ['New Madrid'],
                'historical_rate': 57.0,
                'tectonic_setting': 'intraplate'
            },
            
            # Alaska
            {
                'id': 'alaska_south',
                'name': 'Southern Alaska',
                'center': (61.2181, -149.9003),
                'radius_km': 200,
                'fault_zones': ['Aleutian Megathrust'],
                'historical_rate': 44.5,
                'tectonic_setting': 'subduction'
            },
            
            # Other regions (customize as needed)
            {
                'id': 'yellowstone',
                'name': 'Yellowstone Region',
                'center': (44.4280, -110.5885),
                'radius_km': 100,
                'fault_zones': ['Teton', 'Intermountain Seismic Belt'],
                'historical_rate': 91.0,
                'tectonic_setting': 'volcanic'
            },
            {
        'id': 'central_andes',
        'name': 'Central Andes Fault Belt',
        'center': (-20.0, -68.0),
        'radius_km': 400,
        'fault_zones': ['Central Andes Fault Belt'],
        'historical_rate': 56.99,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'nazca_chile',
        'name': 'Nazca Subduction Zone (Chile)',
        'center': (-30.0, -71.5),
        'radius_km': 450,
        'fault_zones': ['Nazca Plate Subduction'],
        'historical_rate': 8.0,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'himalayan_front',
        'name': 'Himalayan Frontal Thrust',
        'center': (28.0, 84.0),
        'radius_km': 400,
        'fault_zones': ['Main Himalayan Thrust'],
        'historical_rate': 81.1,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'karakoram_fault',
        'name': 'Karakoram Fault System',
        'center': (35.0, 77.0),
        'radius_km': 200,
        'fault_zones': ['Karakoram Fault'],
        'historical_rate': 58.0,
        'tectonic_setting': 'strike-slip'
    },
    {
        'id': 'zagros_belt',
        'name': 'Zagros Fold-Thrust Belt',
        'center': (30.5, 51.5),
        'radius_km': 300,
        'fault_zones': ['Zagros Belt'],
        'historical_rate': 4.6,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'north_anatolian',
        'name': 'North Anatolian Fault',
        'center': (40.8, 35.0),
        'radius_km': 350,
        'fault_zones': ['North Anatolian Fault'],
        'historical_rate': 7.3,
        'tectonic_setting': 'strike-slip'
    },
    {
        'id': 'hellenic_arc',
        'name': 'Hellenic Arc Subduction Zone',
        'center': (36.0, 24.0),
        'radius_km': 250,
        'fault_zones': ['Hellenic Arc'],
        'historical_rate': 1.9,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'east_african_rift',
        'name': 'East African Rift System',
        'center': (-3.0, 36.0),
        'radius_km': 500,
        'fault_zones': ['East African Rift'],
        'historical_rate': 5.0,
        'tectonic_setting': 'extensional'
    },
    {
        'id': 'sumatra_andaman',
        'name': 'Sumatra-Andaman Subduction Zone',
        'center': (3.0, 95.0),
        'radius_km': 600,
        'fault_zones': ['Sumatra Trench', 'Andaman Subduction Zone'],
        'historical_rate': 59.5,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'philippine_trench',
        'name': 'Philippine Trench',
        'center': (10.0, 127.0),
        'radius_km': 400,
        'fault_zones': ['Philippine Trench'],
        'historical_rate': 76.3,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'japan_trench',
        'name': 'Japan Trench',
        'center': (38.0, 143.0),
        'radius_km': 450,
        'fault_zones': ['Japan Trench'],
        'historical_rate': 89.97,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'ryukyu_trench',
        'name': 'Ryukyu Trench',
        'center': (26.0, 129.0),
        'radius_km': 300,
        'fault_zones': ['Ryukyu Subduction Zone'],
        'historical_rate': 86.6,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'taiwan_margin',
        'name': 'Taiwan Convergent Margin',
        'center': (23.5, 121.0),
        'radius_km': 250,
        'fault_zones': ['Longitudinal Valley Fault', 'Chelungpu Fault'],
        'historical_rate': 80.2,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'new_guinea_belt',
        'name': 'New Guinea Thrust Belt',
        'center': (-6.0, 145.0),
        'radius_km': 300,
        'fault_zones': ['Papua Thrust Belt'],
        'historical_rate': 55.4,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'tonga_kermadec',
        'name': 'Tonga-Kermadec Trench',
        'center': (-20.0, -175.0),
        'radius_km': 500,
        'fault_zones': ['Tonga Trench', 'Kermadec Trench'],
        'historical_rate': 92.1,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'new_zealand_alpine',
        'name': 'New Zealand Alpine Fault',
        'center': (-43.5, 170.0),
        'radius_km': 300,
        'fault_zones': ['Alpine Fault'],
        'historical_rate': 3.8,
        'tectonic_setting': 'strike-slip'
    },
    {
        'id': 'hindu_kush',
        'name': 'Hindu Kush Seismic Zone',
        'center': (36.5, 70.5),
        'radius_km': 250,
        'fault_zones': ['Hindu Kush Deep Focus Zone'],
        'historical_rate': 95.6,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'makran_subduction',
        'name': 'Makran Subduction Zone',
        'center': (25.0, 62.0),
        'radius_km': 350,
        'fault_zones': ['Makran Subduction Zone'],
        'historical_rate': 34.2,
        'tectonic_setting': 'subduction'
    },
    {
        'id': 'pamir_system',
        'name': 'Pamir Fault System',
        'center': (38.0, 73.0),
        'radius_km': 200,
        'fault_zones': ['Pamir Thrust Fault'],
        'historical_rate': 87.8,
        'tectonic_setting': 'thrust'
    },
    {
        'id': 'vanuatu_trench',
        'name': 'Vanuatu Subduction Zone',
        'center': (-16.0, 167.0),
        'radius_km': 400,
        'fault_zones': ['Vanuatu Trench'],
        'historical_rate': 9.3,
        'tectonic_setting': 'subduction'
    }
            
        ]
        
        return regions
    
    def _load_historical_patterns(self) -> Dict:
        """
        Load historical earthquake patterns
        In production, load from database or USGS catalog
        """
        return {
            'last_major_events': {
                'southern_california': datetime.now() - timedelta(days=145),
                'bay_area': datetime.now() - timedelta(days=320),
                'cascadia': datetime.now() - timedelta(days=9800),  # ~300 years
                'new_madrid': datetime.now() - timedelta(days=77000),  # ~1811-1812
            },
            'recent_swarms': [
                {'region': 'southern_california', 'date': datetime.now() - timedelta(days=15)},
                {'region': 'yellowstone', 'date': datetime.now() - timedelta(days=8)},
            ]
        }
    
    def predict_all_locations(self, detected_event: Dict = None) -> List[Dict]:
        """
        Predict earthquake probability for all monitored regions
        
        Args:
            detected_event: Recently detected earthquake (optional)
                           Used to update predictions based on triggering
        
        Returns:
            List of predictions for each region
        """
        predictions = []
        current_time = datetime.now()
        
        for region in self.monitoring_regions:
            prediction = self._predict_region(region, current_time, detected_event)
            predictions.append(prediction)
        
        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions
    
    def _predict_region(self, region: Dict, current_time: datetime, 
                       triggered_by: Dict = None) -> Dict:
        """
        Predict earthquake probability for a single region
        
        Uses multiple factors:
        1. Historical rate (Poisson process)
        2. Time since last major event (stress accumulation)
        3. Recent seismic activity (foreshocks/swarms)
        4. Triggering from nearby events
        5. Tectonic setting
        """
        
        # Base probability from historical rate
        base_prob = self._calculate_base_probability(region)
        
        # Stress accumulation factor
        stress_factor = self._calculate_stress_factor(region, current_time)
        
        # Recent activity factor
        activity_factor = self._calculate_activity_factor(region, current_time)
        
        # Triggering factor (if another earthquake just occurred)
        trigger_factor = self._calculate_trigger_factor(region, triggered_by)
        
        # Combine factors
        probability = base_prob * stress_factor * activity_factor * trigger_factor
        
        # Normalize to 0-1 range
        probability = min(probability, 1.0)
        
        # Estimate magnitude range
        magnitude_range = self._estimate_magnitude_range(region, probability)
        
        # Time window (next 24 hours, 7 days, 30 days)
        time_windows = self._calculate_time_windows(probability)
        
        prediction = {
            'region_id': region['id'],
            'region_name': region['name'],
            'center_lat': region['center'][0],
            'center_lon': region['center'][1],
            'radius_km': region['radius_km'],
            'probability': probability,
            'probability_24h': time_windows['24h'],
            'probability_7d': time_windows['7d'],
            'probability_30d': time_windows['30d'],
            'magnitude_range': magnitude_range,
            'confidence': self._calculate_confidence(region, probability),
            'risk_level': self._categorize_risk(probability),
            'fault_zones': region['fault_zones'],
            'tectonic_setting': region['tectonic_setting'],
            'factors': {
                'base_probability': base_prob,
                'stress_accumulation': stress_factor,
                'recent_activity': activity_factor,
                'triggering': trigger_factor
            },
            'last_updated': current_time.isoformat()
        }
        
        return prediction
    
    def _calculate_base_probability(self, region: Dict) -> float:
        """Calculate base probability from historical rate"""
        # Poisson probability: P(at least 1 event) = 1 - e^(-λ)
        # where λ = rate per time period
        
        rate_per_day = region['historical_rate'] / 365.0
        
        # Probability of at least one event in next 30 days
        prob_30d = 1 - np.exp(-rate_per_day * 30)
        
        return prob_30d
    
    def _calculate_stress_factor(self, region: Dict, current_time: datetime) -> float:
        """
        Calculate stress accumulation factor
        More time since last event = higher stress
        """
        last_event = self.historical_data['last_major_events'].get(
            region['id'],
            current_time - timedelta(days=365)
        )
        
        days_since = (current_time - last_event).days
        
        # Different recurrence intervals for different settings
        if region['tectonic_setting'] == 'subduction':
            recurrence_interval = 300 * 365  # 300 years
        elif region['tectonic_setting'] == 'strike-slip':
            recurrence_interval = 150 * 365  # 150 years
        elif region['tectonic_setting'] == 'volcanic':
            recurrence_interval = 50 * 365  # 50 years
        else:
            recurrence_interval = 100 * 365  # 100 years
        
        # Stress factor increases with time
        stress_ratio = days_since / recurrence_interval
        
        # Non-linear relationship
        if stress_ratio < 0.3:
            factor = 0.5
        elif stress_ratio < 0.7:
            factor = 1.0
        elif stress_ratio < 1.0:
            factor = 1.5
        else:
            factor = 2.0  # Overdue
        
        return factor
    
    def _calculate_activity_factor(self, region: Dict, current_time: datetime) -> float:
        """
        Calculate factor based on recent seismic activity
        Swarms and foreshocks increase probability
        """
        factor = 1.0
        
        # Check for recent swarms
        for swarm in self.historical_data['recent_swarms']:
            if swarm['region'] == region['id']:
                days_ago = (current_time - swarm['date']).days
                if days_ago < 30:
                    # Recent swarm increases probability
                    factor *= (1.5 - days_ago / 60)  # Decays over time
        
        return factor
    
    def _calculate_trigger_factor(self, region: Dict, triggered_by: Dict) -> float:
        """
        Calculate probability increase from nearby earthquake (triggering)
        """
        if triggered_by is None or 'location' not in triggered_by:
            return 1.0
        
        # Calculate distance to triggering event
        trigger_loc = triggered_by['location']
        distance = self._haversine_distance(
            region['center'][0], region['center'][1],
            trigger_loc['epicenter_lat'], trigger_loc['epicenter_lon']
        )
        
        # Triggering effect decreases with distance
        if distance < 50:
            # Very close - high triggering probability
            factor = 2.5
        elif distance < 100:
            factor = 2.0
        elif distance < 200:
            factor = 1.5
        elif distance < 500:
            factor = 1.2
        else:
            factor = 1.0
        
        # Magnitude also matters - larger events trigger more
        trigger_mag = triggered_by.get('magnitude', 0)
        if trigger_mag >= 6.0:
            factor *= 1.5
        elif trigger_mag >= 5.0:
            factor *= 1.3
        
        return factor
    
    def _estimate_magnitude_range(self, region: Dict, probability: float) -> Tuple[float, float]:
        """Estimate likely magnitude range"""
        
        # Different regions have different magnitude potentials
        if region['tectonic_setting'] == 'subduction':
            max_mag = 9.0
            min_mag = 4.0
        elif region['tectonic_setting'] == 'strike-slip':
            max_mag = 7.5
            min_mag = 3.0
        elif region['tectonic_setting'] == 'volcanic':
            max_mag = 6.0
            min_mag = 2.5
        else:
            max_mag = 7.0
            min_mag = 3.0
        
        # Higher probability suggests larger event
        if probability > 0.7:
            return (min_mag + 1.0, max_mag)
        elif probability > 0.5:
            return (min_mag + 0.5, max_mag - 0.5)
        else:
            return (min_mag, max_mag - 1.0)
    
    def _calculate_time_windows(self, base_prob: float) -> Dict:
        """Calculate probability for different time windows"""
        
        # 30-day probability (base)
        prob_30d = base_prob
        
        # 7-day probability (proportional)
        prob_7d = prob_30d * (7/30)
        
        # 24-hour probability
        prob_24h = prob_30d * (1/30)
        
        return {
            '24h': min(prob_24h, 1.0),
            '7d': min(prob_7d, 1.0),
            '30d': min(prob_30d, 1.0)
        }
    
    def _calculate_confidence(self, region: Dict, probability: float) -> float:
        """
        Calculate confidence in prediction
        Based on data quality and model reliability
        """
        
        # More historical data = higher confidence
        if region['historical_rate'] > 5.0:
            confidence = 0.8
        elif region['historical_rate'] > 2.0:
            confidence = 0.7
        else:
            confidence = 0.6
        
        # Very high or very low probabilities are less confident
        if 0.3 < probability < 0.7:
            confidence *= 1.1
        
        return min(confidence, 1.0)
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level"""
        if probability >= 0.7:
            return 'high'
        elif probability >= 0.4:
            return 'elevated'
        elif probability >= 0.2:
            return 'moderate'
        else:
            return 'low'
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points (km)"""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


class StressAccumulationModel:
    """
    Simple stress accumulation model
    In production, use coulomb stress transfer models
    """
    
    def __init__(self):
        self.stress_state = {}
    
    def update_stress(self, location: Tuple[float, float], 
                     magnitude: float, depth: float):
        """Update stress field after an earthquake"""
        # Simplified: earthquakes release stress locally
        # but increase stress in surrounding areas
        pass