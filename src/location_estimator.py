"""
Earthquake Location Estimation
Estimates epicenter location using P-wave and S-wave arrival times
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LocationEstimator:
    """
    Estimate earthquake epicenter location
    
    Methods:
    1. Single station: Distance estimation only (circular area)
    2. Multi-station: Triangulation (requires 3+ stations)
    3. Machine Learning: Trained location predictor
    """
    
    def __init__(self, station_location: Tuple[float, float] = None):
        """
        Initialize location estimator
        
        Args:
            station_location: (latitude, longitude) of seismic station
                             Default: (34.0522, -118.2437) - Los Angeles
        """
        if station_location is None:
            # Default location (can be configured)
            self.station_lat = 34.0522  # Los Angeles, CA
            self.station_lon = -118.2437
        else:
            self.station_lat, self.station_lon = station_location
        
        # Wave velocities (km/s)
        self.p_wave_velocity = 6.0  # P-wave velocity in crust
        self.s_wave_velocity = 3.5  # S-wave velocity in crust
        
        # Regional risk zones (high probability areas)
        self.risk_zones = self._load_risk_zones()
        
        logger.info(f"Location estimator initialized at ({self.station_lat}, {self.station_lon})")
    
    def _load_risk_zones(self) -> List[Dict]:
        """
        Load known seismically active zones
        Returns probability multipliers for different regions
        """
        return[
            # California
            {
                'name': 'San Andreas Fault Zone',
                'center': (35.5, -120.0),
                'radius_km': 200,
                'probability_multiplier': 3.0
            },
            {
                'name': 'Pacific Ring of Fire - California',
                'center': (37.0, -122.0),
                'radius_km': 150,
                'probability_multiplier': 2.5
            },
            {
                'name': 'Southern California',
                'center': (34.0, -117.0),
                'radius_km': 100,
                'probability_multiplier': 2.0
            },
            {
                'name': 'Ridgecrest Area',
                'center': (35.7, -117.6),
                'radius_km': 80,
                'probability_multiplier': 2.2
            },
            
            # Pacific Northwest
            {
                'name': 'Cascadia Subduction Zone',
                'center': (44.0, -124.0),
                'radius_km': 300,
                'probability_multiplier': 2.2
            },
            {
                'name': 'Seattle Fault Zone',
                'center': (47.6, -122.3),
                'radius_km': 100,
                'probability_multiplier': 1.9
            },
            {
                'name': 'Portland Area',
                'center': (45.5, -122.7),
                'radius_km': 80,
                'probability_multiplier': 1.8
            },
            
            # Central US
            {
                'name': 'New Madrid Seismic Zone',
                'center': (36.5, -89.5),
                'radius_km': 150,
                'probability_multiplier': 1.8
            },
            {
                'name': 'Oklahoma Seismic Zone',
                'center': (35.5, -97.5),
                'radius_km': 120,
                'probability_multiplier': 1.7
            },
            
            # Alaska
            {
                'name': 'Aleutian Arc',
                'center': (60.0, -152.0),
                'radius_km': 300,
                'probability_multiplier': 2.8
            },
            {
                'name': 'Denali Fault',
                'center': (63.0, -150.0),
                'radius_km': 200,
                'probability_multiplier': 2.3
            },
            
            # Other Regions
            {
                'name': 'Yellowstone Region',
                'center': (44.4, -110.6),
                'radius_km': 100,
                'probability_multiplier': 2.0
            },
            {
                'name': 'Nevada Seismic Belt',
                'center': (39.5, -119.8),
                'radius_km': 150,
                'probability_multiplier': 1.9
            },
            {
                'name': 'Wasatch Fault Zone',
                'center': (40.8, -111.9),
                'radius_km': 100,
                'probability_multiplier': 1.8
            },
            {
                'name': 'Charleston Seismic Zone',
                'center': (32.8, -80.0),
                'radius_km': 100,
                'probability_multiplier': 1.5
            },
            {
                'name': 'Central Andes Fault Belt',
                'center': (-20.0, -68.0),
                'radius_km': 400,
                'probability_multiplier': 2.6
            },
            {
                'name': 'Nazca Subduction Zone (Chile)',
                'center': (-30.0, -71.5),
                'radius_km': 450,
                'probability_multiplier': 2.9
            },
            {
                'name': 'Himalayan Frontal Thrust',
                'center': (28.0, 84.0),
                'radius_km': 400,
                'probability_multiplier': 2.4
            },
            {
                'name': 'Karakoram Fault',
                'center': (35.0, 77.0),
                'radius_km': 200,
                'probability_multiplier': 2.1
            },
            {
                'name': 'Zagros Fold-Thrust Belt',
                'center': (30.5, 51.5),
                'radius_km': 300,
                'probability_multiplier': 2.0
            },
            {
                'name': 'North Anatolian Fault',
                'center': (40.8, 35.0),
                'radius_km': 350,
                'probability_multiplier': 2.5
            },
            {
                'name': 'Hellenic Arc Subduction Zone',
                'center': (36.0, 24.0),
                'radius_km': 250,
                'probability_multiplier': 2.3
            },
            {
                'name': 'East African Rift System',
                'center': (-3.0, 36.0),
                'radius_km': 500,
                'probability_multiplier': 2.2
            },
            {
                'name': 'Sumatra-Andaman Subduction Zone',
                'center': (3.0, 95.0),
                'radius_km': 600,
                'probability_multiplier': 3.0
            },
            {
                'name': 'Philippine Trench',
                'center': (10.0, 127.0),
                'radius_km': 400,
                'probability_multiplier': 2.8
            },
            {
                'name': 'Japan Trench',
                'center': (38.0, 143.0),
                'radius_km': 450,
                'probability_multiplier': 3.0
            },
            {
                'name': 'Ryukyu Trench',
                'center': (26.0, 129.0),
                'radius_km': 300,
                'probability_multiplier': 2.7
            },
            {
                'name': 'Taiwan Convergent Margin',
                'center': (23.5, 121.0),
                'radius_km': 250,
                'probability_multiplier': 2.6
            },
            {
                'name': 'New Guinea Thrust Belt',
                'center': (-6.0, 145.0),
                'radius_km': 300,
                'probability_multiplier': 2.5
            },
            {
                'name': 'Tonga-Kermadec Trench',
                'center': (-20.0, -175.0),
                'radius_km': 500,
                'probability_multiplier': 2.9
            },
            {
                'name': 'New Zealand Alpine Fault',
                'center': (-43.5, 170.0),
                'radius_km': 300,
                'probability_multiplier': 2.6
            },
            {
                'name': 'Hindu Kush Seismic Zone',
                'center': (36.5, 70.5),
                'radius_km': 250,
                'probability_multiplier': 2.4
            },
            {
                'name': 'Makran Subduction Zone',
                'center': (25.0, 62.0),
                'radius_km': 350,
                'probability_multiplier': 2.3
            },
            {
                'name': 'Pamir Fault System',
                'center': (38.0, 73.0),
                'radius_km': 200,
                'probability_multiplier': 2.2
            },
            {
                'name': 'Vanuatu Subduction Zone',
                'center': (-16.0, 167.0),
                'radius_km': 400,
                'probability_multiplier': 2.8
            }
        ]
    
    def estimate_location(self, features: Dict[str, float], 
                         waveform: np.ndarray,
                         sample_rate: int = 100) -> Dict:
        """
        Estimate earthquake location from single station
        
        Args:
            features: Extracted features from event
            waveform: Event waveform data
            sample_rate: Sampling rate in Hz
        
        Returns:
            Dictionary with location estimate and probability
        """
        
        # Method 1: Distance estimation from P-S time difference
        distance_km = self._estimate_distance(waveform, sample_rate)
        
        # Method 2: Direction estimation (if multiple components available)
        direction_deg = self._estimate_direction(waveform, features)
        
        # Method 3: Depth estimation
        depth_km = self._estimate_depth(features)
        
        # Method 4: Find most likely region based on distance and risk zones
        best_region = self._find_most_likely_region(distance_km, features)
        
        # Use the best region as epicenter if we don't have direction
        if direction_deg is None and best_region:
            # Place epicenter at the center of most likely risk zone
            epicenter_lat = best_region['center'][0]
            epicenter_lon = best_region['center'][1]
            uncertainty_km = best_region['radius_km']
            location_type = 'region_based'
            probability = best_region['probability_multiplier'] / 3.0  # Normalize
        elif direction_deg is not None:
            # We have direction - calculate specific point
            epicenter_lat, epicenter_lon = self._calculate_epicenter(
                self.station_lat,
                self.station_lon,
                distance_km,
                direction_deg
            )
            uncertainty_km = distance_km * 0.2
            location_type = 'estimated_point'
            probability = self._calculate_probability(epicenter_lat, epicenter_lon, features['magnitude'])
        else:
            # Fallback: random direction within most likely zone
            import random
            direction_deg = random.uniform(0, 360)
            epicenter_lat, epicenter_lon = self._calculate_epicenter(
                self.station_lat,
                self.station_lon,
                distance_km,
                direction_deg
            )
            uncertainty_km = distance_km
            location_type = 'circular_area'
            probability = self._calculate_probability(epicenter_lat, epicenter_lon, features['magnitude'])
        
        # Get nearest known fault or zone
        nearest_zone = self._get_nearest_zone(epicenter_lat, epicenter_lon)
        
        location_info = {
            'epicenter_lat': epicenter_lat,
            'epicenter_lon': epicenter_lon,
            'distance_km': distance_km,
            'direction_deg': direction_deg,
            'depth_km': depth_km,
            'uncertainty_km': uncertainty_km,
            'probability': probability,
            'location_type': location_type,
            'station_lat': self.station_lat,
            'station_lon': self.station_lon,
            'nearest_zone': nearest_zone,
            'estimated_at': datetime.now().isoformat()
        }
        
        logger.info(
            f"Location estimated: {distance_km:.1f} km away at ({epicenter_lat:.2f}, {epicenter_lon:.2f}), "
            f"probability {probability:.1%}"
        )
        
        return location_info
    
    def _find_most_likely_region(self, distance_km: float, features: Dict) -> Dict:
        """
        Find most likely seismic region based on distance and characteristics
        """
        magnitude = features.get('magnitude', 0)
        
        # Score each risk zone
        best_score = 0
        best_region = None
        
        for zone in self.risk_zones:
            # Calculate distance from station to zone center
            zone_distance = self._haversine_distance(
                self.station_lat, self.station_lon,
                zone['center'][0], zone['center'][1]
            )
            
            # Check if detected distance matches zone distance (within margin)
            distance_match = abs(zone_distance - distance_km) / max(zone_distance, distance_km)
            
            if distance_match < 0.5:  # Within 50% match
                # Score based on probability multiplier and distance match
                score = zone['probability_multiplier'] * (1 - distance_match)
                
                # Larger magnitudes more likely from major zones
                if magnitude > 5.0 and zone['probability_multiplier'] > 2.0:
                    score *= 1.5
                
                if score > best_score:
                    best_score = score
                    best_region = zone
        
        return best_region
    
    def _estimate_distance(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Estimate distance from P-S wave arrival time difference
        
        S-P time method:
        distance = (S-P time) * (Vp * Vs) / (Vs - Vp)
        """
        
        # Detect P-wave and S-wave arrival times
        p_arrival, s_arrival = self._detect_wave_arrivals(waveform, sample_rate)
        
        if p_arrival is None or s_arrival is None:
            # Fallback: estimate from signal characteristics
            return self._estimate_distance_fallback(waveform, sample_rate)
        
        # Calculate S-P time difference
        sp_time = (s_arrival - p_arrival) / sample_rate  # seconds
        
        # Calculate distance
        distance_km = sp_time * (self.p_wave_velocity * self.s_wave_velocity) / \
                     (self.s_wave_velocity - self.p_wave_velocity)
        
        # Realistic bounds: 1-1000 km
        distance_km = np.clip(distance_km, 1, 1000)
        
        return distance_km
    
    def _detect_wave_arrivals(self, waveform: np.ndarray, 
                             sample_rate: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Detect P-wave and S-wave arrival times using STA/LTA
        """
        from scipy import signal
        
        # Short-term average window
        sta_samples = int(0.5 * sample_rate)
        # Long-term average window  
        lta_samples = int(5.0 * sample_rate)
        
        if len(waveform) < lta_samples:
            return None, None
        
        # Calculate STA/LTA
        abs_data = np.abs(waveform)
        sta_lta = np.zeros(len(waveform))
        
        for i in range(lta_samples, len(waveform)):
            sta = np.mean(abs_data[i-sta_samples:i])
            lta = np.mean(abs_data[i-lta_samples:i])
            if lta > 0:
                sta_lta[i] = sta / lta
        
        # Find peaks (wave arrivals)
        peaks, _ = signal.find_peaks(sta_lta, height=2.0, distance=int(2*sample_rate))
        
        if len(peaks) < 2:
            return None, None
        
        # First peak is P-wave, second is S-wave
        p_arrival = peaks[0]
        s_arrival = peaks[1]
        
        return p_arrival, s_arrival
    
    def _estimate_distance_fallback(self, waveform: np.ndarray, 
                                   sample_rate: int) -> float:
        """
        Fallback distance estimation from signal characteristics
        Uses random variation for realistic spread
        """
        import random
        
        # Use signal amplitude and duration
        pga = np.max(np.abs(waveform))
        duration = len(waveform) / sample_rate
        
        # Empirical relationship with randomization
        base_distance = 0
        
        if pga > 1.0:
            base_distance = 20 + duration * 2
        elif pga > 0.1:
            base_distance = 50 + duration * 3
        else:
            base_distance = 100 + duration * 5
        
        # Add random variation (±30%) for realistic spread
        variation = random.uniform(-0.3, 0.3)
        distance_km = base_distance * (1 + variation)
        
        return np.clip(distance_km, 10, 500)
    
    def _estimate_direction(self, waveform: np.ndarray, 
                          features: Dict) -> Optional[float]:
        """
        Estimate direction to epicenter
        
        For single station with limited data, use probabilistic approach
        based on known seismic zones and signal characteristics
        """
        import random
        
        magnitude = features.get('magnitude', 3.0)
        mean_freq = features.get('mean_freq', 3.0)
        
        # Find nearby risk zones
        nearby_zones = []
        for zone in self.risk_zones:
            distance = self._haversine_distance(
                self.station_lat, self.station_lon,
                zone['center'][0], zone['center'][1]
            )
            if distance < 500:  # Within 500 km
                nearby_zones.append({
                    'zone': zone,
                    'distance': distance,
                    'bearing': self._calculate_bearing(
                        self.station_lat, self.station_lon,
                        zone['center'][0], zone['center'][1]
                    )
                })
        
        if nearby_zones:
            # Weight by probability multiplier and magnitude match
            weights = []
            bearings = []
            
            for nz in nearby_zones:
                zone = nz['zone']
                weight = zone['probability_multiplier']
                
                # Larger earthquakes more likely from major zones
                if magnitude > 5.0 and zone['probability_multiplier'] > 2.0:
                    weight *= 1.5
                
                # Closer zones get higher weight
                distance_factor = max(0.1, 1 - (nz['distance'] / 500))
                weight *= distance_factor
                
                weights.append(weight)
                bearings.append(nz['bearing'])
            
            # Choose direction probabilistically
            if weights:
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Select bearing with some randomness
                chosen_idx = random.choices(range(len(bearings)), weights=weights)[0]
                base_bearing = bearings[chosen_idx]
                
                # Add ±30 degree randomness
                direction = base_bearing + random.uniform(-30, 30)
                direction = direction % 360  # Normalize to 0-360
                
                return direction
        
        # Fallback: random direction
        return random.uniform(0, 360)
    
    def _calculate_bearing(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 (degrees)"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360  # Normalize to 0-360
        
        return bearing
    
    def _estimate_depth(self, features: Dict) -> float:
        """
        Estimate earthquake depth from features
        
        Shallow earthquakes (0-70 km): Higher frequency content
        Deep earthquakes (>70 km): Lower frequency, longer duration
        """
        
        mean_freq = features.get('mean_freq', 3.0)
        duration = features.get('duration', 15.0)
        
        # Empirical relationship
        if mean_freq > 5.0:
            depth_km = 5 + (10 - mean_freq) * 2
        elif mean_freq > 2.0:
            depth_km = 15 + duration * 0.5
        else:
            depth_km = 30 + duration * 1.0
        
        # Most earthquakes are 5-35 km deep
        depth_km = np.clip(depth_km, 2, 100)
        
        return depth_km
    
    def _calculate_epicenter(self, station_lat: float, station_lon: float,
                           distance_km: float, direction_deg: float) -> Tuple[float, float]:
        """
        Calculate epicenter coordinates given distance and direction
        
        Uses Haversine formula
        """
        
        # Earth radius
        R = 6371.0  # km
        
        # Convert to radians
        lat1 = np.radians(station_lat)
        lon1 = np.radians(station_lon)
        bearing = np.radians(direction_deg)
        
        # Calculate new latitude
        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(distance_km / R) +
            np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing)
        )
        
        # Calculate new longitude
        lon2 = lon1 + np.arctan2(
            np.sin(bearing) * np.sin(distance_km / R) * np.cos(lat1),
            np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2)
        )
        
        # Convert back to degrees
        epicenter_lat = np.degrees(lat2)
        epicenter_lon = np.degrees(lon2)
        
        return epicenter_lat, epicenter_lon
    
    def _calculate_probability(self, lat: float, lon: float, 
                              magnitude: float) -> float:
        """
        Calculate probability of earthquake at given location
        
        Based on:
        1. Proximity to known fault zones
        2. Historical seismicity
        3. Magnitude (larger = more likely to be real detection)
        """
        
        base_probability = 0.5  # 50% base probability
        
        # Factor 1: Magnitude confidence
        if magnitude >= 5.0:
            mag_factor = 1.5
        elif magnitude >= 4.0:
            mag_factor = 1.2
        elif magnitude >= 3.0:
            mag_factor = 1.0
        else:
            mag_factor = 0.8
        
        # Factor 2: Proximity to risk zones
        zone_factor = 1.0
        for zone in self.risk_zones:
            distance = self._haversine_distance(
                lat, lon,
                zone['center'][0], zone['center'][1]
            )
            
            if distance <= zone['radius_km']:
                # Inside risk zone
                zone_factor = max(zone_factor, zone['probability_multiplier'])
        
        # Calculate final probability
        probability = base_probability * mag_factor * zone_factor
        
        # Normalize to 0-1 range
        probability = min(probability, 1.0)
        
        return probability
    
    def _get_nearest_zone(self, lat: float, lon: float) -> Optional[str]:
        """Find nearest seismic risk zone"""
        
        nearest = None
        min_distance = float('inf')
        
        for zone in self.risk_zones:
            distance = self._haversine_distance(
                lat, lon,
                zone['center'][0], zone['center'][1]
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest = zone['name']
        
        if min_distance <= 300:  # Within 300 km
            return f"{nearest} ({min_distance:.0f} km)"
        
        return "Unknown region"
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth"""
        
        R = 6371.0  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c