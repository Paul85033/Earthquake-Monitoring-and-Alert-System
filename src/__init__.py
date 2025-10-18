"""
Seismic AI Detector Package
"""

__version__ = "1.0.0"
__author__ = "team"

from .detector import EarthquakeDetector
from .database import SeismicDatabase
from .alert_system import AlertSystem
from .sensor_reader import create_sensor

__all__ = [
    'EarthquakeDetector',
    'SeismicDatabase',
    'AlertSystem',
    'create_sensor'
]