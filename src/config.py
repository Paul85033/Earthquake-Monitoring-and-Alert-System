"""
Configuration file for Seismic AI Detector
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
SAMPLES_DIR = DATA_DIR / "samples"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_PATH = DATA_DIR / "seismic_log.db"

# Model configuration
MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"

# Sensor configuration
SENSOR_CONFIG = {
    'sample_rate': 100,  # Hz
    'sensor_type': 'simulated',  # 'simulated', 'api', 'serial'
    'api_endpoint': None,  # For API-based sensors
    'serial_port': None,  # For hardware sensors (e.g., '/dev/ttyUSB0')
    'baud_rate': 9600,  # For serial sensors
}

# Detection configuration
DETECTION_CONFIG = {
    'sta_window': 1.0,  # Short-term average window (seconds)
    'lta_window': 30.0,  # Long-term average window (seconds)
    'trigger_ratio': 4.0,  # STA/LTA trigger threshold
    'detrigger_ratio': 1.5,  # STA/LTA detrigger threshold
    'ml_confidence_threshold': 0.7,  # Minimum confidence for ML detection
    'magnitude_threshold': 3.0,  # Minimum magnitude to alert
    'bandpass_low': 0.5,  # Low frequency cutoff (Hz)
    'bandpass_high': 10.0,  # High frequency cutoff (Hz)
}

# Alert configuration
ALERT_CONFIG = {
    'enable_email': False,
    'enable_sms': False,
    'enable_console': True,
    'enable_database': True,
    'enable_webhook': False,
    
    # Email settings
    'email_smtp_server': 'smtp.gmail.com',
    'email_smtp_port': 587,
    'email_sender': os.getenv('ALERT_EMAIL_SENDER', ''),
    'email_password': os.getenv('ALERT_EMAIL_PASSWORD', ''),
    'email_recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(','),
    
    # SMS settings (Twilio)
    'sms_account_sid': os.getenv('TWILIO_ACCOUNT_SID', ''),
    'sms_auth_token': os.getenv('TWILIO_AUTH_TOKEN', ''),
    'sms_from_number': os.getenv('TWILIO_FROM_NUMBER', ''),
    'sms_to_numbers': os.getenv('TWILIO_TO_NUMBERS', '').split(','),
    
    # Webhook settings
    'webhook_url': os.getenv('ALERT_WEBHOOK_URL', ''),
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    'log_file': LOGS_DIR / 'earthquake_detector.log',
    'max_bytes': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'update_interval': 1000,  # milliseconds
}

# Training configuration
TRAINING_CONFIG = {
    'num_earthquakes': 500,
    'num_noise_samples': 1000,
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
}