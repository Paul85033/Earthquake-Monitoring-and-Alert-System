"""
Main entry point for real-time earthquake detection
"""

import sys
import signal
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    SENSOR_CONFIG, DETECTION_CONFIG, ALERT_CONFIG, 
    LOGGING_CONFIG, MODEL_PATH, DATABASE_PATH
)
from sensor_reader import create_sensor
from detector import EarthquakeDetector
from alert_system import AlertSystem
from database import SeismicDatabase

# Global flag for graceful shutdown
running = True


def setup_logging():
    """Setup logging configuration"""
    log_file = LOGGING_CONFIG['log_file']
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(LOGGING_CONFIG['level'])
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG['max_bytes'],
        backupCount=LOGGING_CONFIG['backup_count']
    )
    file_handler.setFormatter(
        logging.Formatter(LOGGING_CONFIG['format'])
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(LOGGING_CONFIG['format'])
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger = logging.getLogger(__name__)
    logger.info("Shutdown signal received")
    running = False


def main():
    """Main detection loop"""
    global running
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("SEISMIC AI DETECTOR - STARTING")
    logger.info("=" * 70)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Database
        db = SeismicDatabase(str(DATABASE_PATH))
        
        # Detector
        model_path = str(MODEL_PATH) if MODEL_PATH.exists() else None
        detector = EarthquakeDetector(DETECTION_CONFIG, model_path)
        
        # Alert system
        alert_system = AlertSystem(ALERT_CONFIG)
        
        # Sensor
        sensor = create_sensor(SENSOR_CONFIG)
        
        logger.info("All components initialized")
        logger.info("Starting detection loop...")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 70)
        
        # Start sensor
        sensor.start()
        
        # Main detection loop
        sample_count = 0
        last_status_time = time.time()
        
        for sample, timestamp in sensor.read_stream():
            if not running:
                break
            
            # Process sample
            result = detector.process_sample(sample, timestamp)
            
            sample_count += 1
            
            # Log to database (optional, every 100 samples to avoid overhead)
            if ALERT_CONFIG.get('enable_database', True) and sample_count % 100 == 0:
                db.log_sensor_data(
                    result['timestamp'],
                    result['sample'],
                    result['sta'],
                    result['lta'],
                    result['ratio']
                )
            
            # Check for event detection
            if result['event_detected']:
                event = result['event']
                
                # Log to database
                db.log_event(event)
                
                # Send alerts
                alert_system.send_alert(event)
            
            # Status update every 10 seconds
            current_time = time.time()
            if current_time - last_status_time >= 10:
                stats = detector.get_statistics()
                logger.info(
                    f"Status: {stats['total_samples']} samples, "
                    f"{stats['triggers']} triggers, "
                    f"{stats['confirmed_events']} events"
                )
                last_status_time = current_time
        
        # Cleanup
        sensor.stop()
        
        logger.info("-" * 70)
        logger.info("Detection stopped")
        
        # Final statistics
        stats = detector.get_statistics()
        logger.info("=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Runtime:          {stats['runtime_seconds']:.1f}s")
        logger.info(f"Samples:          {stats['total_samples']}")
        logger.info(f"Triggers:         {stats['triggers']}")
        logger.info(f"Confirmed Events: {stats['confirmed_events']}")
        logger.info(f"False Positives:  {stats['false_positives']}")
        logger.info(f"Detection Rate:   {stats['detection_rate']:.1%}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        logger.info("Seismic AI Detector shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())