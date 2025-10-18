"""
Sensor data reader for various input sources
"""

import numpy as np
import time
from datetime import datetime
from typing import Generator, Tuple
import logging

logger = logging.getLogger(__name__)


class SensorReader:
    """Base class for sensor readers"""
    
    def __init__(self, sample_rate: int = 100):
        self.sample_rate = sample_rate
        self.running = False
        
    def start(self):
        """Start reading sensor data"""
        self.running = True
        logger.info(f"Started {self.__class__.__name__}")
        
    def stop(self):
        """Stop reading sensor data"""
        self.running = False
        logger.info(f"Stopped {self.__class__.__name__}")
        
    def read_stream(self) -> Generator[Tuple[float, datetime], None, None]:
        """
        Generator that yields (sample, timestamp) tuples
        Must be implemented by subclasses
        """
        raise NotImplementedError


class SimulatedSensor(SensorReader):
    """Simulated sensor for testing"""
    
    def __init__(self, sample_rate: int = 100, scenario: str = 'mixed'):
        super().__init__(sample_rate)
        self.scenario = scenario
        self.sample_index = 0
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> np.ndarray:
        """Generate test scenario"""
        logger.info(f"Generating test scenario: {self.scenario}")
        
        if self.scenario == 'noise_only':
            # 60 seconds of noise
            return np.random.normal(0, 0.01, 6000)
            
        elif self.scenario == 'single_event':
            # Noise + one earthquake
            noise = np.random.normal(0, 0.01, 5000)
            eq = self._generate_earthquake(4.5, 20)
            return np.concatenate([noise, eq, noise])
            
        else:  # 'mixed'
            # Background noise (60s)
            noise1 = np.random.normal(0, 0.01, 6000)
            
            # Small earthquake (M3.5, 15s)
            eq1 = self._generate_earthquake(3.5, 15)
            
            # More noise (40s)
            noise2 = np.random.normal(0, 0.01, 4000)
            
            # Large earthquake (M5.5, 25s)
            eq2 = self._generate_earthquake(5.5, 25)
            
            # Final noise (30s)
            noise3 = np.random.normal(0, 0.01, 3000)
            
            return np.concatenate([noise1, eq1, noise2, eq2, noise3])
    
    def _generate_earthquake(self, magnitude: float, duration: float) -> np.ndarray:
        """Generate synthetic earthquake waveform"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        amplitude = 10 ** (magnitude - 3)
        
        # Envelope
        onset = 2.0
        decay = 5.0
        envelope = np.exp(-((t - onset) / decay) ** 2) * (t >= onset)
        envelope *= np.exp(-(t - onset) / 10) * (t >= onset)
        
        # P and S waves
        p_wave = np.sin(2 * np.pi * 5 * t)
        s_wave = np.sin(2 * np.pi * 2 * t + np.pi/4)
        
        signal_wave = amplitude * envelope * (0.6 * p_wave + 0.4 * s_wave)
        noise = np.random.normal(0, amplitude * 0.05, num_samples)
        
        return signal_wave + noise
    
    def read_stream(self) -> Generator[Tuple[float, datetime], None, None]:
        """Read simulated sensor stream"""
        self.sample_index = 0
        
        while self.running and self.sample_index < len(self.test_data):
            sample = float(self.test_data[self.sample_index])
            timestamp = datetime.now()
            
            yield sample, timestamp
            
            self.sample_index += 1
            time.sleep(1.0 / self.sample_rate)  # Simulate real-time


class SerialSensor(SensorReader):
    """Read from hardware sensor via serial port"""
    
    def __init__(self, port: str, baud_rate: int = 9600, sample_rate: int = 100):
        super().__init__(sample_rate)
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        
    def start(self):
        """Open serial connection"""
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            super().start()
            logger.info(f"Connected to serial port {self.port}")
        except ImportError:
            logger.error("pyserial not installed. Run: pip install pyserial")
            raise
        except Exception as e:
            logger.error(f"Failed to open serial port: {e}")
            raise
    
    def stop(self):
        """Close serial connection"""
        if self.serial_conn:
            self.serial_conn.close()
        super().stop()
    
    def read_stream(self) -> Generator[Tuple[float, datetime], None, None]:
        """Read from serial port"""
        while self.running:
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    sample = float(line)
                    timestamp = datetime.now()
                    yield sample, timestamp
            except ValueError:
                logger.warning(f"Invalid data from serial: {line}")
            except Exception as e:
                logger.error(f"Serial read error: {e}")
                break


class APISensor(SensorReader):
    """Read from remote API endpoint"""
    
    def __init__(self, endpoint: str, sample_rate: int = 100):
        super().__init__(sample_rate)
        self.endpoint = endpoint
        
    def read_stream(self) -> Generator[Tuple[float, datetime], None, None]:
        """Read from API"""
        import requests
        
        while self.running:
            try:
                response = requests.get(self.endpoint, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    sample = float(data.get('value', 0))
                    timestamp = datetime.now()
                    yield sample, timestamp
                    
                time.sleep(1.0 / self.sample_rate)
            except Exception as e:
                logger.error(f"API read error: {e}")
                time.sleep(1.0)


def create_sensor(config: dict) -> SensorReader:
    """Factory function to create appropriate sensor"""
    sensor_type = config.get('sensor_type', 'simulated')
    sample_rate = config.get('sample_rate', 100)
    
    if sensor_type == 'simulated':
        return SimulatedSensor(sample_rate=sample_rate)
    
    elif sensor_type == 'serial':
        port = config.get('serial_port')
        baud_rate = config.get('baud_rate', 9600)
        if not port:
            raise ValueError("serial_port not specified in config")
        return SerialSensor(port, baud_rate, sample_rate)
    
    elif sensor_type == 'api':
        endpoint = config.get('api_endpoint')
        if not endpoint:
            raise ValueError("api_endpoint not specified in config")
        return APISensor(endpoint, sample_rate)
    
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")