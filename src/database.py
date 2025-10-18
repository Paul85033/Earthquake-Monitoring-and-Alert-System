"""
SQLite database for logging earthquake events
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class SeismicDatabase:
    """Database handler for earthquake events"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
        logger.info(f"Database initialized: {db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                magnitude REAL NOT NULL,
                duration REAL NOT NULL,
                confidence REAL NOT NULL,
                pga REAL NOT NULL,
                mean_freq REAL,
                dominant_freq REAL,
                energy REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Raw sensor data table (optional, for debugging)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                sta REAL,
                lta REAL,
                ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, event: Dict):
        """Log an earthquake event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (
                timestamp, magnitude, duration, confidence,
                pga, mean_freq, dominant_freq, energy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event['timestamp'],
            event['magnitude'],
            event['duration'],
            event['confidence'],
            event['pga'],
            event.get('mean_freq', 0),
            event.get('dominant_freq', 0),
            event.get('energy', 0)
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Event logged to database: M{event['magnitude']:.1f}")
    
    def log_sensor_data(self, timestamp: str, value: float, 
                       sta: float = None, lta: float = None, ratio: float = None):
        """Log raw sensor data (optional, for debugging)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data (timestamp, value, sta, lta, ratio)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, value, sta, lta, ratio))
        
        conn.commit()
        conn.close()
    
    def log_system_message(self, level: str, message: str):
        """Log system message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_logs (timestamp, level, message)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), level, message))
        
        conn.commit()
        conn.close()
    
    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Get most recent events"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM events
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_events_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """Get events within date range"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM events
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total events
        cursor.execute('SELECT COUNT(*) FROM events')
        total_events = cursor.fetchone()[0]
        
        # Average magnitude
        cursor.execute('SELECT AVG(magnitude) FROM events')
        avg_magnitude = cursor.fetchone()[0] or 0
        
        # Max magnitude
        cursor.execute('SELECT MAX(magnitude) FROM events')
        max_magnitude = cursor.fetchone()[0] or 0
        
        # Events by day
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM events
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        ''')
        
        daily_counts = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_events': total_events,
            'avg_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'daily_counts': daily_counts
        }
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM sensor_data
            WHERE created_at < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {deleted} old sensor data records")