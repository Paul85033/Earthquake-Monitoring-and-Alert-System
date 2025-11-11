"""
User Authentication and Management System
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import requests

logger = logging.getLogger(__name__)


class UserManager:
    """Manage user accounts and authentication"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
        logger.info("User manager initialized")
    
    def _create_tables(self):
        """Create users and sessions tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                alert_radius_km REAL DEFAULT 100,
                email_alerts BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        ''')
        
        # Sessions table for login management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Alert history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_id INTEGER,
                alert_type TEXT NOT NULL,
                sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
                event_magnitude REAL,
                event_distance_km REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Password reset tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                used BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, username: str, email: str, password: str,
                     city: str, country: str) -> Dict:
        """
        Register a new user
        
        Returns:
            Dictionary with success status and message
        """
        try:
            # Validate input
            if not all([username, email, password, city, country]):
                return {'success': False, 'message': 'All fields are required'}
            
            if len(password) < 6:
                return {'success': False, 'message': 'Password must be at least 6 characters'}
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Get coordinates for city
            lat, lon = self._geocode_location(city, country)
            
            # Insert user
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, city, country, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, city, country, lat, lon))
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"New user registered: {username} ({email}) from {city}, {country}")
            
            return {
                'success': True,
                'message': 'Registration successful!',
                'user_id': user_id
            }
            
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return {'success': False, 'message': 'Username already exists'}
            elif 'email' in str(e):
                return {'success': False, 'message': 'Email already registered'}
            else:
                return {'success': False, 'message': 'Registration failed'}
        
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {'success': False, 'message': 'Registration failed'}
    
    def login_user(self, username: str, password: str) -> Dict:
        """
        Authenticate user and create session
        
        Returns:
            Dictionary with success status, session token, and user info
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            # Verify password
            if not self._verify_password(password, user['password_hash']):
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now().replace(hour=23, minute=59, second=59)  # End of day
            
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user['id'], session_token, expires_at.isoformat()))
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.now().isoformat(), user['id']))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User logged in: {username}")
            
            return {
                'success': True,
                'message': 'Login successful',
                'session_token': session_token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'city': user['city'],
                    'country': user['country']
                }
            }
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {'success': False, 'message': 'Login failed'}
    
    def get_user_by_session(self, session_token: str) -> Optional[Dict]:
        """Get user info from session token"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.* FROM users u
                JOIN sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.expires_at > ?
            ''', (session_token, datetime.now().isoformat()))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return dict(user)
            return None
            
        except Exception as e:
            logger.error(f"Session lookup error: {e}")
            return None
    
    def create_password_reset_token(self, email: str) -> Dict:
        """
        Create password reset token and return user info
        
        Args:
            email: User's email address
        
        Returns:
            Dictionary with success status and token info
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find user by email
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                # Don't reveal if email exists (security)
                return {
                    'success': True,
                    'message': 'If that email exists, a reset link has been sent'
                }
            
            # Generate secure token
            token = secrets.token_urlsafe(32)
            
            # Token expires in 1 hour
            expires_at = datetime.now() + timedelta(hours=1)
            
            # Store token
            cursor.execute('''
                INSERT INTO password_reset_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (user['id'], token, expires_at.isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Password reset token created for {email}")
            
            return {
                'success': True,
                'message': 'Password reset link sent to your email',
                'token': token,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'username': user['username']
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating reset token: {e}")
            return {
                'success': False,
                'message': 'Error processing request'
            }
    
    def verify_reset_token(self, token: str) -> Optional[Dict]:
        """
        Verify password reset token is valid
        
        Returns:
            User dict if valid, None if invalid/expired
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.*, prt.id as token_id, prt.expires_at, prt.used
                FROM users u
                JOIN password_reset_tokens prt ON u.id = prt.user_id
                WHERE prt.token = ?
            ''', (token,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            # Check if already used
            if result['used']:
                logger.warning(f"Attempted reuse of password reset token")
                return None
            
            # Check if expired
            expires_at = datetime.fromisoformat(result['expires_at'])
            if datetime.now() > expires_at:
                logger.warning(f"Expired password reset token used")
                return None
            
            return dict(result)
            
        except Exception as e:
            logger.error(f"Error verifying reset token: {e}")
            return None
    
    def reset_password(self, token: str, new_password: str) -> Dict:
        """
        Reset user password using valid token
        
        Args:
            token: Password reset token
            new_password: New password
        
        Returns:
            Dictionary with success status
        """
        try:
            # Verify token
            user_data = self.verify_reset_token(token)
            
            if not user_data:
                return {
                    'success': False,
                    'message': 'Invalid or expired reset link'
                }
            
            # Validate new password
            if len(new_password) < 6:
                return {
                    'success': False,
                    'message': 'Password must be at least 6 characters'
                }
            
            # Hash new password
            password_hash = self._hash_password(new_password)
            
            # Update password
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET password_hash = ? WHERE id = ?
            ''', (password_hash, user_data['id']))
            
            # Mark token as used
            cursor.execute('''
                UPDATE password_reset_tokens SET used = 1 WHERE id = ?
            ''', (user_data['token_id'],))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Password reset successful for user {user_data['username']}")
            
            return {
                'success': True,
                'message': 'Password reset successful! You can now login.'
            }
            
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            return {
                'success': False,
                'message': 'Error resetting password'
            }
    
    def cleanup_expired_tokens(self):
        """Remove expired password reset tokens (run periodically)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM password_reset_tokens
                WHERE expires_at < ? OR used = 1
            ''', (datetime.now().isoformat(),))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired password reset tokens")
                
        except Exception as e:
            logger.error(f"Error cleaning up tokens: {e}")
    
    def get_users_in_radius(self, lat: float, lon: float, radius_km: float) -> List[Dict]:
        """
        Get all users within radius of a location
        
        Args:
            lat, lon: Center point coordinates
            radius_km: Search radius in kilometers
        
        Returns:
            List of user dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE email_alerts = 1')
            all_users = cursor.fetchall()
            conn.close()
            
            # Filter by distance
            users_in_radius = []
            for user in all_users:
                if user['latitude'] and user['longitude']:
                    distance = self._haversine_distance(
                        lat, lon,
                        user['latitude'], user['longitude']
                    )
                    
                    # Check against user's alert radius preference
                    alert_radius = user['alert_radius_km'] or 100
                    
                    if distance <= alert_radius:
                        user_dict = dict(user)
                        user_dict['distance_km'] = distance
                        users_in_radius.append(user_dict)
            
            return users_in_radius
            
        except Exception as e:
            logger.error(f"Error getting users in radius: {e}")
            return []
    
    def log_alert_sent(self, user_id: int, event_magnitude: float, 
                      event_distance_km: float, alert_type: str = 'email'):
        """Log that an alert was sent to a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alert_history (user_id, alert_type, event_magnitude, event_distance_km)
                VALUES (?, ?, ?, ?)
            ''', (user_id, alert_type, event_magnitude, event_distance_km))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${password_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt, hash_hex = stored_hash.split('$')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash.hex() == hash_hex
        except:
            return False


    def _geocode_location(self, city: str, country: str) -> tuple:
        try:
            query = f"{city}, {country}"
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query, "format": "json", "limit": 1}
            headers = {"User-Agent": "EarthquakeMonitor/1.0 (contact@example.com)"}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        
            if not data:
                logger.warning(f"No geocoding result for {query}")
                return (0.0, 0.0)
        
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            logger.info(f"Geocoded {query} -> ({lat}, {lon})")
            return (lat, lon)
        
        except Exception as e:
            logger.error(f"Geocoding failed for {city}, {country}: {e}")
            return (0.0, 0.0)

    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points (km)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371.0  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c