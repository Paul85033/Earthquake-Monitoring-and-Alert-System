"""
User Alert System for Earthquake Warnings
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class UserAlertSystem:
    """Send earthquake alerts to registered users"""
    
    def __init__(self, config: dict, user_manager):
        self.config = config
        self.user_manager = user_manager
        self.smtp_server = config['email_smtp_server']
        self.smtp_port = config['email_smtp_port']
        self.sender_email = config['email_sender']
        self.sender_password = config['email_password']
        
        logger.info("User alert system initialized")
    
    def check_and_alert_users(self, event: Dict):
        """
        Check if earthquake affects any users and send alerts
        
        Args:
            event: Detected earthquake event with location info
        """
        if 'location' not in event:
            logger.warning("Event has no location data, skipping user alerts")
            return
        
        location = event['location']
        lat = location.get('epicenter_lat')
        lon = location.get('epicenter_lon')
        
        if not lat or not lon:
            return
        
        magnitude = event['magnitude']
        
        # Only alert for significant earthquakes (M >= 3.0)
        if magnitude < 3.0:
            return
        
        # Determine alert radius based on magnitude
        alert_radius = self._calculate_alert_radius(magnitude)
        
        # Find users in affected area
        affected_users = self.user_manager.get_users_in_radius(lat, lon, alert_radius)
        
        if not affected_users:
            logger.info(f"No users affected by M{magnitude:.1f} earthquake")
            return
        
        logger.info(f"Found {len(affected_users)} users within {alert_radius:.0f} km of M{magnitude:.1f} earthquake")
        
        # Send alerts to each affected user
        for user in affected_users:
            try:
                self._send_user_alert(user, event)
                
                # Log alert
                self.user_manager.log_alert_sent(
                    user['id'],
                    magnitude,
                    user['distance_km'],
                    'email'
                )
                
            except Exception as e:
                logger.error(f"Failed to send alert to {user['email']}: {e}")
    
    def _calculate_alert_radius(self, magnitude: float) -> float:
        """Calculate alert radius based on earthquake magnitude"""
        # Larger earthquakes = wider alert radius
        if magnitude >= 7.0:
            return 500  # 500 km
        elif magnitude >= 6.0:
            return 300  # 300 km
        elif magnitude >= 5.0:
            return 150  # 150 km
        elif magnitude >= 4.0:
            return 100  # 100 km
        else:
            return 50   # 50 km
    
    def _send_user_alert(self, user: Dict, event: Dict):
        """Send email alert to a specific user"""
        
        if not self.sender_email or not self.sender_password:
            logger.warning("Email credentials not configured, skipping alert")
            return
        
        # Prepare email content
        magnitude = event['magnitude']
        location = event['location']
        distance = user['distance_km']
        
        subject = f"🚨 Earthquake Alert: M{magnitude:.1f} detected {distance:.0f} km from {user['city']}"
        
        # Create HTML email body
        html_body = self._generate_alert_email(user, event, distance)
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = self.sender_email
        msg['To'] = user['email']
        msg['Subject'] = subject
        
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert sent to {user['username']} ({user['email']})")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    def _generate_alert_email(self, user: Dict, event: Dict, distance_km: float) -> str:
        """Generate HTML email content"""
        
        magnitude = event['magnitude']
        location = event['location']
        timestamp = event['timestamp']
        confidence = event['confidence']
        
        # Determine severity
        if magnitude >= 6.0:
            severity = "SEVERE"
            severity_color = "#d32f2f"
        elif magnitude >= 5.0:
            severity = "HIGH"
            severity_color = "#f57c00"
        elif magnitude >= 4.0:
            severity = "MODERATE"
            severity_color = "#fbc02d"
        else:
            severity = "LOW"
            severity_color = "#388e3c"
        
        # Generate precautions based on distance and magnitude
        precautions = self._get_precautions(magnitude, distance_km)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                }}
                .header {{
                    background: {severity_color};
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 8px 8px 0 0;
                }}
                .content {{
                    padding: 20px;
                    background: #f9f9f9;
                }}
                .alert-box {{
                    background: white;
                    padding: 15px;
                    margin: 15px 0;
                    border-left: 4px solid {severity_color};
                    border-radius: 4px;
                }}
                .detail {{
                    margin: 10px 0;
                }}
                .label {{
                    font-weight: bold;
                    color: #666;
                }}
                .precautions {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                }}
                .precautions h3 {{
                    margin-top: 0;
                    color: #856404;
                }}
                .precautions ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                .footer {{
                    padding: 15px;
                    text-align: center;
                    font-size: 12px;
                    color: #666;
                    background: #e0e0e0;
                    border-radius: 0 0 8px 8px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 EARTHQUAKE ALERT</h1>
                <h2>Severity: {severity}</h2>
            </div>
            
            <div class="content">
                <div class="alert-box">
                    <h3>Hello {user['username']},</h3>
                    <p>An earthquake has been detected near your location in <strong>{user['city']}, {user['country']}</strong>.</p>
                </div>
                
                <div class="alert-box">
                    <h3>Earthquake Details</h3>
                    <div class="detail">
                        <span class="label">Magnitude:</span> <strong style="color: {severity_color}; font-size: 18px;">M{magnitude:.1f}</strong>
                    </div>
                    <div class="detail">
                        <span class="label">Time:</span> {timestamp}
                    </div>
                    <div class="detail">
                        <span class="label">Distance from You:</span> <strong>{distance_km:.1f} km</strong>
                    </div>
                    <div class="detail">
                        <span class="label">Location:</span> {location['epicenter_lat']:.4f}°, {location['epicenter_lon']:.4f}°
                    </div>
                    <div class="detail">
                        <span class="label">Depth:</span> {location['depth_km']:.1f} km
                    </div>
                    <div class="detail">
                        <span class="label">Detection Confidence:</span> {confidence * 100:.0f}%
                    </div>
                    {f'<div class="detail"><span class="label">Near:</span> {location.get("nearest_zone", "Unknown")}</div>' if location.get("nearest_zone") else ''}
                </div>
                
                <div class="precautions">
                    <h3>⚠️ Safety Precautions</h3>
                    <ul>
                        {''.join(f'<li>{precaution}</li>' for precaution in precautions)}
                    </ul>
                </div>
                
                <div class="alert-box">
                    <p><strong>What to do now:</strong></p>
                    <ol>
                        <li>Stay calm and assess your situation</li>
                        <li>Check for injuries and hazards</li>
                        <li>Be prepared for aftershocks</li>
                        <li>Follow local emergency services guidance</li>
                        <li>Stay informed through official channels</li>
                    </ol>
                </div>
            </div>
            
            <div class="footer">
                <p>This is an automated alert from the Seismic AI Detector system.</p>
                <p>For official earthquake information, visit <a href="https://earthquake.usgs.gov/">USGS Earthquake Hazards Program</a></p>
                <p><small>You received this alert because you registered for earthquake notifications in your area.</small></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_precautions(self, magnitude: float, distance_km: float) -> List[str]:
        """Get appropriate safety precautions based on event characteristics"""
        
        precautions = []
        
        # Distance-based precautions
        if distance_km < 20:
            precautions.append("<strong>You are very close to the epicenter</strong> - strong shaking likely occurred or may occur")
            precautions.append("Drop, Cover, and Hold On immediately if shaking starts")
            precautions.append("Move away from windows, heavy furniture, and objects that could fall")
        elif distance_km < 50:
            precautions.append("You are near the epicenter - moderate to strong shaking is possible")
            precautions.append("Be prepared to Drop, Cover, and Hold On if you feel shaking")
        elif distance_km < 100:
            precautions.append("You are in the affected region - light to moderate shaking may occur")
            precautions.append("Stay alert and be ready to take cover")
        else:
            precautions.append("You are outside the main shaking area, but stay alert")
            precautions.append("Minor shaking might be felt in your location")
        
        # Magnitude-based precautions
        if magnitude >= 6.0:
            precautions.append("<strong>MAJOR EARTHQUAKE</strong> - Significant damage possible")
            precautions.append("Expect aftershocks - some may be strong")
            precautions.append("Check gas lines, electrical systems, and water pipes for damage")
            precautions.append("Do not use elevators")
            precautions.append("If indoors, stay indoors. If outdoors, stay outdoors")
        elif magnitude >= 5.0:
            precautions.append("<strong>STRONG EARTHQUAKE</strong> - Damage to buildings possible")
            precautions.append("Be aware of aftershocks")
            precautions.append("Check your surroundings for hazards")
        elif magnitude >= 4.0:
            precautions.append("MODERATE EARTHQUAKE - Minor damage possible")
            precautions.append("Check for cracks or structural issues")
        
        # General precautions
        precautions.append("Keep emergency supplies readily accessible (water, food, flashlight, first aid)")
        precautions.append("Identify safe spots in each room (under sturdy tables, against interior walls)")
        precautions.append("Have a communication plan with family members")
        precautions.append("Monitor local news and emergency broadcasts")
        
        # If very close and strong
        if distance_km < 50 and magnitude >= 5.0:
            precautions.append("<strong>EVACUATE damaged buildings immediately</strong>")
            precautions.append("Watch for falling debris and unstable structures")
            precautions.append("Stay away from damaged areas")
        
        return precautions