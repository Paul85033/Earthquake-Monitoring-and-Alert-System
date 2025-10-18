"""
Alert system for earthquake notifications
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AlertSystem:
    """Multi-channel alert system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enable_email = config.get('enable_email', False)
        self.enable_sms = config.get('enable_sms', False)
        self.enable_console = config.get('enable_console', True)
        self.enable_webhook = config.get('enable_webhook', False)
        
        logger.info("Alert system initialized")
    
    def send_alert(self, event: Dict):
        """Send alert through all enabled channels"""
        
        if self.enable_console:
            self._console_alert(event)
        
        if self.enable_email:
            try:
                self._email_alert(event)
            except Exception as e:
                logger.error(f"Email alert failed: {e}")
        
        if self.enable_sms:
            try:
                self._sms_alert(event)
            except Exception as e:
                logger.error(f"SMS alert failed: {e}")
        
        if self.enable_webhook:
            try:
                self._webhook_alert(event)
            except Exception as e:
                logger.error(f"Webhook alert failed: {e}")
    
    def _console_alert(self, event: Dict):
        """Print alert to console"""
        print("\n" + "=" * 70)
        print("🚨 EARTHQUAKE ALERT 🚨")
        print("=" * 70)
        print(f"Time:       {event['timestamp']}")
        print(f"Magnitude:  {event['magnitude']:.1f}")
        print(f"Duration:   {event['duration']:.1f} seconds")
        print(f"Confidence: {event['confidence']:.1%}")
        print(f"PGA:        {event['pga']:.3f} m/s²")
        print("=" * 70)
        print()
    
    def _email_alert(self, event: Dict):
        """Send email alert"""
        sender = self.config['email_sender']
        password = self.config['email_password']
        recipients = self.config['email_recipients']
        
        if not sender or not password or not recipients:
            logger.warning("Email credentials not configured")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"🚨 Earthquake Alert - Magnitude {event['magnitude']:.1f}"
        
        body = f"""
Earthquake Detection Alert

Time: {event['timestamp']}
Magnitude: {event['magnitude']:.1f}
Duration: {event['duration']:.1f} seconds
Confidence: {event['confidence']:.1%}
Peak Ground Acceleration: {event['pga']:.3f} m/s²

This is an automated alert from the Seismic AI Detector.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(
            self.config['email_smtp_server'],
            self.config['email_smtp_port']
        )
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent to {len(recipients)} recipient(s)")
    
    def _sms_alert(self, event: Dict):
        """Send SMS alert via Twilio"""
        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio not installed. Run: pip install twilio")
            return
        
        account_sid = self.config['sms_account_sid']
        auth_token = self.config['sms_auth_token']
        from_number = self.config['sms_from_number']
        to_numbers = self.config['sms_to_numbers']
        
        if not account_sid or not auth_token or not from_number:
            logger.warning("SMS credentials not configured")
            return
        
        client = Client(account_sid, auth_token)
        
        message_body = (
            f"🚨 Earthquake Alert!\n"
            f"Magnitude: {event['magnitude']:.1f}\n"
            f"Time: {event['timestamp']}\n"
            f"Confidence: {event['confidence']:.0%}"
        )
        
        for to_number in to_numbers:
            if to_number:
                message = client.messages.create(
                    body=message_body,
                    from_=from_number,
                    to=to_number
                )
                logger.info(f"SMS sent to {to_number}: {message.sid}")
    
    def _webhook_alert(self, event: Dict):
        """Send webhook notification"""
        webhook_url = self.config['webhook_url']
        
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return
        
        payload = {
            'event_type': 'earthquake_detected',
            'timestamp': event['timestamp'],
            'magnitude': event['magnitude'],
            'duration': event['duration'],
            'confidence': event['confidence'],
            'pga': event['pga']
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("Webhook notification sent successfully")
        else:
            logger.error(f"Webhook failed: {response.status_code}")


class HardwareAlert:
    """Hardware-based alerts (LED, buzzer, etc.)"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.gpio_available = False
        self.GPIO = None

        if not enabled:
            logger.info("Hardware alerts disabled by configuration")
            return

        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_available = True
            self._setup_gpio()
            logger.info("Hardware alerts enabled using real GPIO")
        except (ImportError, RuntimeError):
            # Try to fall back to a mock GPIO so the rest of the code still runs
            try:
                from fake_rpi import RPi as GPIO  # pip install fake-rpi
                self.GPIO = GPIO
                self.gpio_available = True
                self._setup_gpio()
                logger.info("Using fake_rpi GPIO emulator (not on Raspberry Pi)")
            except ImportError:
                self.gpio_available = False
                logger.warning("GPIO not available — hardware alerts disabled")

    
    def _setup_gpio(self):
        """Setup GPIO pins for LED and buzzer"""
        self.GPIO.setmode(self.GPIO.BCM)
        self.LED_PIN = 17  # GPIO 17 for LED
        self.BUZZER_PIN = 27  # GPIO 27 for buzzer
        
        self.GPIO.setup(self.LED_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.BUZZER_PIN, self.GPIO.OUT)
    
    def trigger_alert(self, duration: float = 5.0):
        """Trigger hardware alert"""
        if not self.gpio_available:
            return
        
        import time
        
        # Flash LED and sound buzzer
        for _ in range(int(duration)):
            self.GPIO.output(self.LED_PIN, self.GPIO.HIGH)
            self.GPIO.output(self.BUZZER_PIN, self.GPIO.HIGH)
            time.sleep(0.5)
            
            self.GPIO.output(self.LED_PIN, self.GPIO.LOW)
            self.GPIO.output(self.BUZZER_PIN, self.GPIO.LOW)
            time.sleep(0.5)
        
        logger.info("Hardware alert triggered")
    
    def cleanup(self):
        """Cleanup GPIO pins"""
        if self.gpio_available:
            self.GPIO.cleanup()