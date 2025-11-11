"""
Password Reset Email System
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)


class PasswordResetEmailer:
    """Send password reset emails"""
    
    def __init__(self, config: dict):
        self.config = config
        self.smtp_server = config['email_smtp_server']
        self.smtp_port = config['email_smtp_port']
        self.sender_email = config['email_sender']
        self.sender_password = config['email_password']
        self.base_url = config.get('base_url', 'http://localhost:5000')
        
        logger.info("Password reset emailer initialized")
    
    def send_reset_email(self, user: dict, token: str) -> bool:
        """
        Send password reset email to user
        
        Args:
            user: User dictionary with email, username
            token: Password reset token
        
        Returns:
            True if sent successfully, False otherwise
        """
        
        if not self.sender_email or not self.sender_password:
            logger.warning("Email credentials not configured")
            return False
        
        reset_link = f"{self.base_url}/reset-password?token={token}"
        
        subject = "Password Reset Request - Seismic AI Detector"
        
        html_body = self._generate_reset_email(user, reset_link)
        
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
            
            logger.info(f"Password reset email sent to {user['email']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send reset email: {e}")
            return False
    
    def _generate_reset_email(self, user: dict, reset_link: str) -> str:
        """Generate HTML email content"""
        
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
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                    border-radius: 8px 8px 0 0;
                }}
                .content {{
                    padding: 30px 20px;
                    background: #f9f9f9;
                }}
                .box {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                }}
                .button {{
                    display: inline-block;
                    padding: 15px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white !important;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                    margin: 20px 0;
                }}
                .button:hover {{
                    opacity: 0.9;
                }}
                .warning {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 6px;
                    color: #856404;
                }}
                .code {{
                    background: #f4f4f4;
                    padding: 10px;
                    border-radius: 4px;
                    font-family: monospace;
                    word-break: break-all;
                    margin: 10px 0;
                }}
                .footer {{
                    padding: 20px;
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
                <h1>Password Reset Request</h1>
            </div>
            
            <div class="content">
                <div class="box">
                    <h2>Hello {user['username']},</h2>
                    <p>We received a request to reset your password for your Seismic AI Detector account.</p>
                    <p>If you made this request, click the button below to reset your password:</p>
                    
                    <div style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset My Password</a>
                    </div>
                    
                    <p style="margin-top: 20px;">Or copy and paste this link into your browser:</p>
                    <div class="code">{reset_link}</div>
                </div>
                
                <div class="warning">
                    <strong>⚠️ Important Security Information:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>This link will expire in <strong>1 hour</strong></li>
                        <li>This link can only be used <strong>once</strong></li>
                        <li>If you didn't request this, please ignore this email</li>
                        <li>Your password will not change unless you click the link above</li>
                    </ul>
                </div>
                
                <div class="box">
                    <h3>Security Tips:</h3>
                    <ul>
                        <li>Never share your password with anyone</li>
                        <li>Use a strong, unique password</li>
                        <li>Enable two-factor authentication if available</li>
                        <li>Be cautious of phishing emails</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Seismic AI Detector</strong></p>
                <p>This is an automated email. Please do not reply.</p>
                <p>If you didn't request a password reset, you can safely ignore this email.</p>
                <p style="margin-top: 15px;">
                    <small>Having trouble? Contact support or visit our website for help.</small>
                </p>
            </div>
        </body>
        </html>
        """
        
        return html