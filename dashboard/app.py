"""
Flask web dashboard for real-time earthquake monitoring
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
import sys
from pathlib import Path
from datetime import datetime, timedelta
import secrets

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import DATABASE_PATH, DASHBOARD_CONFIG, ALERT_CONFIG
from src.database import SeismicDatabase
from src.multi_location import MultiLocationPredictor
from src.auth import UserManager
from src.password_reset import PasswordResetEmailer

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # For session management

db = SeismicDatabase(str(DATABASE_PATH))
predictor = MultiLocationPredictor()
user_manager = UserManager(str(DATABASE_PATH))
password_reset_emailer = PasswordResetEmailer(ALERT_CONFIG)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        city = request.form.get('city')
        country = request.form.get('country')
        
        # Validate
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        # Register user
        result = user_manager.register_user(username, email, password, city, country)
        
        if result['success']:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result['message'], 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        result = user_manager.login_user(username, password)
        
        if result['success']:
            session['session_token'] = result['session_token']
            session['user'] = result['user']
            flash(f"Welcome back, {username}!", 'success')
            return redirect(url_for('index'))
        else:
            flash(result['message'], 'error')
            return render_template('login.html')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/profile')
def profile():
    """User profile page"""
    if 'user' not in session:
        flash('Please log in to view your profile', 'error')
        return redirect(url_for('login'))
    
    user = session['user']
    
    # Get alert history
    conn = db.db_path
    import sqlite3
    conn = sqlite3.connect(conn)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM alert_history 
        WHERE user_id = ? 
        ORDER BY sent_at DESC 
        LIMIT 10
    ''', (user['id'],))
    
    alerts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return render_template('profile.html', user=user, alerts=alerts)


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password page"""
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Please enter your email address', 'error')
            return render_template('forgot_password.html')
        
        # Create reset token
        result = user_manager.create_password_reset_token(email)
        
        if result['success'] and 'token' in result:
            # Send reset email
            password_reset_emailer.send_reset_email(result['user'], result['token'])
        
        # Always show success message (don't reveal if email exists)
        flash('If that email is registered, you will receive a password reset link shortly.', 'success')
        return render_template('forgot_password.html')
    
    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Reset password page"""
    token = request.args.get('token') or request.form.get('token')
    
    if not token:
        flash('Invalid reset link', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate
        if not new_password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return render_template('reset_password.html', token=token)
        
        if new_password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('reset_password.html', token=token)
        
        # Reset password
        result = user_manager.reset_password(token, new_password)
        
        if result['success']:
            flash('Password reset successful! Please log in with your new password.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result['message'], 'error')
            return render_template('reset_password.html', token=token)
    
    # Verify token is valid
    user_data = user_manager.verify_reset_token(token)
    if not user_data:
        flash('This reset link is invalid or has expired. Please request a new one.', 'error')
        return redirect(url_for('forgot_password'))
    
    return render_template('reset_password.html', token=token, username=user_data['username'])


@app.route('/api/events/recent')
def get_recent_events():
    """Get recent earthquake events"""
    limit = 20
    events = db.get_recent_events(limit)
    return jsonify(events)


@app.route('/api/events/today')
def get_today_events():
    """Get today's events"""
    today = datetime.now().date()
    start = f"{today}T00:00:00"
    end = f"{today}T23:59:59"
    
    events = db.get_events_by_date(start, end)
    return jsonify(events)


@app.route('/api/predictions')
def get_predictions():
    """Get earthquake predictions for all monitored regions"""
    
    # Get most recent detected event (for triggering analysis)
    recent_events = db.get_recent_events(1)
    latest_event = recent_events[0] if recent_events else None
    
    # Generate predictions
    predictions = predictor.predict_all_locations(latest_event)
    
    return jsonify(predictions)


@app.route('/api/predictions/<region_id>')
def get_region_prediction(region_id):
    """Get prediction for specific region"""
    predictions = predictor.predict_all_locations()
    
    for pred in predictions:
        if pred['region_id'] == region_id:
            return jsonify(pred)
    
    return jsonify({'error': 'Region not found'}), 404


@app.route('/api/stats')
def get_statistics():
    """Get overall statistics"""
    stats = db.get_statistics()
    
    # Add current time
    stats['current_time'] = datetime.now().isoformat()
    
    return jsonify(stats)


@app.route('/api/events/magnitude/<float:min_mag>')
def get_events_by_magnitude(min_mag):
    """Get events above minimum magnitude"""
    events = db.get_recent_events(100)
    filtered = [e for e in events if e['magnitude'] >= min_mag]
    return jsonify(filtered)


@app.route('/api/timeline')
def get_timeline():
    """Get event timeline for charts"""
    events = db.get_recent_events(50)
    
    timeline = []
    for event in reversed(events):
        timeline.append({
            'time': event['timestamp'],
            'magnitude': event['magnitude'],
            'confidence': event['confidence']
        })
    
    return jsonify(timeline)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 70)
    print("SEISMIC AI DETECTOR - WEB DASHBOARD")
    print("=" * 70)
    print(f"\n Dashboard running at: http://{DASHBOARD_CONFIG['host']}:{DASHBOARD_CONFIG['port']}")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(
        host=DASHBOARD_CONFIG['host'],
        port=DASHBOARD_CONFIG['port'],
        debug=DASHBOARD_CONFIG['debug']
    )