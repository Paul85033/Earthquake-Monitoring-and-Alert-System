"""
Flask web dashboard for real-time earthquake monitoring
"""

from flask import Flask, render_template, jsonify
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import DATABASE_PATH, DASHBOARD_CONFIG
from src.database import SeismicDatabase

app = Flask(__name__)
db = SeismicDatabase(str(DATABASE_PATH))


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


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
    print(f"\n🌐 Dashboard running at: http://{DASHBOARD_CONFIG['host']}:{DASHBOARD_CONFIG['port']}")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(
        host=DASHBOARD_CONFIG['host'],
        port=DASHBOARD_CONFIG['port'],
        debug=DASHBOARD_CONFIG['debug']
    )