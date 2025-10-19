# Earthquake monitoring and alert system

Real-time earthquake detection system.

## Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python notebooks/train_model.py
```

Creates `models/anomaly_model.pkl`.

### 3. Run the System

**Option A: Detection Only**

```bash
cd src
python main.py
```

**Option B: Detection + Dashboard**

Terminal 1:

```bash
cd src
python main.py
```

Terminal 2:

```bash
cd dashboard
python app.py

or

python -m dashboard.app

```

from root

Then open: **http://localhost:5000**

## Expectation:

- **Console**: Real-time detection with STA/LTA ratios and earthquake alerts
- **Dashboard**: Live charts, statistics, and event history

## Configuration

Edit `src/config.py` to customize:

- Detection sensitivity
- Alert channels (email, SMS, webhook)
- Sensor type (simulated, serial, API)

For email/SMS alerts, copy `.env.example` to `.env` and add credentials.
