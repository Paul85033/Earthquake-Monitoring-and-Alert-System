#!/bin/bash

# Seismic AI Detector - Quick Start Script

echo "=================================="
echo "Seismic AI Detector"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if model exists
if [ ! -f "models/anomaly_model.pkl" ]; then
    echo ""
    echo "⚠️  ML model not found!"
    echo "Training model (this may take a minute)..."
    python notebooks/train_model.py
    echo ""
fi

# Create necessary directories
mkdir -p data/samples
mkdir -p data/logs
mkdir -p models

# Ask user what to run
echo ""
echo "What would you like to run?"
echo "1) Detection system only"
echo "2) Dashboard only"
echo "3) Both (in separate terminals)"
echo "4) Train new model"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting detection system..."
        echo "Press Ctrl+C to stop"
        echo ""
        cd src && python main.py
        ;;
    2)
        echo ""
        echo "Starting web dashboard..."
        echo "Open http://localhost:5000 in your browser"
        echo "Press Ctrl+C to stop"
        echo ""
        cd dashboard && python app.py
        ;;
    3)
        echo ""
        echo "This requires multiple terminals."
        echo ""
        echo "Terminal 1: cd src && python main.py"
        echo "Terminal 2: cd dashboard && python app.py"
        echo ""
        echo "Starting detection system in this terminal..."
        echo "Open another terminal and run: cd dashboard && python app.py"
        echo ""
        cd src && python main.py
        ;;
    4)
        echo ""
        echo "Training new model..."
        python notebooks/train_model.py
        echo ""
        echo "Model training complete!"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac