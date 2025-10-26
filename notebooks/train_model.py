"""
ML Model Training Script
Train earthquake detection model and save to models/
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODEL_PATH, TRAINING_CONFIG


class TrainingDataGenerator:
    """Generate synthetic training data"""
    
    def __init__(self, sample_rate: int = 100):
        self.sample_rate = sample_rate
    
    def generate_dataset(self, num_earthquakes: int, num_noise: int):
        """Generate complete training dataset"""
        print(f"Generating {num_earthquakes} earthquake samples...")
        X_eq = np.array([
            self._extract_features(self._generate_earthquake())
            for _ in range(num_earthquakes)
        ])
        
        print(f"Generating {num_noise} noise samples...")
        X_noise = np.array([
            self._extract_features(self._generate_noise())
            for _ in range(num_noise)
        ])
        
        X = np.vstack([X_eq, X_noise])
        y = np.hstack([np.ones(num_earthquakes), np.zeros(num_noise)])
        
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _generate_earthquake(self):
        """Generate earthquake waveform"""
        magnitude = np.random.uniform(2.0, 7.0)
        duration = np.random.uniform(10, 40)
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        amplitude = 10 ** (magnitude - 3)
        
        # Envelope
        onset = np.random.uniform(1.0, 3.0)
        decay = np.random.uniform(4.0, 8.0)
        envelope = np.exp(-((t - onset) / decay) ** 2) * (t >= onset)
        envelope *= np.exp(-(t - onset) / (duration * 0.5)) * (t >= onset)
        
        # Waves
        f_p = np.random.uniform(3, 7)
        f_s = np.random.uniform(1, 3)
        p_wave = np.sin(2 * np.pi * f_p * t)
        s_wave = np.sin(2 * np.pi * f_s * t + np.random.uniform(0, np.pi))
        
        waveform = amplitude * envelope * (0.6 * p_wave + 0.4 * s_wave)
        noise = np.random.normal(0, amplitude * 0.08, num_samples)
        
        return waveform + noise
    
    def _generate_noise(self):
        """Generate noise waveform"""
        duration = np.random.uniform(5, 30)
        num_samples = int(duration * self.sample_rate)
        
        # Background noise
        waveform = np.random.normal(0, 0.02, num_samples)
        
        # Random spikes
        num_events = np.random.randint(1, 5)
        for _ in range(num_events):
            pos = np.random.randint(0, num_samples - 50)
            width = np.random.randint(5, 30)
            amplitude = np.random.uniform(0.05, 0.3)
            waveform[pos:pos+width] += amplitude * np.exp(-np.linspace(0, 3, width))
        
        # Drift
        drift = 0.01 * np.sin(2 * np.pi * 0.1 * np.linspace(0, duration, num_samples))
        
        return waveform + drift
    
    def _extract_features(self, waveform):
        """Extract features from waveform"""
        features = []
        
        # Time domain
        features.append(np.max(np.abs(waveform)))  # PGA
        features.append(np.mean(np.abs(waveform)))  # Mean amplitude
        features.append(np.std(waveform))  # Std
        features.append(np.sum(waveform ** 2))  # Energy
        
        # Statistical
        mean = np.mean(waveform)
        std = np.std(waveform)
        if std > 1e-10:
            features.append(np.mean(((waveform - mean) / std) ** 4) - 3)  # Kurtosis
            features.append(np.mean(((waveform - mean) / std) ** 3))  # Skewness
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Frequency domain
        fft = np.fft.fft(waveform)
        freqs = np.fft.fftfreq(len(waveform), 1/self.sample_rate)
        positive_idx = freqs > 0
        magnitude = np.abs(fft[positive_idx])
        positive_freqs = freqs[positive_idx]
        
        if np.sum(magnitude) > 0:
            features.append(np.sum(positive_freqs * magnitude) / np.sum(magnitude))
            features.append(positive_freqs[np.argmax(magnitude)])
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Zero crossings
        features.append(np.sum(np.diff(np.sign(waveform)) != 0) / len(waveform))
        
        # Duration
        features.append(len(waveform) / self.sample_rate)
        
        return features


def train_model():
    """Train Random Forest model"""
    print("=" * 70)
    print("EARTHQUAKE DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Generate training data
    print("\n[1] Generating Training Data")
    print("-" * 70)
    
    generator = TrainingDataGenerator()
    X, y = generator.generate_dataset(
        num_earthquakes=TRAINING_CONFIG['num_earthquakes'],
        num_noise=TRAINING_CONFIG['num_noise_samples']
    )
    
    # Split data
    print("\n[2] Splitting Data")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state'],
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    
    # Scale features
    print("\n[3] Scaling Features")
    print("-" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled")
    
    # Train model
    print("\n[4] Training Random Forest")
    print("-" * 70)
    
    model = RandomForestClassifier(
        n_estimators=TRAINING_CONFIG['n_estimators'],
        max_depth=TRAINING_CONFIG['max_depth'],
        min_samples_split=TRAINING_CONFIG['min_samples_split'],
        random_state=TRAINING_CONFIG['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("Model trained successfully")
    
    # Evaluate
    print("\n[5] Evaluation")
    print("-" * 70)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Training Accuracy: {train_score:.2%}")
    print(f"Testing Accuracy:  {test_score:.2%}")
    print(f"ROC-AUC Score:     {auc_score:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Earthquake']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"            Noise  Earthquake")
    print(f"Actual Noise  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"    Earthquake {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = [
        'PGA', 'Mean Amp', 'Std Amp', 'Energy',
        'Kurtosis', 'Skewness', 'Mean Freq', 'Dom Freq',
        'Zero Cross', 'Duration'
    ]
    
    for name, importance in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {name:12s}: {importance:.3f}")
    
    # Save model
    print("\n[6] Saving Model")
    print("-" * 70)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'model_type': 'random_forest',
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': auc_score
        }
    }
    
    joblib.dump(model_package, MODEL_PATH)
    
    print(f"Model saved to: {MODEL_PATH}")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run main.py to start real-time detection")
    print("2. Check dashboard/app.py for web interface")
    print("3. Monitor logs in data/logs/")


if __name__ == "__main__":
    train_model()