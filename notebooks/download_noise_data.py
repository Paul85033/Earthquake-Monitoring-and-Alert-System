"""
Automated Noise Data Downloader
Downloads background seismic noise from IRIS

Requirements:
    pip install obspy pandas

Usage:
    python notebooks/download_noise_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
except ImportError:
    print("ERROR: obspy not installed")
    print("Install with: pip install obspy")
    sys.exit(1)


def download_noise_samples(
    output_dir: Path,
    num_samples: int = 100,
    duration_minutes: int = 15,
    year: int = 2024
):
    """
    Download background noise samples from IRIS
    
    Args:
        output_dir: Where to save CSV files
        num_samples: Number of noise samples to download
        duration_minutes: Length of each sample in minutes
        year: Year to sample from
    """
    
    print("=" * 70)
    print("IRIS NOISE DATA DOWNLOADER")
    print("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Station list (high-quality global stations)
    stations = [
        ('IU', 'ANMO'),  # Albuquerque, New Mexico
        ('IU', 'HRV'),   # Harvard, Massachusetts
        ('IU', 'CCM'),   # Cathedral Cave, Missouri
        ('IU', 'COLA'),  # College, Alaska
        ('II', 'PFO'),   # Pinon Flat, California
        ('II', 'BFO'),   # Black Forest, Germany
        ('CI', 'CLC'),   # China Lake, California
        ('CI', 'PAS'),   # Pasadena, California
        ('G', 'CAN'),    # Canberra, Australia
        ('IU', 'TATO'),  # Taipei, Taiwan
    ]
    
    print(f"\n⚙️  Configuration:")
    print(f"   Samples:  {num_samples}")
    print(f"   Duration: {duration_minutes} minutes each")
    print(f"   Year:     {year}")
    print(f"   Stations: {len(stations)} available")
    
    # Generate random quiet times
    # Avoid major earthquake times by sampling late night hours
    print(f"\n📅 Generating random sampling times...")
    
    sampling_times = []
    for _ in range(num_samples * 2):  # Generate extra in case some fail
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe for all months
        hour = random.randint(2, 5)  # 2-5 AM (usually quietest)
        minute = random.randint(0, 59)
        
        try:
            time = UTCDateTime(f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00")
            sampling_times.append(time)
        except:
            continue
    
    random.shuffle(sampling_times)
    print(f"   Generated {len(sampling_times)} time slots")
    
    # Initialize IRIS client
    print("\n🌍 Connecting to IRIS data center...")
    try:
        client = Client("IRIS")
        print("   Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Download noise samples
    print(f"\n📥 Downloading noise samples...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    for time in sampling_times:
        if successful >= num_samples:
            break
        
        try:
            # Randomly select station
            network, station = random.choice(stations)
            
            print(f"\n[{successful+1}/{num_samples}] {network}.{station} at {time}")
            
            # Download waveform
            try:
                st = client.get_waveforms(
                    network=network,
                    station=station,
                    location="*",
                    channel="BHZ",  # Vertical component
                    starttime=time,
                    endtime=time + duration_minutes * 60
                )
            except Exception as e:
                print(f"    ❌ Download failed: {e}")
                failed += 1
                continue
            
            if len(st) == 0:
                print(f"    ⚠️  No data available")
                failed += 1
                continue
            
            # Process waveform
            trace = st[0]
            data = trace.data
            sample_rate = trace.stats.sampling_rate
            
            print(f"    Processing: {len(data)} samples at {sample_rate} Hz")
            
            # Check for data gaps or anomalies
            if len(data) < sample_rate * duration_minutes * 60 * 0.8:
                print(f"    ⚠️  Incomplete data (too short)")
                failed += 1
                continue
            
            # Check if data looks like noise (not an earthquake)
            max_amplitude = np.max(np.abs(data))
            if max_amplitude > np.median(np.abs(data)) * 100:
                print(f"    ⚠️  Suspicious large amplitude (possible earthquake)")
                failed += 1
                continue
            
            # Resample to 100 Hz if needed
            if sample_rate != 100:
                from scipy import signal as scipy_signal
                num_samples_new = int(len(data) * 100 / sample_rate)
                data = scipy_signal.resample(data, num_samples_new)
                print(f"    Resampled to 100 Hz")
            
            # Normalize to reasonable values
            data = data * 1e-7
            
            # Save to CSV
            filename = f"noise_{successful+1:03d}_{network}_{station}.csv"
            filepath = output_dir / filename
            
            df = pd.DataFrame({
                'acceleration': data
            })
            
            df.to_csv(filepath, index=False)
            
            print(f"    ✅ Saved: {filename}")
            successful += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Download interrupted by user")
            break
            
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
            failed += 1
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {successful + failed}")
    print(f"\nFiles saved to: {output_dir}")
    
    if successful > 0:
        print("\n✅ Download complete!")
        print("\nNext step:")
        print("  python notebooks/train_model_real_data.py")
    else:
        print("\n⚠️  No files downloaded")
        print("\nTroubleshooting:")
        print("- Check internet connection")
        print("- Try different year or stations")
        print("- Some stations may be temporarily offline")


def main():
    """Main function"""
    
    # Configuration
    output_dir = Path("data/samples/noise")
    num_samples = 100
    duration_minutes = 15
    
    print("\n⚙️  Starting noise data download...")
    
    download_noise_samples(
        output_dir=output_dir,
        num_samples=num_samples,
        duration_minutes=duration_minutes
    )


if __name__ == "__main__":
    main()