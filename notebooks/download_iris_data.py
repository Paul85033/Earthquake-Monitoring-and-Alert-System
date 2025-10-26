"""
Automated IRIS Data Downloader
Downloads earthquake waveforms using USGS catalog

Requirements:
    pip install obspy pandas

Usage:
    python notebooks/download_iris_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
except ImportError:
    print("ERROR: obspy not installed")
    print("Install with: pip install obspy")
    sys.exit(1)


def download_earthquake_waveforms(
    catalog_path: str,
    output_dir: Path,
    max_events: int = 50,
    min_magnitude: float = 3.5,
    max_radius_deg: float = 5.0
):
    """
    Download earthquake waveforms from IRIS
    
    Args:
        catalog_path: Path to USGS CSV catalog
        output_dir: Where to save CSV files
        max_events: Maximum number of events to download
        min_magnitude: Minimum earthquake magnitude
        max_radius_deg: Maximum station distance in degrees (~111 km per degree)
    """
    
    print("=" * 70)
    print("IRIS EARTHQUAKE DATA DOWNLOADER")
    print("=" * 70)
    
    # Check if catalog exists
    if not Path(catalog_path).exists():
        print(f"\n❌ ERROR: Catalog not found: {catalog_path}")
        print("\nPlease download USGS catalog first:")
        print("1. Visit: https://earthquake.usgs.gov/earthquakes/search/")
        print("2. Set date range and magnitude >= 4.0")
        print("3. Download as CSV")
        print(f"4. Save to: {catalog_path}")
        return
    
    # Load catalog
    print(f"\n📂 Loading catalog: {catalog_path}")
    try:
        catalog = pd.read_csv(catalog_path)
        print(f"   Found {len(catalog)} events in catalog")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Filter by magnitude
    catalog = catalog[catalog['mag'] >= min_magnitude]
    print(f"   Filtered to {len(catalog)} events (mag >= {min_magnitude})")
    
    # Limit number of events
    if len(catalog) > max_events:
        catalog = catalog.head(max_events)
        print(f"   Limited to {max_events} events")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize IRIS client
    print("\n🌍 Connecting to IRIS data center...")
    try:
        client = Client("IRIS")
        print("   Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Download waveforms
    print(f"\n📥 Downloading waveforms...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    for idx, event in catalog.iterrows():
        try:
            # Parse event info
            event_time = UTCDateTime(event['time'])
            lat = event['latitude']
            lon = event['longitude']
            mag = event['mag']
            depth = event.get('depth', 0)
            place = event.get('place', 'Unknown')
            
            print(f"\n[{idx+1}/{len(catalog)}] M{mag:.1f} - {place[:40]}")
            print(f"    Time: {event_time}")
            print(f"    Location: {lat:.2f}, {lon:.2f}, depth {depth:.1f} km")
            
            # Find nearby stations
            print(f"    Searching for stations...")
            try:
                inventory = client.get_stations(
                    starttime=event_time - 60,
                    endtime=event_time + 360,
                    latitude=lat,
                    longitude=lon,
                    maxradius=max_radius_deg,
                    channel="BH?",  # Broadband high-gain
                    level="station"
                )
            except Exception as e:
                print(f"    ❌ Station search failed: {e}")
                failed += 1
                continue
            
            if len(inventory.networks) == 0:
                print(f"    ⚠️  No stations found within {max_radius_deg}°")
                failed += 1
                continue
            
            # Get first available station
            network = inventory.networks[0]
            station = network.stations[0]
            
            print(f"    Found station: {network.code}.{station.code}")
            
            # Download waveform (vertical component)
            print(f"    Downloading waveform...")
            try:
                st = client.get_waveforms(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel="BHZ",  # Vertical component
                    starttime=event_time - 30,  # 30 sec before
                    endtime=event_time + 270    # 4.5 min after
                )
            except Exception as e:
                print(f"    ❌ Waveform download failed: {e}")
                failed += 1
                continue
            
            if len(st) == 0:
                print(f"    ⚠️  No waveform data available")
                failed += 1
                continue
            
            # Process waveform
            trace = st[0]
            data = trace.data
            sample_rate = trace.stats.sampling_rate
            
            print(f"    Processing: {len(data)} samples at {sample_rate} Hz")
            
            # Resample to 100 Hz if needed
            if sample_rate != 100:
                from scipy import signal as scipy_signal
                num_samples = int(len(data) * 100 / sample_rate)
                data = scipy_signal.resample(data, num_samples)
                print(f"    Resampled to 100 Hz")
            
            # Normalize to reasonable acceleration values
            # Convert to approximate m/s²
            data = data * 1e-7  # Rough conversion factor
            
            # Save to CSV
            filename = f"eq_{successful+1:03d}_M{mag:.1f}_{network.code}_{station.code}.csv"
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
        print("  python notebooks/download_noise_data.py")
    else:
        print("\n⚠️  No files downloaded")
        print("\nTroubleshooting:")
        print("- Check internet connection")
        print("- Try different date range in USGS catalog")
        print("- Increase max_radius_deg parameter")


def main():
    """Main function"""
    
    # Configuration
    catalog_path = "data/samples/usgs_catalog/usgs_earthquakes.csv"
    output_dir = Path("data/samples/earthquakes")
    max_events = 50
    min_magnitude = 3.5
    
    print("\n⚙️  Configuration:")
    print(f"   Catalog: {catalog_path}")
    print(f"   Output:  {output_dir}")
    print(f"   Events:  {max_events}")
    print(f"   Min mag: {min_magnitude}")
    
    download_earthquake_waveforms(
        catalog_path=catalog_path,
        output_dir=output_dir,
        max_events=max_events,
        min_magnitude=min_magnitude
    )


if __name__ == "__main__":
    main()