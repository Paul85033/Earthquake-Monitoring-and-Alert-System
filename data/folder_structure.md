data/
├── logs/ # Runtime or sensor logs generated during execution
│ ├── system.log # Example: general runtime logs
│ ├── detector.log # Example: model or detector-related logs
│ └── ...  
│
├── samples/ # Example/sample data for testing and development
│ ├── earthquakes/ # Sample earthquake waveform files
│ ├── noise/ # Sample background/noise waveform files
│ └── usgs_catalog/ # Sample or cached data from USGS (metadata, events)
│
├── seismic_log.db # Local SQLite database for storing seismic detections
│
└── README.md # This documentation file
