# Generic HRV Pipeline Container - Project Structure

```
generic-hrv-pipeline-container/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── LICENSE
├── .dockerignore
├── .gitignore
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   ├── setup.sh
│   └── test.sh
├── src/
│   ├── __init__.py
│   ├── hrv_pipeline.py      # Main generic HRV pipeline
│   ├── web_interface.py     # Optional web interface
│   └── utils/
│       ├── __init__.py
│       ├── signal_validators.py
│       ├── format_converters.py
│       └── helpers.py
├── config/
│   ├── hrv_config.yaml      # Default configuration
│   ├── docker_config.yaml  # Container-specific config
│   ├── ecg_config.yaml     # ECG-specific settings
│   ├── heartrate_config.yaml # Heart rate specific settings
│   └── rri_config.yaml     # RR intervals specific settings
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_input_detection.py
│   ├── test_signal_processing.py
│   ├── test_data/
│   │   ├── sample_ecg.csv
│   │   ├── sample_heartrate.csv
│   │   ├── sample_rri.csv
│   │   └── sample_rri_seconds.csv
│   └── conftest.py
├── data/                    # Mount point for input data
│   ├── ecg/                # Raw ECG files
│   ├── heartrate/          # Heart rate (BPM) files
│   ├── rr_intervals/       # RR interval files
│   └── mixed/              # Mixed format files (auto-detect)
├── output/                 # Mount point for results
├── cache/                  # Mount point for cache
└── logs/                   # Mount point for logs
```

## Directory Descriptions

- **`src/`**: Core application code for generic HRV processing
- **`config/`**: Configuration files for different input types and environments  
- **`scripts/`**: Helper scripts for building and running
- **`tests/`**: Comprehensive test suite with sample data for each input type
- **`data/`**: Input data directory organized by signal type (mounted as volume)
  - **`ecg/`**: Raw ECG signal files (.csv, .txt, .h5, etc.)
  - **`heartrate/`**: Heart rate/BPM files from wearables or monitors
  - **`rr_intervals/`**: Pre-calculated RR interval time series
  - **`mixed/`**: Files of mixed or unknown type (auto-detection enabled)
- **`output/`**: Results output directory (mounted as volume)
- **`cache/`**: Cache directory for performance (mounted as volume)
- **`logs/`**: Log files directory (mounted as volume)

## Supported Input Formats

### File Types
- **CSV**: Comma-separated values
- **TXT/TSV**: Tab or space-separated text files
- **Excel**: .xlsx, .xls files
- **Parquet**: Columnar data format
- **HDF5**: Hierarchical data format

### Signal Types
- **ECG**: Raw electrocardiogram signals (requires sampling rate)
- **Heart Rate**: BPM values from wearables, monitors, etc.
- **RR Intervals**: Pre-calculated inter-beat intervals (ms or seconds)
- **Auto-detect**: Automatically determine signal type from data characteristics