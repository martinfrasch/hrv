#!/usr/bin/env python3
"""
Generic HRV Pipeline - Production-ready implementation
Supports RR intervals, heart rate, and raw ECG time series data
Author: Optimized version based on Martin Frasch's HRV pipeline
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Union
import logging
import time
import hashlib
import joblib
import yaml
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class HRVConfig:
    """Configuration class for generic HRV pipeline parameters"""
    
    # Input data parameters
    input_type: str = "auto"  # "rr_intervals", "heart_rate", "ecg", "auto"
    sampling_rate: int = 1000  # For ECG signals (Hz)
    time_unit: str = "ms"  # "ms", "s" for RR intervals
    
    # Data validation parameters
    min_heart_rate: float = 30.0
    max_heart_rate: float = 200.0
    min_rr_interval: float = 300.0  # ms
    max_rr_interval: float = 2000.0  # ms
    
    # Processing parameters
    artifact_threshold: float = 0.2
    window_length: Optional[int] = None  # seconds, None = process entire signal
    window_overlap: float = 0.5
    min_signal_length: int = 120  # minimum seconds for reliable HRV
    
    # HRV computation parameters
    hrv_time_domain: bool = True
    hrv_frequency_domain: bool = True
    hrv_nonlinear_domain: bool = True
    hrv_higher_order: bool = True  # Temporal fluctuation analysis
    
    # Performance settings
    n_workers: Optional[int] = None
    chunk_size: int = 10000
    enable_caching: bool = True
    cache_dir: str = "./hrv_cache"
    
    # Output settings
    save_intermediate: bool = False
    output_format: str = "parquet"  # parquet, csv, hdf5
    include_metadata: bool = True
    
    # Advanced processing options
    interpolation_method: str = "linear"  # for missing data
    detrending_method: str = "loess"  # for trend removal
    outlier_removal: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'HRVConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.__dict__.copy()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# ============================================================================
# Data Processing Classes
# ============================================================================

class ProcessingStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProcessingResult:
    """Results container for file processing"""
    file_path: str
    status: ProcessingStatus
    input_type: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    n_samples: int = 0
    signal_duration: float = 0.0  # in seconds

class InputTypeDetector:
    """Automatically detect input data type and format"""
    
    @staticmethod
    def detect_input_type(data: pd.DataFrame, config: HRVConfig) -> str:
        """Detect whether input is RR intervals, heart rate, or ECG"""
        
        if data.empty or data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns (time, signal)")
        
        # Get the signal column (assume second column)
        signal = data.iloc[:, 1].dropna()
        
        if len(signal) == 0:
            raise ValueError("Signal column contains no valid data")
        
        # Statistical analysis of the signal
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        # Decision logic based on signal characteristics
        if 30 <= mean_val <= 200 and std_val < 50:
            # Likely heart rate (BPM)
            return "heart_rate"
        elif 300 <= mean_val <= 2000 and std_val > 50:
            # Likely RR intervals in milliseconds
            return "rr_intervals"
        elif -5 <= mean_val <= 5 and std_val > 0.1:
            # Likely ECG signal (mV range)
            return "ecg"
        elif 0.3 <= mean_val <= 2.0 and std_val > 0.05:
            # Likely RR intervals in seconds
            return "rr_intervals"
        else:
            # Fall back to heart rate assumption
            logging.warning(f"Could not definitively detect input type. "
                          f"Signal stats: mean={mean_val:.2f}, std={std_val:.2f}, "
                          f"range=[{min_val:.2f}, {max_val:.2f}]. "
                          f"Assuming heart rate.")
            return "heart_rate"
    
    @staticmethod
    def validate_input_format(data: pd.DataFrame, detected_type: str, config: HRVConfig) -> pd.DataFrame:
        """Validate and standardize input data format"""
        
        if data.shape[1] < 2:
            raise ValueError("Input data must have at least 2 columns")
        
        # Ensure we have time and signal columns
        if data.columns.tolist() == [0, 1]:  # Default integer column names
            if detected_type == "ecg":
                data.columns = ["time", "ecg"]
            elif detected_type == "heart_rate":
                data.columns = ["time", "heart_rate"]
            else:  # rr_intervals
                data.columns = ["time", "rr_intervals"]
        
        # Remove rows with invalid data
        data = data.dropna()
        
        # Validate signal values based on detected type
        signal_col = data.iloc[:, 1]
        
        if detected_type == "heart_rate":
            valid_mask = (signal_col >= config.min_heart_rate) & (signal_col <= config.max_heart_rate)
        elif detected_type == "rr_intervals":
            if config.time_unit == "s":
                # Convert to milliseconds for internal processing
                signal_col = signal_col * 1000
            valid_mask = (signal_col >= config.min_rr_interval) & (signal_col <= config.max_rr_interval)
            data.iloc[:, 1] = signal_col  # Update with converted values
        else:  # ECG
            # For ECG, we'll validate after peak detection
            valid_mask = pd.Series([True] * len(signal_col))
        
        # Apply validation mask
        data = data[valid_mask].copy()
        
        if len(data) < 100:  # Minimum samples required
            raise ValueError(f"Insufficient valid data points after cleaning: {len(data)}")
        
        return data

class GenericDataLoader:
    """Load and process various cardiovascular time series formats"""
    
    @staticmethod
    def load_time_series_file(file_path: str, config: HRVConfig) -> Tuple[pd.DataFrame, str]:
        """Load time series file and detect format"""
        
        file_path = Path(file_path)
        
        # Support multiple file formats
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.txt', '.tsv']:
            # Try different separators
            try:
                data = pd.read_csv(file_path, sep='\t')
            except:
                try:
                    data = pd.read_csv(file_path, sep=' ')
                except:
                    data = pd.read_csv(file_path, sep=',')
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.h5':
            # Assume first dataset in HDF5 file
            data = pd.read_hdf(file_path)
        else:
            # Default to CSV
            data = pd.read_csv(file_path)
        
        # Auto-detect input type if configured
        if config.input_type == "auto":
            detected_type = InputTypeDetector.detect_input_type(data, config)
        else:
            detected_type = config.input_type
        
        # Validate and standardize format
        data = InputTypeDetector.validate_input_format(data, detected_type, config)
        
        return data, detected_type
    
    @staticmethod
    def load_chunked(file_path: str, config: HRVConfig, chunk_size: int = 10000) -> Generator[Tuple[pd.DataFrame, str], None, None]:
        """Load large files in chunks"""
        try:
            # For chunked loading, we need to detect type from first chunk
            first_chunk = pd.read_csv(file_path, nrows=1000)
            detected_type = InputTypeDetector.detect_input_type(first_chunk, config) if config.input_type == "auto" else config.input_type
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                validated_chunk = InputTypeDetector.validate_input_format(chunk, detected_type, config)
                yield validated_chunk, detected_type
                
        except Exception as e:
            raise IOError(f"Failed to load file {file_path}: {e}")

# ============================================================================
# Signal Processing and HRV Computation
# ============================================================================

class SignalProcessor:
    """Process different types of cardiovascular signals"""
    
    def __init__(self, config: HRVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_signal_to_rri(self, data: pd.DataFrame, input_type: str) -> np.ndarray:
        """Convert any input signal type to RR intervals"""
        
        if input_type == "rr_intervals":
            return self._process_rr_intervals(data)
        elif input_type == "heart_rate":
            return self._process_heart_rate(data)
        elif input_type == "ecg":
            return self._process_ecg(data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _process_rr_intervals(self, data: pd.DataFrame) -> np.ndarray:
        """Process RR interval data"""
        rr_intervals = data.iloc[:, 1].values
        
        # Remove outliers if configured
        if self.config.outlier_removal:
            rr_intervals = self._remove_outliers(rr_intervals)
        
        # Interpolate missing values if any
        if np.any(np.isnan(rr_intervals)):
            rr_intervals = self._interpolate_missing(rr_intervals)
        
        return rr_intervals
    
    def _process_heart_rate(self, data: pd.DataFrame) -> np.ndarray:
        """Convert heart rate (BPM) to RR intervals (ms)"""
        heart_rate = data.iloc[:, 1].values
        
        # Remove outliers
        if self.config.outlier_removal:
            heart_rate = self._remove_outliers(heart_rate)
        
        # Convert BPM to RR intervals: RRI(ms) = 60000 / BPM
        valid_mask = (heart_rate > 0) & np.isfinite(heart_rate)
        rr_intervals = np.full_like(heart_rate, np.nan, dtype=np.float64)
        rr_intervals[valid_mask] = 60000.0 / heart_rate[valid_mask]
        
        # Remove NaN values
        rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
        
        return rr_intervals
    
    def _process_ecg(self, data: pd.DataFrame) -> np.ndarray:
        """Process raw ECG signal to extract RR intervals"""
        ecg_signal = data.iloc[:, 1].values
        
        try:
            # Clean ECG signal
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.config.sampling_rate)
            
            # Detect R peaks
            peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.config.sampling_rate, 
                                  correct_artifacts=True)
            
            # Extract RR intervals from peaks
            peak_indices = peaks["ECG_R_Peaks"]
            if len(peak_indices) < 2:
                raise ValueError("Insufficient R peaks detected in ECG signal")
            
            # Convert peak indices to time intervals
            rr_intervals = np.diff(peak_indices) / self.config.sampling_rate * 1000  # Convert to ms
            
            # Remove outliers
            if self.config.outlier_removal:
                rr_intervals = self._remove_outliers(rr_intervals)
            
            return rr_intervals
            
        except Exception as e:
            raise ValueError(f"Failed to process ECG signal: {e}")
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using specified method"""
        
        if self.config.outlier_method == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (data >= lower_bound) & (data <= upper_bound)
            
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            mask = z_scores <= 3
            
        elif self.config.outlier_method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(data.reshape(-1, 1))
                mask = outlier_pred == 1
            except ImportError:
                self.logger.warning("sklearn not available, falling back to IQR method")
                return self._remove_outliers(data)  # Fallback to IQR
        else:
            # No outlier removal
            return data
        
        return data[mask]
    
    def _interpolate_missing(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values"""
        if not np.any(np.isnan(data)):
            return data
        
        # Create index for interpolation
        indices = np.arange(len(data))
        valid_mask = ~np.isnan(data)
        
        if self.config.interpolation_method == "linear":
            data[~valid_mask] = np.interp(indices[~valid_mask], 
                                        indices[valid_mask], 
                                        data[valid_mask])
        elif self.config.interpolation_method == "cubic":
            from scipy.interpolate import CubicSpline
            if np.sum(valid_mask) >= 4:  # Need at least 4 points for cubic
                cs = CubicSpline(indices[valid_mask], data[valid_mask])
                data[~valid_mask] = cs(indices[~valid_mask])
            else:
                # Fallback to linear
                data[~valid_mask] = np.interp(indices[~valid_mask], 
                                            indices[valid_mask], 
                                            data[valid_mask])
        
        return data

class OptimizedHRVComputer:
    """High-performance HRV computation for generic time series"""
    
    def __init__(self, config: HRVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_comprehensive_hrv(self, rr_intervals: np.ndarray, metadata: Dict = None) -> Dict:
        """Compute all HRV metrics for RR intervals"""
        results = {"metadata": metadata or {}}
        
        try:
            # Basic signal statistics
            results.update(self._compute_basic_statistics(rr_intervals))
            
            # Convert RRI to peaks for NeuroKit2
            peaks = nk.intervals_to_peaks(rr_intervals)
            
            # Time domain metrics
            if self.config.hrv_time_domain:
                time_results = self._compute_time_domain(peaks)
                results.update(time_results)
            
            # Frequency domain metrics
            if self.config.hrv_frequency_domain:
                freq_results = self._compute_frequency_domain(peaks)
                results.update(freq_results)
            
            # Non-linear domain metrics
            if self.config.hrv_nonlinear_domain:
                nonlinear_results = self._compute_nonlinear_domain(peaks)
                results.update(nonlinear_results)
            
            # Higher-order temporal fluctuation metrics
            if self.config.hrv_higher_order:
                ho_results = self._compute_higher_order_metrics(rr_intervals)
                results.update(ho_results)
                
        except Exception as e:
            self.logger.error(f"HRV computation failed: {e}")
            raise
        
        return results
    
    def _compute_basic_statistics(self, rr_intervals: np.ndarray) -> Dict:
        """Compute basic signal statistics"""
        return {
            "n_intervals": len(rr_intervals),
            "signal_duration_minutes": np.sum(rr_intervals) / 1000 / 60,
            "mean_rr": np.mean(rr_intervals),
            "std_rr": np.std(rr_intervals),
            "median_rr": np.median(rr_intervals),
            "min_rr": np.min(rr_intervals),
            "max_rr": np.max(rr_intervals),
            "mean_heart_rate": 60000 / np.mean(rr_intervals),
        }
    
    def _compute_time_domain(self, peaks: np.ndarray) -> Dict:
        """Compute time domain HRV metrics"""
        try:
            time_hrv = nk.hrv_time(peaks, sampling_rate=self.config.sampling_rate)
            return {f"time_{k}": v for k, v in time_hrv.to_dict('records')[0].items()}
        except Exception as e:
            self.logger.warning(f"Time domain computation failed: {e}")
            return {}
    
    def _compute_frequency_domain(self, peaks: np.ndarray) -> Dict:
        """Compute frequency domain HRV metrics"""
        try:
            freq_hrv = nk.hrv_frequency(peaks, sampling_rate=self.config.sampling_rate,
                                      method='welch')
            return {f"freq_{k}": v for k, v in freq_hrv.to_dict('records')[0].items()}
        except Exception as e:
            self.logger.warning(f"Frequency domain computation failed: {e}")
            return {}
    
    def _compute_nonlinear_domain(self, peaks: np.ndarray) -> Dict:
        """Compute nonlinear domain HRV metrics"""
        try:
            nonlinear_hrv = nk.hrv_nonlinear(peaks, sampling_rate=self.config.sampling_rate)
            return {f"nonlinear_{k}": v for k, v in nonlinear_hrv.to_dict('records')[0].items()}
        except Exception as e:
            self.logger.warning(f"Nonlinear domain computation failed: {e}")
            return {}
    
    def _compute_higher_order_metrics(self, rr_intervals: np.ndarray) -> Dict:
        """Compute temporal fluctuation and higher-order metrics"""
        results = {}
        
        try:
            # Coefficient of variation
            results['ho_rr_cv'] = np.std(rr_intervals) / np.mean(rr_intervals) * 100
            
            # Temporal variability using sliding windows
            if len(rr_intervals) > 100:
                window_size = min(50, len(rr_intervals) // 4)
                if window_size > 10:
                    windowed_means = []
                    windowed_stds = []
                    
                    for i in range(0, len(rr_intervals) - window_size, window_size // 2):
                        window = rr_intervals[i:i + window_size]
                        windowed_means.append(np.mean(window))
                        windowed_stds.append(np.std(window))
                    
                    if len(windowed_means) > 1:
                        results['ho_temporal_mean_cv'] = np.std(windowed_means) / np.mean(windowed_means) * 100
                        results['ho_temporal_std_cv'] = np.std(windowed_stds) / np.mean(windowed_stds) * 100
            
            # Additional complexity metrics
            if len(rr_intervals) > 200:
                try:
                    # Sample entropy with adaptive parameters
                    results['ho_sample_entropy'] = nk.entropy_sample(rr_intervals, delay=1, dimension=2)
                    
                    # Approximate entropy
                    results['ho_approximate_entropy'] = nk.entropy_approximate(rr_intervals, delay=1, dimension=2)
                    
                except Exception as e:
                    self.logger.debug(f"Advanced entropy computation failed: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Higher-order metrics computation failed: {e}")
        
        return results

# ============================================================================
# Caching and Pipeline Classes (same as before but with updated metadata)
# ============================================================================

class HRVResultsCache:
    """Intelligent caching system for HRV results"""
    
    def __init__(self, cache_dir: str = "./hrv_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _get_config_hash(self, config: HRVConfig) -> str:
        """Generate hash of configuration"""
        config_str = str(sorted(config.__dict__.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_cache_key(self, file_path: str, config: HRVConfig) -> str:
        """Generate unique cache key"""
        file_hash = self._get_file_hash(file_path)
        config_hash = self._get_config_hash(config)
        return f"{file_hash}_{config_hash}"
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached results"""
        cache_file = self.cache_dir / f"{cache_key}.joblib"
        
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {e}")
                cache_file.unlink()  # Remove corrupted cache file
        
        return None
    
    def cache_result(self, cache_key: str, result: Dict):
        """Cache computation results"""
        cache_file = self.cache_dir / f"{cache_key}.joblib"
        
        try:
            joblib.dump(result, cache_file, compress=3)
        except Exception as e:
            self.logger.warning(f"Failed to cache result {cache_key}: {e}")

# ============================================================================
# Main Pipeline
# ============================================================================

class GenericHRVPipeline:
    """Generic HRV processing pipeline for any cardiovascular time series"""
    
    def __init__(self, config: HRVConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.signal_processor = SignalProcessor(config)
        self.hrv_computer = OptimizedHRVComputer(config)
        self.cache = HRVResultsCache(config.cache_dir) if config.enable_caching else None
        
        # Set up parallel processing
        if self.config.n_workers is None:
            self.config.n_workers = min(mp.cpu_count() - 1, 8)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def discover_files(self, data_path: str, pattern: str = "*") -> List[str]:
        """Discover time series files in directory"""
        data_dir = Path(data_path)
        
        # Support multiple file extensions
        extensions = ['*.csv', '*.txt', '*.tsv', '*.xlsx', '*.xls', '*.parquet', '*.h5']
        files = []
        
        for ext in extensions:
            files.extend(data_dir.glob(f"{pattern}{ext}"))
        
        files = sorted([str(f) for f in files])
        
        self.logger.info(f"Found {len(files)} time series files")
        
        return files
    
    def process_single_file(self, file_path: str) -> ProcessingResult:
        """Process a single time series file"""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cache_key = self.cache.get_cache_key(file_path, self.config)
                cached_result = self.cache.get_cached_result(cache_key)
                if cached_result:
                    self.logger.info(f"Using cached result for {Path(file_path).name}")
                    return ProcessingResult(
                        file_path=file_path,
                        status=ProcessingStatus.SUCCESS,
                        data=cached_result,
                        processing_time=time.time() - start_time
                    )
            
            # Load and detect input type
            data, input_type = GenericDataLoader.load_time_series_file(file_path, self.config)
            
            # Check minimum signal length
            if len(data) < self.config.min_signal_length * (self.config.sampling_rate if input_type == "ecg" else 1):
                raise ValueError(f"Signal too short: {len(data)} samples")
            
            # Process signal to RR intervals
            rr_intervals = self.signal_processor.process_signal_to_rri(data, input_type)
            
            # Validate RR intervals
            if len(rr_intervals) < 50:
                raise ValueError("Insufficient RR intervals for reliable HRV analysis")
            
            # Create metadata
            metadata = {
                "file_path": file_path,
                "input_type": input_type,
                "original_samples": len(data),
                "processed_rr_intervals": len(rr_intervals),
                "processing_config": {
                    "sampling_rate": self.config.sampling_rate if input_type == "ecg" else None,
                    "outlier_removal": self.config.outlier_removal,
                    "interpolation_method": self.config.interpolation_method
                }
            }
            
            # Compute HRV metrics
            hrv_results = self.hrv_computer.compute_comprehensive_hrv(rr_intervals, metadata)
            
            # Cache results
            if self.cache:
                self.cache.cache_result(cache_key, hrv_results)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Successfully processed {Path(file_path).name} "
                           f"({input_type}, {len(rr_intervals)} RR intervals) in {processing_time:.2f}s")
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.SUCCESS,
                input_type=input_type,
                data=hrv_results,
                processing_time=processing_time,
                n_samples=len(rr_intervals),
                signal_duration=np.sum(rr_intervals) / 1000  # seconds
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process {Path(file_path).name}: {str(e)}"
            
            self.logger.error(error_msg)
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.FAILED,
                error=error_msg,
                processing_time=processing_time
            )
    
    def process_dataset_parallel(self, files: List[str]) -> List[ProcessingResult]:
        """Process multiple files in parallel"""
        self.logger.info(f"Processing {len(files)} files using {self.config.n_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in files
            }
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Worker failed for {file_path}: {e}")
                    results.append(ProcessingResult(
                        file_path=file_path,
                        status=ProcessingStatus.FAILED,
                        error=str(e)
                    ))
        
        return results
    
    def save_results(self, results: List[ProcessingResult], output_path: str):
        """Save processing results to file"""
        # Filter successful results
        successful_results = [r for r in results if r.status == ProcessingStatus.SUCCESS]
        
        if not successful_results:
            self.logger.warning("No successful results to save")
            return
        
        # Combine all data
        all_data = []
        for result in successful_results:
            if result.data:
                # Flatten nested dictionaries
                flattened = {}
                for key, value in result.data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened[f"{key}_{subkey}"] = subvalue
                    else:
                        flattened[key] = value
                
                # Add result metadata
                flattened['result_input_type'] = result.input_type
                flattened['result_n_samples'] = result.n_samples
                flattened['result_signal_duration'] = result.signal_duration
                flattened['result_processing_time'] = result.processing_time
                
                all_data.append(flattened)
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Save based on format preference
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            if self.config.output_format == "parquet":
                df.to_parquet(output_path.with_suffix('.parquet'), index=False)
            elif self.config.output_format == "hdf5":
                df.to_hdf(output_path.with_suffix('.h5'), key='hrv_results', mode='w')
            else:  # CSV
                df.to_csv(output_path.with_suffix('.csv'), index=False)
            
            self.logger.info(f"Saved {len(df)} HRV results to {output_path}")
        
        # Save processing summary
        self._save_processing_summary(results, output_path.parent / "processing_summary.txt")
    
    def _save_processing_summary(self, results: List[ProcessingResult], summary_path: str):
        """Save processing summary statistics"""
        successful = [r for r in results if r.status == ProcessingStatus.SUCCESS]
        failed = [r for r in results if r.status == ProcessingStatus.FAILED]
        
        # Group by input type
        input_types = {}
        for r in successful:
            if r.input_type not in input_types:
                input_types[r.input_type] = []
            input_types[r.input_type].append(r)
        
        total_time = sum(r.processing_time for r in results)
        total_samples = sum(r.n_samples for r in successful)
        total_duration = sum(r.signal_duration for r in successful)
        
        summary = f"""
Generic HRV Processing Summary
=============================
Total files processed: {len(results)}
Successful: {len(successful)}
Failed: {len(failed)}
Total processing time: {total_time:.2f} seconds
Total RR intervals processed: {total_samples:,}
Total signal duration: {total_duration/3600:.2f} hours
Average processing speed: {total_samples/total_time:.0f} intervals/second

Input Type Distribution:
{chr(10).join(f"  {input_type}: {len(files)} files" for input_type, files in input_types.items())}

Failed files:
{chr(10).join(r.file_path + ': ' + (r.error or 'Unknown error') for r in failed)}
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Processing summary saved to {summary_path}")

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point for the generic HRV pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generic HRV Processing Pipeline for RR intervals, Heart Rate, and ECG data"
    )
    parser.add_argument("--data-path", required=True, 
                       help="Path to directory containing time series files or single file")
    parser.add_argument("--output-path", required=True,
                       help="Output file path for results")
    parser.add_argument("--config", 
                       help="Configuration YAML file path")
    parser.add_argument("--input-type", choices=["auto", "rr_intervals", "heart_rate", "ecg"],
                       default="auto", help="Input data type (auto-detect if not specified)")
    parser.add_argument("--sampling-rate", type=int, default=1000,
                       help="Sampling rate for ECG signals (Hz)")
    parser.add_argument("--n-workers", type=int,
                       help="Number of parallel workers")
    parser.add_argument("--file-pattern", default="*",
                       help="File pattern to match (e.g., 'subject_*' for subject files)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = HRVConfig.from_yaml(args.config)
    else:
        config = HRVConfig()
    
    # Override with command line arguments
    if args.n_workers:
        config.n_workers = args.n_workers
    if args.input_type != "auto":
        config.input_type = args.input_type
    if args.sampling_rate != 1000:
        config.sampling_rate = args.sampling_rate
    
    # Initialize pipeline
    pipeline = GenericHRVPipeline(config)
    
    # Check if data_path is a single file or directory
    data_path = Path(args.data_path)
    if data_path.is_file():
        files = [str(data_path)]
    elif data_path.is_dir():
        files = pipeline.discover_files(str(data_path), args.file_pattern)
    else:
        pipeline.logger.error(f"Data path does not exist: {data_path}")
        return
    
    if not files:
        pipeline.logger.error("No valid time series files found!")
        return
    
    # Process files
    results = pipeline.process_dataset_parallel(files)
    
    # Save results
    pipeline.save_results(results, args.output_path)
    
    pipeline.logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
