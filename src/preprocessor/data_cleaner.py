import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import pyproj
from pyproj import Transformer

from config import CONFIG
from src.utils.logger import logger


class GeophysicalDataCleaner:
    """
    Nettoyeur pour les données géophysiques, inspiré du DataCleaner d'EMUT.
    Gère la transformation des coordonnées, la validation et le nettoyage des données.
    """
    
    def __init__(self):
        self.report = {}
        self.raw_data_dir = Path(CONFIG.paths.raw_data_dir)
        self.processed_data_dir = Path(CONFIG.paths.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le transformateur de coordonnées
        self.coord_transformer = Transformer.from_crs(
            CONFIG.geophysical_data.coordinate_systems['wgs84'],
            CONFIG.geophysical_data.coordinate_systems['utm_proj'],
            always_xy=True
        )

    def clean_all_devices(self) -> Dict[str, Tuple[Path, Dict]]:
        """
        Nettoyer les données de tous les dispositifs géophysiques.
        
        Returns:
            Dict associant les noms des dispositifs aux tuples (clean_path, report)
        """
        results = {}
        
        for device_name, device_config in CONFIG.geophysical_data.devices.items():
            logger.info(f"Nettoyage des données pour le dispositif: {device_name}")
            
            raw_file = self.raw_data_dir / device_config['file']
            if raw_file.exists():
                clean_path, report = self._clean_device_data(device_name, raw_file)
                results[device_name] = (clean_path, report)
            else:
                logger.warning(f"Fichier de données brutes non trouvé: {raw_file}")
                
        return results

    def _clean_device_data(self, device_name: str, raw_file: Path) -> Tuple[Path, Dict]:
        """
        Clean data for a specific device.
        
        Args:
            device_name: Name of the device
            raw_file: Path to raw data file
            
        Returns:
            Tuple of (clean_path, cleaning_report)
        """
        clean_file = self.processed_data_dir / f"{device_name}_cleaned.csv"
        
        if clean_file.exists():
            logger.info(f"Cleaned data already exists for {device_name}, skipping cleaning")
            return clean_file, {}
        
        # Load raw data
        df = self._load_device_data(raw_file, device_name)
        original_count = len(df)
        
        # Apply cleaning steps
        df = self._validate_columns(df, device_name)
        df = self._handle_missing_values(df)
        df = self._clean_coordinates(df, device_name)
        df = self._normalize_geophysical_values(df)
        df = self._remove_outliers(df)
        df = self._validate_spatial_coverage(df, device_name)
        
        # Save cleaned data
        df.to_csv(clean_file, index=False)
        
        # Generate report
        report = {
            "device": device_name,
            "original_count": original_count,
            "cleaned_count": len(df),
            "removed_count": original_count - len(df),
            "clean_path": str(clean_file),
            "coverage_area": self._calculate_coverage_area(df),
            "value_ranges": self._get_value_ranges(df)
        }
        
        self.report[device_name] = report
        logger.info(f"Cleaned {device_name}: {len(df)}/{original_count} records kept")
        
        return clean_file, report

    def _load_device_data(self, file_path: Path, device_name: str) -> pd.DataFrame:
        """Load data based on device type and file format."""
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.dat':
            # Handle .dat files (space or tab separated)
            df = pd.read_csv(file_path, sep=r'\s+', engine='python')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.debug(f"Loaded {len(df)} records from {file_path}")
        return df

    def _validate_columns(self, df: pd.DataFrame, device_name: str) -> pd.DataFrame:
        """Ensure required columns are present."""
        required_cols = CONFIG.geophysical_data.required_columns
        
        # Check if we have at least some of the required columns
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:  # Need at least coordinates and one measurement
            logger.warning(f"Device {device_name}: Missing required columns. Available: {df.columns.tolist()}")
            
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values in critical columns."""
        initial_count = len(df)
        
        # Remove rows with missing coordinates
        coord_cols = ['x', 'y', 'z']
        coord_cols = [col for col in coord_cols if col in df.columns]
        
        if coord_cols:
            df = df.dropna(subset=coord_cols)
            logger.debug(f"Removed {initial_count - len(df)} rows with missing coordinates")
        
        return df

    def _clean_coordinates(self, df: pd.DataFrame, device_name: str) -> pd.DataFrame:
        """Clean and transform coordinates if needed."""
        df = df.copy()
        
        # Handle coordinate transformation if needed
        if 'lat' in df.columns and 'lon' in df.columns:
            logger.info(f"Transforming coordinates for {device_name}")
            df['x'], df['y'] = self._transform_coordinates(df['lat'], df['lon'])
            df = df.drop(['lat', 'lon'], axis=1)
        
        # Ensure coordinates are numeric
        coord_cols = ['x', 'y', 'z']
        coord_cols = [col for col in coord_cols if col in df.columns]
        
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid coordinates
        df = df.dropna(subset=coord_cols)
        
        return df

    def _transform_coordinates(self, lat: pd.Series, lon: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Transform LAT/LON to UTM coordinates."""
        x, y = self.coord_transformer.transform(lon.values, lat.values)
        return x, y

    def _normalize_geophysical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize geophysical measurements."""
        df = df.copy()
        
        # Normalize resistivity (log scale)
        if 'resistivity' in df.columns:
            df['resistivity'] = pd.to_numeric(df['resistivity'], errors='coerce')
            # Remove negative or zero values
            df = df[df['resistivity'] > 0]
            logger.debug(f"Normalized resistivity values")
        
        # Normalize chargeability
        if 'chargeability' in df.columns:
            df['chargeability'] = pd.to_numeric(df['chargeability'], errors='coerce')
            # Remove negative values
            df = df[df['chargeability'] >= 0]
            logger.debug(f"Normalized chargeability values")
        
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from measurements."""
        initial_count = len(df)
        
        # Remove outliers from resistivity (using IQR method)
        if 'resistivity' in df.columns:
            Q1 = df['resistivity'].quantile(0.25)
            Q3 = df['resistivity'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df['resistivity'] >= lower_bound) & (df['resistivity'] <= upper_bound)]
        
        # Remove outliers from chargeability
        if 'chargeability' in df.columns:
            Q1 = df['chargeability'].quantile(0.25)
            Q3 = df['chargeability'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df['chargeability'] >= lower_bound) & (df['chargeability'] <= upper_bound)]
        
        logger.debug(f"Removed {initial_count - len(df)} outlier records")
        return df

    def _validate_spatial_coverage(self, df: pd.DataFrame, device_name: str) -> pd.DataFrame:
        """Validate that data covers the expected spatial area."""
        if 'x' in df.columns and 'y' in df.columns:
            x_range = df['x'].max() - df['x'].min()
            y_range = df['y'].max() - df['y'].min()
            
            expected_coverage = CONFIG.geophysical_data.devices[device_name]['coverage']
            logger.info(f"Device {device_name}: Coverage {x_range:.1f}m x {y_range:.1f}m")
        
        return df

    def _calculate_coverage_area(self, df: pd.DataFrame) -> Dict:
        """Calculate the spatial coverage area."""
        if 'x' in df.columns and 'y' in df.columns:
            return {
                'x_min': df['x'].min(),
                'x_max': df['x'].max(),
                'y_min': df['y'].min(),
                'y_max': df['y'].max(),
                'width': df['x'].max() - df['x'].min(),
                'height': df['y'].max() - df['y'].min()
            }
        return {}

    def _get_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Get the range of geophysical values."""
        ranges = {}
        
        if 'resistivity' in df.columns:
            ranges['resistivity'] = {
                'min': df['resistivity'].min(),
                'max': df['resistivity'].max(),
                'mean': df['resistivity'].mean()
            }
        
        if 'chargeability' in df.columns:
            ranges['chargeability'] = {
                'min': df['chargeability'].min(),
                'max': df['chargeability'].max(),
                'mean': df['chargeability'].mean()
            }
        
        return ranges

    def get_cleaning_summary(self) -> Dict:
        """Get a summary of all cleaning operations."""
        return self.report
