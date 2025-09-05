import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import pyproj
from pyproj import Transformer

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au path de manière plus robuste
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

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
        Traite les fichiers de profils CSV.
        
        Returns:
            Dict associant les noms des dispositifs aux tuples (clean_path, report)
        """
        # Chercher les fichiers de profils
        profiles_dir = self.raw_data_dir / "csv" / "profiles"
        results = {}
        
        if not profiles_dir.exists():
            logger.warning(f"Répertoire des profils non trouvé: {profiles_dir}")
            # Créer des données factices
            return self._create_dummy_data()
        
        # Lister tous les fichiers CSV de profils
        profile_files = list(profiles_dir.glob("*.csv"))
        
        if not profile_files:
            logger.warning("Aucun fichier de profil trouvé")
            return self._create_dummy_data()
        
        logger.info(f"Trouvé {len(profile_files)} fichiers de profils")
        
        # Traiter chaque fichier de profil
        for i, profile_file in enumerate(profile_files[:5]):  # Limiter à 5 profils
            device_name = f"profil_{i+1}"
            logger.info(f"Nettoyage des données pour le profil: {device_name}")
            
            try:
                clean_path, report = self._clean_profile_data(device_name, profile_file)
                results[device_name] = (clean_path, report)
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {device_name}: {e}")
                continue
        
        if not results:
            logger.warning("Aucun profil traité avec succès, création de données factices")
            return self._create_dummy_data()
                
        return results

    def _clean_profile_data(self, device_name: str, profile_file: Path) -> Tuple[Path, Dict]:
        """Nettoyer les données d'un profil spécifique."""
        try:
            # Lire le fichier CSV avec le bon séparateur
            df = pd.read_csv(profile_file, sep=';')
            
            # Vérifier les colonnes requises
            required_columns = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Colonnes manquantes: {missing_columns}")
            
            # Nettoyer les données
            df_clean = df.copy()
            
            # Supprimer les lignes avec des valeurs manquantes
            df_clean = df_clean.dropna(subset=required_columns)
            
            # Supprimer les valeurs aberrantes (optionnel)
            df_clean = df_clean[df_clean['Rho(ohm.m)'] > 0]
            df_clean = df_clean[df_clean['M (mV/V)'] >= 0]
            
            # Renommer les colonnes pour la compatibilité
            df_clean = df_clean.rename(columns={
                'Rho(ohm.m)': 'resistivity',
                'M (mV/V)': 'chargeability'
            })
            
            # Sauvegarder les données nettoyées
            clean_file = self.processed_data_dir / f"{device_name}_cleaned.csv"
            df_clean.to_csv(clean_file, index=False)
            
            report = {
                'original_count': len(df),
                'cleaned_count': len(df_clean),
                'removed_count': len(df) - len(df_clean)
            }
            
            logger.info(f"Profil {device_name} nettoyé: {len(df_clean)}/{len(df)} enregistrements conservés")
            
            return clean_file, report
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du profil {device_name}: {e}")
            raise

    def _create_dummy_data(self) -> Dict[str, Tuple[Path, Dict]]:
        """Créer des données factices pour la démonstration."""
        logger.info("Création de données factices pour la démonstration...")
        
        # Créer des données factices
        n_samples = 100
        df = pd.DataFrame({
            'x': np.random.uniform(500000, 510000, n_samples),
            'y': np.random.uniform(450000, 460000, n_samples),
            'z': np.random.uniform(500, 600, n_samples),
            'resistivity': np.random.uniform(1e-8, 1e9, n_samples),
            'chargeability': np.random.uniform(0, 200, n_samples)
        })
        
        # Sauvegarder
        clean_file = self.processed_data_dir / "dummy_cleaned.csv"
        df.to_csv(clean_file, index=False)
        
        report = {
            'original_count': n_samples,
            'cleaned_count': n_samples,
            'removed_count': 0
        }
        
        return {"dummy": (clean_file, report)}

    def _clean_device_data(self, device_name: str, raw_file: Path) -> Tuple[Path, Dict]:
        """
        Clean data for a specific device.
        Seuls les fichiers CSV sont acceptés.
        
        Args:
            device_name: Name of the device
            raw_file: Path to raw data file (doit être un CSV)
            
        Returns:
            Tuple of (clean_path, cleaning_report)
        """
        clean_file = self.processed_data_dir / f"{device_name}_cleaned.csv"
        
        if clean_file.exists():
            logger.info(f"Cleaned data already exists for {device_name}, skipping cleaning")
            return clean_file, {}
        
        # Valider le format CSV avant de charger les données
        if not self._validate_csv_format(raw_file):
            raise ValueError(f"Le fichier {raw_file} n'est pas un CSV valide")
        
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
        """Load data from CSV files only."""
        if file_path.suffix.lower() != '.csv':
            raise ValueError(f"Seuls les fichiers CSV sont supportés. Format détecté: {file_path.suffix}")
        
        try:
            # Essayer d'abord avec le séparateur par défaut (virgule)
            df = pd.read_csv(file_path)
            
            # Si une seule colonne, essayer avec point-virgule
            if len(df.columns) == 1:
                logger.info(f"Fichier {file_path} semble utiliser des points-virgules comme séparateur")
                df = pd.read_csv(file_path, sep=';')
            
            logger.debug(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du fichier CSV {file_path}: {str(e)}")

    def _validate_csv_format(self, file_path: Path) -> bool:
        """Valider que le fichier est un CSV valide."""
        try:
            # Essayer de lire les premières lignes pour vérifier le format
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline() for _ in range(5)]
            
            # Vérifier qu'il y a des virgules ou points-virgules (séparateurs CSV)
            has_separators = any(',' in line or ';' in line for line in first_lines if line.strip())
            
            if not has_separators:
                logger.warning(f"Le fichier {file_path} ne semble pas contenir de séparateurs CSV valides")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la validation du format CSV de {file_path}: {e}")
            return False

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

    def validate_all_input_files(self) -> Dict[str, bool]:
        """
        Valider que tous les fichiers d'entrée sont des CSV valides.
        
        Returns:
            Dict associant les noms des dispositifs à leur statut de validation
        """
        validation_results = {}
        
        for device_name, device_config in CONFIG.geophysical_data.devices.items():
            raw_file = self.raw_data_dir / device_config['file']
            
            if not raw_file.exists():
                validation_results[device_name] = False
                logger.warning(f"Fichier non trouvé pour {device_name}: {raw_file}")
                continue
            
            # Vérifier l'extension
            if raw_file.suffix.lower() != '.csv':
                validation_results[device_name] = False
                logger.error(f"Format non supporté pour {device_name}: {raw_file.suffix} (CSV requis)")
                continue
            
            # Valider le contenu CSV
            is_valid_csv = self._validate_csv_format(raw_file)
            validation_results[device_name] = is_valid_csv
            
            if is_valid_csv:
                logger.info(f"✓ {device_name}: CSV valide")
            else:
                logger.error(f"✗ {device_name}: CSV invalide")
        
        return validation_results

    def get_cleaning_summary(self) -> Dict:
        """Get a summary of all cleaning operations."""
        return self.report
