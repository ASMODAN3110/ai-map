import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import pyproj
from pyproj import Transformer
import torch
import torch.nn as nn
import csv
import io

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au path de manière plus robuste
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import du CONFIG depuis le backend
from backend.config import CONFIG
from backend.utils.logger import logger


def detect_csv_separator(file_path: Path) -> str:
    """
    Détecte automatiquement le séparateur CSV utilisé dans un fichier.
    
    Args:
        file_path: Chemin vers le fichier CSV
        
    Returns:
        Le séparateur détecté (',' ou ';' ou '\t')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Lire les premières lignes pour analyser
            sample = file.read(1024)
            
        # Analyser avec différents séparateurs
        separators = [',', ';', '\t']
        best_separator = ','
        max_columns = 0
        
        for sep in separators:
            try:
                # Utiliser csv.Sniffer pour détecter le format
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=sep)
                
                # Compter le nombre de colonnes
                reader = csv.reader(io.StringIO(sample), dialect=dialect)
                first_row = next(reader)
                num_columns = len(first_row)
                
                if num_columns > max_columns:
                    max_columns = num_columns
                    best_separator = sep
                    
            except Exception:
                continue
        
        # Si aucun séparateur n'est détecté, essayer de compter manuellement
        if max_columns == 0:
            for sep in separators:
                lines = sample.split('\n')[:3]  # Prendre les 3 premières lignes
                if lines:
                    num_columns = len(lines[0].split(sep))
                    if num_columns > max_columns:
                        max_columns = num_columns
                        best_separator = sep
        
        logger.debug(f"Fichier {file_path.name} semble utiliser '{best_separator}' comme séparateur")
        return best_separator
        
    except Exception as e:
        logger.warning(f"Impossible de détecter le séparateur pour {file_path.name}: {e}")
        return ','  # Par défaut, utiliser la virgule


def read_csv_with_auto_separator(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Lit un fichier CSV en détectant automatiquement le séparateur.
    
    Args:
        file_path: Chemin vers le fichier CSV
        **kwargs: Arguments supplémentaires pour pd.read_csv
        
    Returns:
        DataFrame pandas
    """
    separator = detect_csv_separator(file_path)
    return pd.read_csv(file_path, sep=separator, **kwargs)


def normalize_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes de coordonnées en détectant différentes variantes.
    
    Args:
        df: DataFrame avec des colonnes de coordonnées
        
    Returns:
        DataFrame avec des colonnes de coordonnées normalisées
    """
    df_normalized = df.copy()
    
    # Mapping des variantes de coordonnées X
    x_variants = ['x', 'X', 'X_UTM', 'x_utm', 'X_UTM', 'x_coord', 'X_COORD', 
                  'xA', 'xA(m)', 'xA (m)', 'X_A', 'X_A(m)', 'X_A (m)']
    
    # Mapping des variantes de coordonnées Y  
    y_variants = ['y', 'Y', 'Y_UTM', 'y_utm', 'Y_UTM', 'y_coord', 'Y_COORD',
                  'yB', 'yB(m)', 'yB (m)', 'Y_B', 'Y_B(m)', 'Y_B (m)']
    
    # Mapping des variantes de coordonnées Z
    z_variants = ['z', 'Z', 'Z_UTM', 'z_utm', 'Z_UTM', 'z_coord', 'Z_COORD',
                  'zM', 'zM(m)', 'zM (m)', 'Z_M', 'Z_M(m)', 'Z_M (m)', 'depth', 'DEPTH']
    
    # Mapping des variantes de latitude
    lat_variants = ['lat', 'LAT', 'latitude', 'LATITUDE', 'Lat', 'LAT_deg', 'lat_deg']
    
    # Mapping des variantes de longitude
    lon_variants = ['lon', 'LON', 'longitude', 'LONGITUDE', 'Lon', 'LON_deg', 'lon_deg']
    
    # Normaliser les colonnes X
    for variant in x_variants:
        if variant in df_normalized.columns and 'x' not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={variant: 'x'})
            logger.debug(f"Colonne {variant} renommée en 'x'")
            break
    
    # Normaliser les colonnes Y
    for variant in y_variants:
        if variant in df_normalized.columns and 'y' not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={variant: 'y'})
            logger.debug(f"Colonne {variant} renommée en 'y'")
            break
    
    # Normaliser les colonnes Z
    for variant in z_variants:
        if variant in df_normalized.columns and 'z' not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={variant: 'z'})
            logger.debug(f"Colonne {variant} renommée en 'z'")
            break
    
    # Normaliser les colonnes LAT
    for variant in lat_variants:
        if variant in df_normalized.columns and 'LAT' not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={variant: 'LAT'})
            logger.debug(f"Colonne {variant} renommée en 'LAT'")
            break
    
    # Normaliser les colonnes LON
    for variant in lon_variants:
        if variant in df_normalized.columns and 'LON' not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={variant: 'LON'})
            logger.debug(f"Colonne {variant} renommée en 'LON'")
            break
    
    return df_normalized


def validate_coordinate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valide les données de coordonnées et retourne un rapport de validation.
    
    Args:
        df: DataFrame avec des colonnes de coordonnées
        
    Returns:
        Dictionnaire avec les résultats de validation
    """
    validation_report = {
        'has_utm_coords': False,
        'has_wgs84_coords': False,
        'coordinate_systems': [],
        'coverage_area': {},
        'issues': []
    }
    
    # Vérifier les coordonnées UTM (x, y, z)
    utm_cols = ['x', 'y']
    if all(col in df.columns for col in utm_cols):
        validation_report['has_utm_coords'] = True
        validation_report['coordinate_systems'].append('UTM')
        
        # Calculer la couverture spatiale
        x_range = df['x'].max() - df['x'].min()
        y_range = df['y'].max() - df['y'].min()
        
        validation_report['coverage_area'] = {
            'x_min': df['x'].min(),
            'x_max': df['x'].max(),
            'y_min': df['y'].min(),
            'y_max': df['y'].max(),
            'width': x_range,
            'height': y_range
        }
        
        # Vérifier la validité des coordonnées UTM
        if x_range < 1:
            validation_report['issues'].append("Couverture X très faible (< 1m)")
        if y_range < 1:
            validation_report['issues'].append("Couverture Y très faible (< 1m)")
    
    # Vérifier les coordonnées WGS84 (LAT, LON)
    wgs84_cols = ['LAT', 'LON']
    if all(col in df.columns for col in wgs84_cols):
        validation_report['has_wgs84_coords'] = True
        validation_report['coordinate_systems'].append('WGS84')
        
        # Vérifier la validité des coordonnées WGS84
        lat_range = df['LAT'].max() - df['LAT'].min()
        lon_range = df['LON'].max() - df['LON'].min()
        
        if lat_range < 0.001:  # < 0.001 degré
            validation_report['issues'].append("Couverture latitude très faible (< 0.001°)")
        if lon_range < 0.001:  # < 0.001 degré
            validation_report['issues'].append("Couverture longitude très faible (< 0.001°)")
    
    # Vérifier s'il n'y a aucune coordonnée
    if not validation_report['has_utm_coords'] and not validation_report['has_wgs84_coords']:
        validation_report['issues'].append("Aucune coordonnée valide trouvée")
    
    return validation_report


class GeophysicalDataCleaner:
    """
    Nettoyeur pour les données géophysiques, inspiré du DataCleaner d'EMUT.
    Gère la transformation des coordonnées, la validation et le nettoyage des données.
    Intégré avec les générateurs U-Net 2D et VoxNet 3D pour la génération d'images.
    """
    
    def __init__(self, device: str = "cpu"):
        self.report = {}
        self.device = device
        self.raw_data_dir = Path(CONFIG.paths.raw_data_dir)
        self.processed_data_dir = Path(CONFIG.paths.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le transformateur de coordonnées
        self.coord_transformer = Transformer.from_crs(
            CONFIG.geophysical_data.coordinate_systems['wgs84'],
            CONFIG.geophysical_data.coordinate_systems['utm_proj'],
            always_xy=True
        )
        
        # Définir les 2 dispositifs géophysiques supportés
        self.supported_devices = {
            'pole_dipole': {
                'name': 'Pole-Dipole',
                'description': 'Dispositif Pole-Dipole pour la prospection électrique',
                'electrodes': ['A', 'B', 'M', 'N'],
                'measurements': ['resistivity', 'chargeability', 'spontaneous_potential']
            },
            'schlumberger': {
                'name': 'Schlumberger',
                'description': 'Dispositif Schlumberger pour la prospection électrique',
                'electrodes': ['A', 'B', 'M', 'N'],
                'measurements': ['resistivity', 'chargeability', 'spontaneous_potential']
            }
        }
        
        # Paramètres pour la préparation des données pour les générateurs
        self.generator_config = {
            'unet_2d': {
                'input_size': (64, 64, 4),  # (height, width, channels)
                'output_channels': 2,  # resistivity + chargeability
                'spatial_resolution': 1.0  # mètres
            },
            'voxnet_3d': {
                'input_size': (32, 32, 32, 4),  # (depth, height, width, channels)
                'output_channels': 1,  # chargeability volume
                'spatial_resolution': 2.0  # mètres
            }
        }

    def clean_all_devices(self) -> Dict[str, Tuple[Path, Dict]]:
        """
        Nettoyer les données des dispositifs géophysiques.
        Traite les 2 dispositifs supportés (Pole-Dipole et Schlumberger) et les fichiers de profils CSV.
        
        Returns:
            Dict associant les noms des dispositifs aux tuples (clean_path, report)
        """
        results = {}
        
        # Traiter les 2 dispositifs géophysiques supportés
        for device_id, device_info in self.supported_devices.items():
            logger.info(f"Nettoyage des données pour le dispositif: {device_info['name']}")
            
            try:
                # Chercher les fichiers correspondants au dispositif
                device_files = self._find_device_files(device_id)
                
                if device_files:
                    for i, device_file in enumerate(device_files):
                        device_name = f"{device_id}_{i+1}" if len(device_files) > 1 else device_id
                        clean_path, report = self._clean_device_data(device_name, device_file)
                        results[device_name] = (clean_path, report)
                else:
                    logger.warning(f"Aucun fichier trouvé pour le dispositif {device_info['name']}")
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement du dispositif {device_info['name']}: {e}")
                continue
        
        # Traiter les fichiers de profils génériques
        profile_results = self._clean_profile_files()
        results.update(profile_results)
        
        if not results:
            logger.warning("Aucun dispositif traité avec succès, création de données factices")
            return self._create_dummy_data()
                
        return results
    
    def _find_device_files(self, device_id: str) -> List[Path]:
        """Trouver les fichiers correspondant à un dispositif spécifique."""
        device_files = []
        
        # Patterns de recherche pour chaque dispositif
        patterns = {
            'pole_dipole': ['*pole*dipole*', '*PD*', '*pole_dipole*'],
            'schlumberger': ['*schlumberger*', '*S*', '*schlumberger*']
        }
        
        if device_id in patterns:
            for pattern in patterns[device_id]:
                device_files.extend(self.raw_data_dir.glob(f"{pattern}.csv"))
        
        return list(set(device_files))  # Supprimer les doublons
    
    def _clean_profile_files(self) -> Dict[str, Tuple[Path, Dict]]:
        """Nettoyer les fichiers de profils génériques."""
        results = {}
        
        # Chercher les fichiers de profils corrigés
        profiles_dir = self.raw_data_dir
        
        if not profiles_dir.exists():
            logger.warning(f"Répertoire des profils corrigés non trouvé: {profiles_dir}")
            return results
        
        # Lister tous les fichiers CSV de profils
        profile_files = list(profiles_dir.glob("*.csv"))
        
        if not profile_files:
            logger.warning("Aucun fichier de profil trouvé")
            return results
        
        logger.info(f"Trouvé {len(profile_files)} fichiers de profils corrigés")
        
        # Traiter chaque fichier de profil
        for i, profile_file in enumerate(profile_files):
            device_name = f"profil_{i+1}"
            logger.info(f"Nettoyage des données pour le profil: {device_name}")
            
            try:
                clean_path, report = self._clean_profile_data(device_name, profile_file)
                results[device_name] = (clean_path, report)
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {device_name}: {e}")
                continue
        
        return results

    def _clean_profile_data(self, device_name: str, profile_file: Path) -> Tuple[Path, Dict]:
        """Nettoyer les données d'un profil spécifique."""
        try:
            # Lire le fichier CSV avec détection automatique du séparateur
            df = read_csv_with_auto_separator(profile_file)
            
            # Appliquer le mapping des colonnes d'abord
            column_mapping = {
                'Rho(ohm.m)': 'resistivity',
                'M (mV/V)': 'chargeability',
                'SP (mV)': 'spontaneous_potential',
                'xA (m)': 'x',
                'xB (m)': 'y',
                'xM (m)': 'z',
                'xN (m)': 'xN',
                'Dev. M': 'dev_m',
                'Dev. M (mV/V)': 'dev_m',
                'VMN (mV)': 'vmn',
                'IAB (mA)': 'iab'
            }
            
            # Renommer les colonnes
            df_clean = df.rename(columns=column_mapping)
            
            # Vérifier les colonnes requises après mapping
            required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            
            if missing_columns:
                logger.warning(f"Colonnes manquantes après mapping: {missing_columns}")
                logger.warning(f"Colonnes disponibles: {list(df_clean.columns)}")
                # Continuer avec les colonnes disponibles
                available_required = [col for col in required_columns if col in df_clean.columns]
                if len(available_required) < 3:  # Au moins 3 colonnes requises
                    raise ValueError(f"Pas assez de colonnes requises disponibles. Disponibles: {list(df_clean.columns)}")
            # Supprimer les lignes avec des valeurs manquantes (seulement pour les colonnes disponibles)
            available_required = [col for col in required_columns if col in df_clean.columns]
            if available_required:
                df_clean = df_clean.dropna(subset=available_required)
            
            # Supprimer les valeurs aberrantes (optionnel) - seulement si les colonnes existent
            if 'resistivity' in df_clean.columns:
                df_clean = df_clean[df_clean['resistivity'] > 0]
            if 'chargeability' in df_clean.columns:
                df_clean = df_clean[df_clean['chargeability'] >= 0]
            
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
        """Load data from CSV files with automatic separator detection and coordinate normalization."""
        if file_path.suffix.lower() != '.csv':
            raise ValueError(f"Seuls les fichiers CSV sont supportés. Format détecté: {file_path.suffix}")
        
        try:
            # Utiliser la détection automatique des séparateurs
            df = read_csv_with_auto_separator(file_path)
            
            # Normaliser les colonnes de coordonnées
            df = normalize_coordinate_columns(df)
            
            # Valider les données de coordonnées
            coord_validation = validate_coordinate_data(df)
            if coord_validation['issues']:
                logger.warning(f"Problèmes de coordonnées détectés pour {device_name}: {coord_validation['issues']}")
            
            logger.debug(f"Loaded {len(df)} records from {file_path}")
            logger.debug(f"Systèmes de coordonnées détectés: {coord_validation['coordinate_systems']}")
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
        """Ensure required columns are present and map them to standard names."""
        df = df.copy()
        
        # Mapping des colonnes vers les noms standard
        column_mapping = {
            'Rho(ohm.m)': 'resistivity',
            'M (mV/V)': 'chargeability',
            'SP (mV)': 'spontaneous_potential',
            'xA (m)': 'xA',
            'xB (m)': 'xB',
            'xM (m)': 'xM',
            'xN (m)': 'xN',
            'Dev. M': 'dev_m',
            'Dev. M (mV/V)': 'dev_m',
            'VMN (mV)': 'vmn',
            'IAB (mA)': 'iab'
        }
        
        # Renommer les colonnes
        df = df.rename(columns=column_mapping)
        
        # Vérifier les colonnes requises
        required_cols = CONFIG.geophysical_data.required_columns
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:  # Need at least coordinates and one measurement
            logger.warning(f"Device {device_name}: Missing required columns. Available: {df.columns.tolist()}")
            logger.warning(f"Required: {required_cols}")
            
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
            
            # Vérifier si le dispositif existe dans la configuration
            if device_name in CONFIG.geophysical_data.devices:
                expected_coverage = CONFIG.geophysical_data.devices[device_name]['coverage']
                logger.info(f"Device {device_name}: Coverage {x_range:.1f}m x {y_range:.1f}m")
            else:
                logger.warning(f"Device {device_name} not found in configuration, skipping coverage validation")
        
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

    
    def prepare_data_for_generators(self, csv_file: Path, device_type: str = "pole_dipole") -> Dict[str, torch.Tensor]:
        """
        Préparer les données nettoyées pour les générateurs U-Net 2D et VoxNet 3D.
        
        Args:
            csv_file: Chemin vers le fichier CSV nettoyé
            device_type: Type de dispositif géophysique
            
        Returns:
            Dict contenant les tenseurs préparés pour chaque générateur
        """
        try:
            # Charger les données nettoyées
            df = pd.read_csv(csv_file)
            logger.info(f"Préparation des données pour les générateurs: {len(df)} points")
            
            # Préparer les données pour U-Net 2D
            unet_2d_data = self._prepare_unet_2d_data(df, device_type)
            
            # Préparer les données pour VoxNet 3D
            voxnet_3d_data = self._prepare_voxnet_3d_data(df, device_type)
            
            return {
                'unet_2d': unet_2d_data,
                'voxnet_3d': voxnet_3d_data,
                'metadata': {
                    'device_type': device_type,
                    'num_points': len(df),
                    'spatial_bounds': self._get_spatial_bounds(df),
                    'value_ranges': self._get_value_ranges(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données pour les générateurs: {e}")
            raise
    
    def _prepare_unet_2d_data(self, df: pd.DataFrame, device_type: str) -> torch.Tensor:
        """Préparer les données pour U-Net 2D (grille 2D 64x64x4)."""
        config = self.generator_config['unet_2d']
        height, width, channels = config['input_size']
        
        # Créer une grille 2D
        grid_2d = self._create_2d_grid(df, height, width, channels)
        
        # Convertir en tenseur PyTorch
        tensor_2d = torch.from_numpy(grid_2d).float().to(self.device)
        
        logger.info(f"Données U-Net 2D préparées: {tensor_2d.shape}")
        return tensor_2d
    
    def _prepare_voxnet_3d_data(self, df: pd.DataFrame, device_type: str) -> torch.Tensor:
        """Préparer les données pour VoxNet 3D (volume 3D 32x32x32x4)."""
        config = self.generator_config['voxnet_3d']
        depth, height, width, channels = config['input_size']
        
        # Créer un volume 3D
        volume_3d = self._create_3d_volume(df, depth, height, width, channels)
        
        # Convertir en tenseur PyTorch
        tensor_3d = torch.from_numpy(volume_3d).float().to(self.device)
        
        logger.info(f"Données VoxNet 3D préparées: {tensor_3d.shape}")
        return tensor_3d
    
    def _create_2d_grid(self, df: pd.DataFrame, height: int, width: int, channels: int) -> np.ndarray:
        """Créer une grille 2D à partir des données CSV."""
        # Vérifier si le DataFrame est vide
        if df.empty:
            # Retourner une grille vide remplie de zéros
            return np.zeros((height, width, channels))

        # Vérifier si les colonnes nécessaires existent
        required_columns = ['x', 'y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Retourner une grille vide si les colonnes essentielles manquent
            return np.zeros((height, width, channels))

        # Calculer les limites spatiales
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()

        # Créer la grille
        grid = np.zeros((height, width, channels))

        # Version ultra-optimisée : interpolation par grille régulière
        # Créer les coordonnées de la grille
        grid_x_coords = np.linspace(x_min, x_max, width)
        grid_y_coords = np.linspace(y_min, y_max, height)

        # Échantillonnage très agressif pour les tests
        max_points = min(100, len(df))  # Réduit de 1000 à 100
        sample_df = df.sample(n=max_points, random_state=42) if len(df) > max_points else df

        # Interpolation vectorisée ultra-rapide
        # Créer des grilles de coordonnées
        X, Y = np.meshgrid(grid_x_coords, grid_y_coords)
        
        # Convertir en coordonnées 1D pour le calcul
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        data_points = sample_df[['x', 'y']].values
        
        # Calculer les distances de manière vectorisée
        from scipy.spatial.distance import cdist
        distances = cdist(grid_points, data_points)
        closest_indices = np.argmin(distances, axis=1)
        
        # Remplir la grille de manière vectorisée
        grid_flat = grid.reshape(-1, channels)
        
        for i, idx in enumerate(closest_indices):
            if 'x' in sample_df.columns:
                grid_flat[i, 0] = sample_df.iloc[idx]['x']
            if 'y' in sample_df.columns:
                grid_flat[i, 1] = sample_df.iloc[idx]['y']
            if 'resistivity' in sample_df.columns:
                grid_flat[i, 2] = sample_df.iloc[idx]['resistivity']
            if 'chargeability' in sample_df.columns:
                grid_flat[i, 3] = sample_df.iloc[idx]['chargeability']

        return grid
    
    def _create_3d_volume(self, df: pd.DataFrame, depth: int, height: int, width: int, channels: int) -> np.ndarray:
        """Créer un volume 3D à partir des données CSV."""
        # Vérifier si le DataFrame est vide
        if df.empty:
            # Retourner un volume vide rempli de zéros
            return np.zeros((depth, height, width, channels))
        
        # Vérifier si les colonnes nécessaires existent
        required_columns = ['x', 'y', 'z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Retourner un volume vide si les colonnes essentielles manquent
            return np.zeros((depth, height, width, channels))
        
        # Calculer les limites spatiales
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        z_min, z_max = df['z'].min(), df['z'].max()
        
        # Créer le volume
        volume = np.zeros((depth, height, width, channels))
        
        # Version ultra-optimisée : interpolation par grille régulière 3D
        # Créer les coordonnées du volume
        vol_x_coords = np.linspace(x_min, x_max, width)
        vol_y_coords = np.linspace(y_min, y_max, height)
        vol_z_coords = np.linspace(z_min, z_max, depth)
        
        # Échantillonnage très agressif pour les tests
        max_points = min(50, len(df))  # Réduit de 500 à 50
        sample_df = df.sample(n=max_points, random_state=42) if len(df) > max_points else df
        
        # Interpolation vectorisée ultra-rapide 3D
        # Créer des grilles de coordonnées 3D
        Z, Y, X = np.meshgrid(vol_z_coords, vol_y_coords, vol_x_coords, indexing='ij')
        
        # Convertir en coordonnées 1D pour le calcul
        volume_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        data_points = sample_df[['x', 'y', 'z']].values
        
        # Calculer les distances de manière vectorisée
        from scipy.spatial.distance import cdist
        distances = cdist(volume_points, data_points)
        closest_indices = np.argmin(distances, axis=1)
        
        # Remplir le volume de manière vectorisée
        volume_flat = volume.reshape(-1, channels)
        
        for i, idx in enumerate(closest_indices):
            if 'x' in sample_df.columns:
                volume_flat[i, 0] = sample_df.iloc[idx]['x']
            if 'y' in sample_df.columns:
                volume_flat[i, 1] = sample_df.iloc[idx]['y']
            if 'z' in sample_df.columns:
                volume_flat[i, 2] = sample_df.iloc[idx]['z']
            if 'chargeability' in sample_df.columns:
                volume_flat[i, 3] = sample_df.iloc[idx]['chargeability']
        
        return volume
    
    def _get_spatial_bounds(self, df: pd.DataFrame) -> Dict:
        """Obtenir les limites spatiales des données."""
        bounds = {}
        
        if 'x' in df.columns:
            bounds['x'] = {'min': df['x'].min(), 'max': df['x'].max()}
        if 'y' in df.columns:
            bounds['y'] = {'min': df['y'].min(), 'max': df['y'].max()}
        if 'z' in df.columns:
            bounds['z'] = {'min': df['z'].min(), 'max': df['z'].max()}
        
        return bounds
    
    def generate_synthetic_data_for_training(self, num_samples: int = 1000, device_type: str = "pole_dipole") -> Dict[str, torch.Tensor]:
        """
        Générer des données synthétiques pour l'entraînement des générateurs.
        
        Args:
            num_samples: Nombre d'échantillons à générer
            device_type: Type de dispositif géophysique
            
        Returns:
            Dict contenant les tenseurs synthétiques pour chaque générateur
        """
        logger.info(f"Génération de {num_samples} échantillons synthétiques pour {device_type}")
        
        # Générer des données synthétiques
        synthetic_df = self._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Préparer pour les générateurs
        return self.prepare_data_for_generators_from_df(synthetic_df, device_type)
    
    def _generate_synthetic_geophysical_data(self, num_samples: int, device_type: str) -> pd.DataFrame:
        """Générer des données géophysiques synthétiques."""
        # Paramètres de base
        x_range = (500000, 510000)  # UTM X
        y_range = (450000, 460000)  # UTM Y
        z_range = (500, 600)        # Profondeur
        
        # Générer les coordonnées
        x = np.random.uniform(x_range[0], x_range[1], num_samples)
        y = np.random.uniform(y_range[0], y_range[1], num_samples)
        z = np.random.uniform(z_range[0], z_range[1], num_samples)
        
        # Générer les mesures géophysiques selon le type de dispositif
        if device_type == "pole_dipole":
            resistivity = np.random.lognormal(mean=2.0, sigma=1.0, size=num_samples)
            chargeability = np.random.exponential(scale=50.0, size=num_samples)
        elif device_type == "schlumberger":
            resistivity = np.random.lognormal(mean=1.5, sigma=0.8, size=num_samples)
            chargeability = np.random.exponential(scale=40.0, size=num_samples)
        else:
            # Valeurs par défaut si le type n'est pas reconnu
            resistivity = np.random.lognormal(mean=1.8, sigma=0.9, size=num_samples)
            chargeability = np.random.exponential(scale=45.0, size=num_samples)
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'resistivity': resistivity,
            'chargeability': chargeability
        })
        
        return df
    
    def prepare_data_for_generators_from_df(self, df: pd.DataFrame, device_type: str) -> Dict[str, torch.Tensor]:
        """Préparer les données à partir d'un DataFrame pour les générateurs."""
        # Préparer les données pour U-Net 2D
        unet_2d_data = self._prepare_unet_2d_data(df, device_type)
        
        # Préparer les données pour VoxNet 3D
        voxnet_3d_data = self._prepare_voxnet_3d_data(df, device_type)
        
        return {
            'unet_2d': unet_2d_data,
            'voxnet_3d': voxnet_3d_data,
            'metadata': {
                'device_type': device_type,
                'num_points': len(df),
                'spatial_bounds': self._get_spatial_bounds(df),
                'value_ranges': self._get_value_ranges(df)
            }
        }
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des opérations de nettoyage effectuées."""
        if not self.report:
            return {
                'status': 'no_cleaning_performed',
                'message': 'Aucun nettoyage n\'a encore été effectué',
                'devices_processed': 0,
                'total_records_processed': 0,
                'total_records_removed': 0
            }
        
        # Calculer les statistiques globales
        total_original = sum(report.get('original_count', 0) for report in self.report.values())
        total_cleaned = sum(report.get('cleaned_count', 0) for report in self.report.values())
        total_removed = total_original - total_cleaned
        
        return {
            'status': 'cleaning_completed',
            'devices_processed': len(self.report),
            'devices': list(self.report.keys()),
            'total_records_processed': total_original,
            'total_records_cleaned': total_cleaned,
            'total_records_removed': total_removed,
            'removal_rate': (total_removed / total_original * 100) if total_original > 0 else 0,
            'details': self.report
        }

