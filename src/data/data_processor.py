import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from config import CONFIG
from src.utils.logger import logger


class GeophysicalDataProcessor:
    """
    Processeur pour les données géophysiques, inspiré du DataProcessor d'EMUT.
    """
    
    def __init__(self):
        self.scalers = {}
        self.spatial_grids = {}
        self.device_data = {}
        
    def load_and_validate(self) -> Dict[str, pd.DataFrame]:
        """Charger et valider les données de tous les dispositifs."""
        logger.info("Chargement et validation des données géophysiques...")
        
        # D'abord essayer les fichiers nettoyés
        processed_dir = Path(CONFIG.paths.processed_data_dir)
        clean_files = list(processed_dir.glob("*_cleaned.csv"))
        
        if clean_files:
            for clean_file in clean_files:
                device_name = clean_file.stem.replace("_cleaned", "")
                try:
                    df = pd.read_csv(clean_file)
                    logger.info(f"Chargé {len(df)} enregistrements pour {device_name}")
                    self.device_data[device_name] = df
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement de {clean_file}: {e}")
        else:
            # Si pas de fichiers nettoyés, utiliser les fichiers bruts
            logger.info("Aucun fichier nettoyé trouvé, utilisation des fichiers bruts...")
            raw_dir = Path(CONFIG.paths.raw_data_dir)
            
            if raw_dir.exists():
                for csv_file in raw_dir.glob("*.csv"):
                    device_name = csv_file.stem
                    try:
                        # Lire avec le bon séparateur
                        df = pd.read_csv(csv_file, sep=',')
                        logger.info(f"Chargé {len(df)} enregistrements pour {device_name}")
                        self.device_data[device_name] = df
                    except Exception as e:
                        logger.warning(f"Erreur lors du chargement de {csv_file}: {e}")
            else:
                logger.warning("Aucun fichier de données trouvé")
                
        return self.device_data
    
    def create_spatial_grids(self) -> Dict[str, np.ndarray]:
        """Créer des grilles spatiales pour chaque dispositif."""
        logger.info("Création des grilles spatiales pour les dispositifs...")
        
        for device_name, df in self.device_data.items():
            grid_2d = self._create_2d_grid(df, device_name)
            self.spatial_grids[device_name] = grid_2d
            
        return self.spatial_grids
    
    def _create_2d_grid(self, df: pd.DataFrame, device_name: str) -> np.ndarray:
        """Créer une grille 2D pour les données du dispositif."""
        grid_shape = CONFIG.processing.grid_2d
        
        # Vérifier si les colonnes nécessaires existent
        required_cols = ['x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Essayer d'utiliser les colonnes de coordonnées alternatives
            if 'LAT' in df.columns and 'LON' in df.columns:
                logger.info(f"Utilisation des coordonnées LAT/LON pour {device_name}")
                # Convertir LAT/LON en coordonnées cartésiennes approximatives
                df = df.copy()
                df['x'] = df['LON'] * 111320 * np.cos(np.radians(df['LAT'].mean()))  # Approximation
                df['y'] = df['LAT'] * 110540  # Approximation
                df['z'] = 0  # Profondeur par défaut
                x_min, x_max = df['x'].min(), df['x'].max()
                y_min, y_max = df['y'].min(), df['y'].max()
            else:
                logger.warning(f"Colonnes manquantes pour {device_name}: {missing_cols}")
                # Créer des données factices si les colonnes manquent
                x_min, x_max = 0, 100
                y_min, y_max = 0, 100
        else:
            # Obtenir les limites spatiales
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            
            # Normaliser les coordonnées si elles sont trop grandes
            if x_max - x_min > 1000 or y_max - y_min > 1000:
                logger.info(f"Normalisation des coordonnées pour {device_name}")
                df = df.copy()
                df['x'] = (df['x'] - x_min) / (x_max - x_min) * 100
                df['y'] = (df['y'] - y_min) / (y_max - y_min) * 100
                x_min, x_max = 0, 100
                y_min, y_max = 0, 100
        
        # Créer les coordonnées de la grille
        x_grid = np.linspace(x_min, x_max, grid_shape[0])
        y_grid = np.linspace(y_min, y_max, grid_shape[1])
        
        # Initialiser la grille avec des zéros
        grid = np.zeros((grid_shape[0], grid_shape[1], 4))
        
        # Interpolation simple - remplir avec le plus proche voisin
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                if missing_cols:
                    # Si les colonnes manquent, utiliser des valeurs aléatoires
                    grid[i, j, 0] = np.random.uniform(1e-8, 1e9)
                    grid[i, j, 1] = np.random.uniform(0, 200)
                    grid[i, j, 2] = np.random.uniform(-100, 100)
                    grid[i, j, 3] = np.random.uniform(0, 100)
                else:
                    # Trouver le point de données le plus proche
                    distances = np.sqrt((df['x'] - x)**2 + (df['y'] - y)**2)
                    nearest_idx = np.argmin(distances)
                    
                    # Remplir la grille avec les mesures
                    # Mapper les colonnes réelles aux indices de la grille
                    if 'Rho(ohm.m)' in df.columns:
                        grid[i, j, 0] = df.iloc[nearest_idx]['Rho(ohm.m)']
                    elif 'resistivity' in df.columns:
                        grid[i, j, 0] = df.iloc[nearest_idx]['resistivity']
                        
                    if 'M (mV/V)' in df.columns:
                        grid[i, j, 1] = df.iloc[nearest_idx]['M (mV/V)']
                    elif 'chargeability' in df.columns:
                        grid[i, j, 1] = df.iloc[nearest_idx]['chargeability']
                        
                    grid[i, j, 2] = df.iloc[nearest_idx]['x']
                    grid[i, j, 3] = df.iloc[nearest_idx]['y']
        
        logger.debug(f"Grille {grid_shape} créée pour {device_name}")
        return grid
    
    def create_multi_device_tensor(self) -> np.ndarray:
        """Créer un tenseur multi-dispositifs pour l'entrée CNN."""
        logger.info("Création du tenseur multi-dispositifs...")
        
        # Créer d'abord les grilles spatiales
        spatial_grids = self.create_spatial_grids()
        
        # Empiler toutes les grilles des dispositifs
        device_tensors = []
        for device_name in CONFIG.geophysical_data.devices.keys():
            if device_name in spatial_grids:
                device_tensors.append(spatial_grids[device_name])
        
        if not device_tensors:
            # Créer un tenseur factice si aucune donnée
            height, width = CONFIG.processing.grid_2d
            channels = 4
            dummy_tensor = np.zeros((1, height, width, channels))
            logger.warning("Aucune donnée de dispositif trouvée, tenseur factice créé")
            return dummy_tensor
        
        # Empiler le long de la dimension batch
        multi_device_tensor = np.stack(device_tensors, axis=0)
        
        logger.info(f"Tenseur multi-dispositifs créé: {multi_device_tensor.shape}")
        return multi_device_tensor
    
    def create_3d_volume(self) -> np.ndarray:
        """Créer un volume 3D pour l'entrée VoxNet."""
        logger.info("Création du volume 3D pour VoxNet...")
        
        grid_3d = CONFIG.processing.grid_3d
        channels = 4
        
        # Créer un volume 3D avec les canaux en premier (format PyTorch)
        volume = np.zeros((channels, grid_3d[0], grid_3d[1], grid_3d[2]))
        
        # Obtenir le tenseur 2D et l'étendre en 3D
        multi_device_tensor = self.create_multi_device_tensor()
        
        if multi_device_tensor.shape[0] > 0:
            # Utiliser le premier dispositif comme base et l'étendre en 3D
            base_2d = multi_device_tensor[0]  # (height, width, channels)
            
            # Redimensionner la grille 2D pour correspondre au volume 3D
            from scipy.ndimage import zoom
            zoom_factors = (grid_3d[1]/base_2d.shape[0], grid_3d[2]/base_2d.shape[1], 1)
            base_2d_resized = zoom(base_2d, zoom_factors, order=1)
            
            # Dupliquer à travers la profondeur pour chaque canal
            for c in range(channels):
                for d in range(grid_3d[0]):
                    volume[c, d] = base_2d_resized[:, :, c]
        
        logger.info(f"Volume 3D créé: {volume.shape}")
        return volume
    
    def split_data(self, tensor: np.ndarray, labels: np.ndarray = None) -> Tuple:
        """Diviser les données en ensembles d'entraînement et de test."""
        # Division simple pour l'instant
        split_idx = int(len(tensor) * 0.8)
        x_train = tensor[:split_idx]
        x_test = tensor[split_idx:]
        
        logger.info(f"Taille de l'ensemble d'entraînement: {len(x_train)}")
        logger.info(f"Taille de l'ensemble de test: {len(x_test)}")
        
        return x_train, x_test
    
    def get_data_summary(self) -> Dict:
        """Obtenir un résumé des données traitées."""
        summary = {
            'devices_processed': len(self.device_data),
            'spatial_grids_created': len(self.spatial_grids),
            'scalers_created': len(self.scalers),
            'device_details': {}
        }
        
        for device_name, df in self.device_data.items():
            summary['device_details'][device_name] = {
                'record_count': len(df),
                'spatial_coverage': {
                    'x_range': df['x'].max() - df['x'].min(),
                    'y_range': df['y'].max() - df['y'].min()
                }
            }
        
        return summary
