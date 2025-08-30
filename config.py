from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Paths:
    """
    Stocke les configurations des chemins de fichiers pour le projet AI-MAP.
    """
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    raw_data_dir: Path = BASE_DIR / "data/raw"
    processed_data_dir: Path = BASE_DIR / "data/processed"
    intermediate_data_dir: Path = BASE_DIR / "data/intermediate"

@dataclass
class GeophysicalDataConfig:
    """
    Configuration pour la gestion des données géophysiques.
    """
    # Configurations des dispositifs
    devices = {
        'pole_pole': {
            'file': 'profil 1.csv',
            'measurements': 164,
            'coverage': (950, 450),
            'features': ['resistivity', 'chargeability']
        },
        'pole_dipole': {
            'file': 'PD_Line1s.dat',
            'measurements': 144,
            'coverage': (1000, 'moderate'),
            'features': ['resistivity', 'chargeability']
        },
        'schlumberger_6': {
            'file': 'PRO 6 COMPLET.csv',
            'measurements': 469,
            'coverage': (945, 94),
            'features': ['resistivity', 'chargeability']
        },
        'schlumberger_7': {
            'file': 'PRO 7 COMPLET.csv',
            'measurements': 100,
            'coverage': (180, 31),
            'features': ['resistivity', 'chargeability']
        }
    }
    
    required_columns: List[str] = field(default_factory=lambda: [
        'x', 'y', 'z', 'resistivity', 'chargeability'
    ])
    
    coordinate_systems = {
        'utm_zone': "30N",  # À adapter selon votre zone
        'wgs84': "EPSG:4326",
        'utm_proj': "EPSG:32630"
    }

@dataclass
class ProcessingConfig:
    """
    Configuration pour le prétraitement des données.
    """
    # Configuration des grilles
    grid_2d: Tuple[int, int] = (64, 64)  # Pour U-Net 2D
    grid_3d: Tuple[int, int, int] = (32, 32, 32)  # Pour VoxNet 3D
    
    # Paramètres de normalisation
    resistivity_range: Tuple[float, float] = (1e-8, 1e9)
    chargeability_range: Tuple[float, float] = (0, 200)
    
    # Traitement spatial
    interpolation_method: str = "linear"
    coordinate_tolerance: float = 1.0  # mètres
    
    # Division des données
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

@dataclass
class CNNConfig:
    """
    Configuration pour les modèles CNN.
    """
    # Configuration U-Net 2D
    unet_2d = {
        'input_shape': (64, 64, 4),  # 4 canaux pour les dispositifs
        'encoder_filters': [64, 128, 256, 512, 1024],
        'decoder_filters': [512, 256, 128, 64],
        'output_channels': 2,  # résistivité + chargeabilité
        'dropout_rate': 0.3
    }
    
    # Configuration VoxNet 3D
    voxnet_3d = {
        'input_shape': (32, 32, 32, 4),
        'conv_filters': [32, 64, 128],
        'output_channels': 1,  # chargeabilité uniquement
        'dropout_rate': 0.3
    }
    
    # Paramètres d'entraînement
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-4
    early_stopping_patience: int = 20

@dataclass
class WebAppConfig:
    """
    Configuration pour l'application web.
    """
    backend_port: int = 5000
    frontend_port: int = 3000
    debug_mode: bool = True
    
    # Configuration de la base de données
    database_url: str = "postgresql://localhost/aimap_db"
    
    # Upload de fichiers
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.csv', '.dat', '.bin', '.txt'
    ])

@dataclass
class GeneralConfig:
    """
    L'objet de configuration principal pour AI-MAP.
    """
    paths: Paths = field(default_factory=Paths)
    geophysical_data: GeophysicalDataConfig = field(default_factory=GeophysicalDataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    webapp: WebAppConfig = field(default_factory=WebAppConfig)

CONFIG = GeneralConfig()

# Exemple d'utilisation:
# print(CONFIG.geophysical_data.devices['pole_pole'])
# print(CONFIG.processing.grid_2d)
# print(CONFIG.cnn.unet_2d['input_shape'])
