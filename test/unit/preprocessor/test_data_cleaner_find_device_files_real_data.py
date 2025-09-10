"""
Tests unitaires pour la méthode _find_device_files de GeophysicalDataCleaner.
Utilise des données réelles pour tester la recherche de fichiers de dispositifs.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerFindDeviceFilesRealData(unittest.TestCase):
    """Tests pour la méthode _find_device_files avec données réelles."""
    
    def setUp(self):
        """Configuration des tests."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = tempfile.mkdtemp()
        
        # Créer des fichiers de test réels
        self.pd_file = Path(self.test_dir) / "PD.csv"
        self.s_file = Path(self.test_dir) / "S.csv"
        
        # Créer des données de test réalistes
        pd_data = pd.DataFrame({
            'x': [500000, 500100, 500200],
            'y': [450000, 450100, 450200],
            'z': [500, 510, 520],
            'resistivity': [100, 150, 200],
            'chargeability': [10, 15, 20]
        })
        
        s_data = pd.DataFrame({
            'x': [501000, 501100, 501200],
            'y': [451000, 451100, 451200],
            'z': [500, 510, 520],
            'resistivity': [120, 170, 220],
            'chargeability': [12, 17, 22]
        })
        
        pd_data.to_csv(self.pd_file, index=False)
        s_data.to_csv(self.s_file, index=False)
        
        # Mock CONFIG pour les tests
        self.config_mock = MagicMock()
        self.config_mock.paths.raw_data_dir = str(self.test_dir)
        self.config_mock.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        self.config_mock.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        self.config_mock.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        self.config_mock.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
    
    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_pole_dipole(self, mock_config):
        """Test de recherche des fichiers Pole-Dipole."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        
        # Tester la recherche de fichiers Pole-Dipole
        device_files = cleaner._find_device_files('pole_dipole')
        
        # Vérifications
        self.assertIsInstance(device_files, list)
        self.assertGreater(len(device_files), 0, "Devrait trouver au moins un fichier Pole-Dipole")
        
        # Vérifier que le fichier PD.csv est trouvé
        pd_found = any('PD' in str(f) for f in device_files)
        self.assertTrue(pd_found, "Le fichier PD.csv devrait être trouvé")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_schlumberger(self, mock_config):
        """Test de recherche des fichiers Schlumberger."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        
        # Tester la recherche de fichiers Schlumberger
        device_files = cleaner._find_device_files('schlumberger')
        
        # Vérifications
        self.assertIsInstance(device_files, list)
        self.assertGreater(len(device_files), 0, "Devrait trouver au moins un fichier Schlumberger")
        
        # Vérifier que le fichier S.csv est trouvé
        s_found = any('S' in str(f) for f in device_files)
        self.assertTrue(s_found, "Le fichier S.csv devrait être trouvé")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_unknown_device(self, mock_config):
        """Test de recherche pour un dispositif inconnu."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        
        # Tester la recherche pour un dispositif inconnu
        device_files = cleaner._find_device_files('unknown_device')
        
        # Vérifications
        self.assertIsInstance(device_files, list)
        self.assertEqual(len(device_files), 0, "Ne devrait trouver aucun fichier pour un dispositif inconnu")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_multiple_patterns(self, mock_config):
        """Test de recherche avec plusieurs patterns."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        # Créer des fichiers supplémentaires avec différents patterns
        pole_dipole_file = Path(self.test_dir) / "pole_dipole_data.csv"
        schlumberger_file = Path(self.test_dir) / "schlumberger_data.csv"
        
        pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'resistivity': [1], 'chargeability': [1]}).to_csv(pole_dipole_file, index=False)
        pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'resistivity': [1], 'chargeability': [1]}).to_csv(schlumberger_file, index=False)
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        
        # Tester la recherche Pole-Dipole
        pd_files = cleaner._find_device_files('pole_dipole')
        self.assertGreater(len(pd_files), 0, "Devrait trouver des fichiers Pole-Dipole")
        
        # Tester la recherche Schlumberger
        s_files = cleaner._find_device_files('schlumberger')
        self.assertGreater(len(s_files), 0, "Devrait trouver des fichiers Schlumberger")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_no_duplicates(self, mock_config):
        """Test que la méthode ne retourne pas de doublons."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        
        # Tester la recherche
        device_files = cleaner._find_device_files('pole_dipole')
        
        # Vérifier qu'il n'y a pas de doublons
        unique_files = list(set(device_files))
        self.assertEqual(len(device_files), len(unique_files), "Ne devrait pas y avoir de doublons")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_find_device_files_empty_directory(self, mock_config):
        """Test de recherche dans un répertoire vide."""
        empty_dir = Path(self.test_dir) / "empty"
        empty_dir.mkdir()
        
        mock_config.paths.raw_data_dir = str(empty_dir)
        mock_config.paths.processed_data_dir = str(Path(self.test_dir) / "processed")
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = empty_dir
        
        # Tester la recherche dans un répertoire vide
        device_files = cleaner._find_device_files('pole_dipole')
        
        # Vérifications
        self.assertIsInstance(device_files, list)
        self.assertEqual(len(device_files), 0, "Ne devrait trouver aucun fichier dans un répertoire vide")


if __name__ == '__main__':
    unittest.main()
