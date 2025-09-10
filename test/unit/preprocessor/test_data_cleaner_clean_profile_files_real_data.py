"""
Tests unitaires pour la méthode _clean_profile_files de GeophysicalDataCleaner.
Utilise des données réelles pour tester le nettoyage des fichiers de profils.
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


class TestDataCleanerCleanProfileFilesRealData(unittest.TestCase):
    """Tests pour la méthode _clean_profile_files avec données réelles."""
    
    def setUp(self):
        """Configuration des tests."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.test_dir) / "processed"
        self.processed_dir.mkdir()
        
        # Créer des fichiers de profils de test
        self.profile1_file = Path(self.test_dir) / "profil_1.csv"
        self.profile2_file = Path(self.test_dir) / "profil_2.csv"
        self.profile3_file = Path(self.test_dir) / "profil_3.csv"
        
        # Créer des données de test réalistes
        profile1_data = pd.DataFrame({
            'x': [500000, 500100, 500200, 500300],
            'y': [450000, 450100, 450200, 450300],
            'z': [500, 510, 520, 530],
            'resistivity': [100, 150, 200, 250],
            'chargeability': [10, 15, 20, 25]
        })
        
        profile2_data = pd.DataFrame({
            'x': [501000, 501100, 501200, 501300],
            'y': [451000, 451100, 451200, 451300],
            'z': [500, 510, 520, 530],
            'resistivity': [120, 170, 220, 270],
            'chargeability': [12, 17, 22, 27]
        })
        
        profile3_data = pd.DataFrame({
            'x': [502000, 502100, 502200, 502300],
            'y': [452000, 452100, 452200, 452300],
            'z': [500, 510, 520, 530],
            'resistivity': [130, 180, 230, 280],
            'chargeability': [13, 18, 23, 28]
        })
        
        profile1_data.to_csv(self.profile1_file, index=False)
        profile2_data.to_csv(self.profile2_file, index=False)
        profile3_data.to_csv(self.profile3_file, index=False)
        
        # Mock CONFIG pour les tests
        self.config_mock = MagicMock()
        self.config_mock.paths.raw_data_dir = str(self.test_dir)
        self.config_mock.paths.processed_data_dir = str(self.processed_dir)
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
    def test_clean_profile_files_success(self, mock_config):
        """Test de nettoyage réussi des fichiers de profils."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester le nettoyage des fichiers de profils
        results = cleaner._clean_profile_files()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0, "Devrait traiter au moins un fichier de profil")
        
        # Vérifier que les fichiers nettoyés existent
        for profile_name, (clean_path, report) in results.items():
            self.assertIsInstance(clean_path, Path)
            self.assertIsInstance(report, dict)
            self.assertTrue(clean_path.exists(), f"Le fichier nettoyé {clean_path} devrait exister")
            
            # Vérifier le contenu du fichier nettoyé
            df_clean = pd.read_csv(clean_path)
            self.assertGreater(len(df_clean), 0, f"Le fichier nettoyé {profile_name} devrait contenir des données")
            
            # Vérifier les colonnes requises
            required_cols = ['x', 'y', 'z', 'resistivity', 'chargeability']
            for col in required_cols:
                self.assertIn(col, df_clean.columns, f"La colonne {col} devrait être présente dans {profile_name}")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_files_no_files(self, mock_config):
        """Test avec aucun fichier de profil."""
        empty_dir = Path(self.test_dir) / "empty"
        empty_dir.mkdir()
        
        mock_config.paths.raw_data_dir = str(empty_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec aucun fichier
        results = cleaner._clean_profile_files()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0, "Ne devrait traiter aucun fichier")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_files_nonexistent_directory(self, mock_config):
        """Test avec un répertoire inexistant."""
        nonexistent_dir = Path(self.test_dir) / "nonexistent"
        
        mock_config.paths.raw_data_dir = str(nonexistent_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
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
        cleaner.raw_data_dir = nonexistent_dir
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec un répertoire inexistant
        results = cleaner._clean_profile_files()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0, "Ne devrait traiter aucun fichier")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_files_error_handling(self, mock_config):
        """Test de gestion des erreurs lors du nettoyage."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv'},
            'schlumberger': {'file': 'S.csv'}
        }
        
        # Créer un fichier CSV invalide
        invalid_file = Path(self.test_dir) / "invalid.csv"
        with open(invalid_file, 'w') as f:
            f.write("invalid,csv,content\n")
        
        cleaner = GeophysicalDataCleaner()
        cleaner.raw_data_dir = Path(self.test_dir)
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec un fichier invalide
        results = cleaner._clean_profile_files()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        # Devrait traiter les fichiers valides et ignorer les invalides
        self.assertGreaterEqual(len(results), 0, "Devrait gérer les erreurs gracieusement")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_files_multiple_profiles(self, mock_config):
        """Test avec plusieurs fichiers de profils."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec plusieurs fichiers
        results = cleaner._clean_profile_files()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 3, "Devrait traiter au moins 3 fichiers de profils")
        
        # Vérifier que tous les profils sont traités
        profile_names = list(results.keys())
        self.assertIn('profil_1', profile_names)
        self.assertIn('profil_2', profile_names)
        self.assertIn('profil_3', profile_names)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_files_report_structure(self, mock_config):
        """Test de la structure du rapport de nettoyage."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester le nettoyage
        results = cleaner._clean_profile_files()
        
        # Vérifier la structure du rapport
        for profile_name, (clean_path, report) in results.items():
            self.assertIsInstance(report, dict)
            self.assertIn('original_count', report)
            self.assertIn('cleaned_count', report)
            self.assertIn('removed_count', report)
            
            # Vérifier les valeurs du rapport
            self.assertIsInstance(report['original_count'], int)
            self.assertIsInstance(report['cleaned_count'], int)
            self.assertIsInstance(report['removed_count'], int)
            
            self.assertGreater(report['original_count'], 0)
            self.assertGreater(report['cleaned_count'], 0)
            self.assertGreaterEqual(report['removed_count'], 0)


if __name__ == '__main__':
    unittest.main()
