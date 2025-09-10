"""
Tests unitaires pour la méthode _clean_profile_data de GeophysicalDataCleaner.
Utilise des données réelles pour tester le nettoyage des données de profil.
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


class TestDataCleanerCleanProfileDataRealData(unittest.TestCase):
    """Tests pour la méthode _clean_profile_data avec données réelles."""
    
    def setUp(self):
        """Configuration des tests."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.test_dir) / "processed"
        self.processed_dir.mkdir()
        
        # Créer des fichiers de test
        self.profile_file = Path(self.test_dir) / "test_profile.csv"
        self.profile_file_comma = Path(self.test_dir) / "test_profile_comma.csv"
        self.profile_file_semicolon = Path(self.test_dir) / "test_profile_semicolon.csv"
        
        # Créer des données de test réalistes
        profile_data = pd.DataFrame({
            'x': [500000, 500100, 500200, 500300, 500400],
            'y': [450000, 450100, 450200, 450300, 450400],
            'z': [500, 510, 520, 530, 540],
            'resistivity': [100, 150, 200, 250, 300],
            'chargeability': [10, 15, 20, 25, 30]
        })
        
        # Données avec mapping de colonnes
        profile_data_mapped = pd.DataFrame({
            'Rho(ohm.m)': [100, 150, 200, 250, 300],
            'M (mV/V)': [10, 15, 20, 25, 30],
            'xA (m)': [500000, 500100, 500200, 500300, 500400],
            'xB (m)': [450000, 450100, 450200, 450300, 450400],
            'xM (m)': [500, 510, 520, 530, 540]
        })
        
        # Données avec valeurs manquantes
        profile_data_missing = pd.DataFrame({
            'x': [500000, None, 500200, 500300, 500400],
            'y': [450000, 450100, None, 450300, 450400],
            'z': [500, 510, 520, None, 540],
            'resistivity': [100, 150, None, 250, 300],
            'chargeability': [10, None, 20, 25, 30]
        })
        
        # Données avec valeurs aberrantes
        profile_data_outliers = pd.DataFrame({
            'x': [500000, 500100, 500200, 500300, 500400],
            'y': [450000, 450100, 450200, 450300, 450400],
            'z': [500, 510, 520, 530, 540],
            'resistivity': [100, 150, 200, 250, -50],  # Valeur négative
            'chargeability': [10, 15, 20, 25, -5]      # Valeur négative
        })
        
        # Sauvegarder les fichiers
        profile_data.to_csv(self.profile_file, index=False)
        profile_data_mapped.to_csv(self.profile_file_comma, index=False)
        profile_data_missing.to_csv(self.profile_file_semicolon, index=False)
        profile_data_outliers.to_csv(Path(self.test_dir) / "test_profile_outliers.csv", index=False)
        
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
    def test_clean_profile_data_success(self, mock_config):
        """Test de nettoyage réussi des données de profil."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester le nettoyage
        clean_path, report = cleaner._clean_profile_data("test_profile", self.profile_file)
        
        # Vérifications
        self.assertIsInstance(clean_path, Path)
        self.assertIsInstance(report, dict)
        self.assertTrue(clean_path.exists(), "Le fichier nettoyé devrait exister")
        
        # Vérifier le contenu du fichier nettoyé
        df_clean = pd.read_csv(clean_path)
        self.assertGreater(len(df_clean), 0, "Le fichier nettoyé devrait contenir des données")
        
        # Vérifier les colonnes requises
        required_cols = ['x', 'y', 'z', 'resistivity', 'chargeability']
        for col in required_cols:
            self.assertIn(col, df_clean.columns, f"La colonne {col} devrait être présente")
        
        # Vérifier la structure du rapport
        self.assertIn('original_count', report)
        self.assertIn('cleaned_count', report)
        self.assertIn('removed_count', report)
        
        self.assertIsInstance(report['original_count'], int)
        self.assertIsInstance(report['cleaned_count'], int)
        self.assertIsInstance(report['removed_count'], int)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_column_mapping(self, mock_config):
        """Test du mapping des colonnes."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec des colonnes mappées
        clean_path, report = cleaner._clean_profile_data("test_profile_mapped", self.profile_file_comma)
        
        # Vérifications
        self.assertTrue(clean_path.exists(), "Le fichier nettoyé devrait exister")
        
        df_clean = pd.read_csv(clean_path)
        self.assertGreater(len(df_clean), 0, "Le fichier nettoyé devrait contenir des données")
        
        # Vérifier que les colonnes ont été mappées
        expected_cols = ['x', 'y', 'z', 'resistivity', 'chargeability']
        for col in expected_cols:
            self.assertIn(col, df_clean.columns, f"La colonne {col} devrait être mappée")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_semicolon_separator(self, mock_config):
        """Test avec séparateur point-virgule."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec séparateur point-virgule
        clean_path, report = cleaner._clean_profile_data("test_profile_semicolon", self.profile_file_semicolon)
        
        # Vérifications
        self.assertTrue(clean_path.exists(), "Le fichier nettoyé devrait exister")
        
        df_clean = pd.read_csv(clean_path)
        self.assertGreater(len(df_clean), 0, "Le fichier nettoyé devrait contenir des données")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_missing_values(self, mock_config):
        """Test avec valeurs manquantes."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec valeurs manquantes
        clean_path, report = cleaner._clean_profile_data("test_profile_missing", self.profile_file_semicolon)
        
        # Vérifications
        self.assertTrue(clean_path.exists(), "Le fichier nettoyé devrait exister")
        
        df_clean = pd.read_csv(clean_path)
        self.assertGreater(len(df_clean), 0, "Le fichier nettoyé devrait contenir des données")
        
        # Vérifier que les valeurs manquantes ont été supprimées
        self.assertFalse(df_clean.isnull().any().any(), "Ne devrait pas y avoir de valeurs manquantes")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_outliers(self, mock_config):
        """Test avec valeurs aberrantes."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec valeurs aberrantes
        outliers_file = Path(self.test_dir) / "test_profile_outliers.csv"
        clean_path, report = cleaner._clean_profile_data("test_profile_outliers", outliers_file)
        
        # Vérifications
        self.assertTrue(clean_path.exists(), "Le fichier nettoyé devrait exister")
        
        df_clean = pd.read_csv(clean_path)
        self.assertGreater(len(df_clean), 0, "Le fichier nettoyé devrait contenir des données")
        
        # Vérifier que les valeurs aberrantes ont été supprimées
        self.assertTrue((df_clean['resistivity'] > 0).all(), "Toutes les valeurs de résistivité devraient être positives")
        self.assertTrue((df_clean['chargeability'] >= 0).all(), "Toutes les valeurs de chargeabilité devraient être non négatives")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_insufficient_columns(self, mock_config):
        """Test avec colonnes insuffisantes."""
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
        
        # Créer un fichier avec colonnes insuffisantes
        insufficient_file = Path(self.test_dir) / "insufficient.csv"
        insufficient_data = pd.DataFrame({
            'x': [500000, 500100],
            'y': [450000, 450100]
        })
        insufficient_data.to_csv(insufficient_file, index=False)
        
        cleaner = GeophysicalDataCleaner()
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec colonnes insuffisantes
        with self.assertRaises(ValueError):
            cleaner._clean_profile_data("insufficient", insufficient_file)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_file_not_found(self, mock_config):
        """Test avec fichier inexistant."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester avec fichier inexistant
        nonexistent_file = Path(self.test_dir) / "nonexistent.csv"
        
        with self.assertRaises(FileNotFoundError):
            cleaner._clean_profile_data("nonexistent", nonexistent_file)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_clean_profile_data_report_accuracy(self, mock_config):
        """Test de l'exactitude du rapport."""
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
        cleaner.processed_data_dir = self.processed_dir
        
        # Tester le nettoyage
        clean_path, report = cleaner._clean_profile_data("test_profile", self.profile_file)
        
        # Vérifier l'exactitude du rapport
        df_original = pd.read_csv(self.profile_file)
        df_clean = pd.read_csv(clean_path)
        
        self.assertEqual(report['original_count'], len(df_original))
        self.assertEqual(report['cleaned_count'], len(df_clean))
        self.assertEqual(report['removed_count'], len(df_original) - len(df_clean))
        
        # Vérifier la cohérence
        self.assertEqual(report['original_count'], report['cleaned_count'] + report['removed_count'])


if __name__ == '__main__':
    unittest.main()
