"""
Tests unitaires pour la méthode _create_dummy_data de GeophysicalDataCleaner.
Utilise des données réelles pour tester la création de données factices.
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


class TestDataCleanerCreateDummyDataRealData(unittest.TestCase):
    """Tests pour la méthode _create_dummy_data avec données réelles."""
    
    def setUp(self):
        """Configuration des tests."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.test_dir) / "processed"
        self.processed_dir.mkdir()
        
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
    def test_create_dummy_data_success(self, mock_config):
        """Test de création réussie de données factices."""
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
        
        # Tester la création de données factices
        results = cleaner._create_dummy_data()
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertIn('dummy', results, "Devrait contenir une entrée 'dummy'")
        
        # Vérifier la structure du résultat
        dummy_entry = results['dummy']
        self.assertIsInstance(dummy_entry, tuple)
        self.assertEqual(len(dummy_entry), 2, "Devrait contenir (clean_path, report)")
        
        clean_path, report = dummy_entry
        
        # Vérifier le chemin du fichier
        self.assertIsInstance(clean_path, Path)
        self.assertTrue(clean_path.exists(), "Le fichier de données factices devrait exister")
        self.assertEqual(clean_path.name, "dummy_cleaned.csv", "Le nom du fichier devrait être 'dummy_cleaned.csv'")
        
        # Vérifier le rapport
        self.assertIsInstance(report, dict)
        self.assertIn('original_count', report)
        self.assertIn('cleaned_count', report)
        self.assertIn('removed_count', report)
        
        # Vérifier les valeurs du rapport
        self.assertIsInstance(report['original_count'], int)
        self.assertIsInstance(report['cleaned_count'], int)
        self.assertIsInstance(report['removed_count'], int)
        
        self.assertGreater(report['original_count'], 0, "Devrait avoir un nombre d'enregistrements original > 0")
        self.assertGreater(report['cleaned_count'], 0, "Devrait avoir un nombre d'enregistrements nettoyés > 0")
        self.assertEqual(report['removed_count'], 0, "Ne devrait pas supprimer d'enregistrements pour des données factices")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_create_dummy_data_file_content(self, mock_config):
        """Test du contenu du fichier de données factices."""
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
        
        # Tester la création de données factices
        results = cleaner._create_dummy_data()
        clean_path, report = results['dummy']
        
        # Vérifier le contenu du fichier
        df = pd.read_csv(clean_path)
        
        # Vérifier les colonnes requises
        required_cols = ['x', 'y', 'z', 'resistivity', 'chargeability']
        for col in required_cols:
            self.assertIn(col, df.columns, f"La colonne {col} devrait être présente")
        
        # Vérifier le nombre d'enregistrements
        self.assertEqual(len(df), report['original_count'], "Le nombre d'enregistrements devrait correspondre au rapport")
        self.assertEqual(len(df), report['cleaned_count'], "Le nombre d'enregistrements nettoyés devrait correspondre")
        
        # Vérifier les types de données
        self.assertTrue(pd.api.types.is_numeric_dtype(df['x']), "La colonne x devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['y']), "La colonne y devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['z']), "La colonne z devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['resistivity']), "La colonne resistivity devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['chargeability']), "La colonne chargeability devrait être numérique")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_create_dummy_data_value_ranges(self, mock_config):
        """Test des plages de valeurs des données factices."""
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
        
        # Tester la création de données factices
        results = cleaner._create_dummy_data()
        clean_path, report = results['dummy']
        
        # Vérifier le contenu du fichier
        df = pd.read_csv(clean_path)
        
        # Vérifier les plages de valeurs pour les coordonnées UTM
        self.assertTrue((df['x'] >= 500000).all(), "Toutes les valeurs x devraient être >= 500000")
        self.assertTrue((df['x'] <= 510000).all(), "Toutes les valeurs x devraient être <= 510000")
        self.assertTrue((df['y'] >= 450000).all(), "Toutes les valeurs y devraient être >= 450000")
        self.assertTrue((df['y'] <= 460000).all(), "Toutes les valeurs y devraient être <= 460000")
        self.assertTrue((df['z'] >= 500).all(), "Toutes les valeurs z devraient être >= 500")
        self.assertTrue((df['z'] <= 600).all(), "Toutes les valeurs z devraient être <= 600")
        
        # Vérifier les plages de valeurs pour les mesures géophysiques
        self.assertTrue((df['resistivity'] > 0).all(), "Toutes les valeurs de résistivité devraient être > 0")
        self.assertTrue((df['resistivity'] <= 1e9).all(), "Toutes les valeurs de résistivité devraient être <= 1e9")
        self.assertTrue((df['chargeability'] >= 0).all(), "Toutes les valeurs de chargeabilité devraient être >= 0")
        self.assertTrue((df['chargeability'] <= 200).all(), "Toutes les valeurs de chargeabilité devraient être <= 200")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_create_dummy_data_no_missing_values(self, mock_config):
        """Test qu'il n'y a pas de valeurs manquantes."""
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
        
        # Tester la création de données factices
        results = cleaner._create_dummy_data()
        clean_path, report = results['dummy']
        
        # Vérifier le contenu du fichier
        df = pd.read_csv(clean_path)
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        self.assertFalse(df.isnull().any().any(), "Ne devrait pas y avoir de valeurs manquantes")
        
        # Vérifier chaque colonne individuellement
        for col in df.columns:
            self.assertFalse(df[col].isnull().any(), f"La colonne {col} ne devrait pas avoir de valeurs manquantes")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_create_dummy_data_consistency(self, mock_config):
        """Test de la cohérence des données factices."""
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
        
        # Tester la création de données factices
        results = cleaner._create_dummy_data()
        clean_path, report = results['dummy']
        
        # Vérifier le contenu du fichier
        df = pd.read_csv(clean_path)
        
        # Vérifier la cohérence du rapport
        self.assertEqual(report['original_count'], report['cleaned_count'], "Le nombre d'enregistrements original et nettoyé devrait être identique")
        self.assertEqual(report['removed_count'], 0, "Aucun enregistrement ne devrait être supprimé")
        
        # Vérifier que le fichier est lisible
        self.assertTrue(clean_path.exists(), "Le fichier devrait exister")
        self.assertTrue(clean_path.is_file(), "Le fichier devrait être un fichier")
        self.assertGreater(clean_path.stat().st_size, 0, "Le fichier ne devrait pas être vide")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_create_dummy_data_multiple_calls(self, mock_config):
        """Test de plusieurs appels à la méthode."""
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
        
        # Tester plusieurs appels
        results1 = cleaner._create_dummy_data()
        results2 = cleaner._create_dummy_data()
        
        # Vérifier que les deux appels fonctionnent
        self.assertIsInstance(results1, dict)
        self.assertIsInstance(results2, dict)
        
        # Vérifier que les deux contiennent une entrée 'dummy'
        self.assertIn('dummy', results1)
        self.assertIn('dummy', results2)
        
        # Vérifier que les fichiers existent
        clean_path1, report1 = results1['dummy']
        clean_path2, report2 = results2['dummy']
        
        self.assertTrue(clean_path1.exists())
        self.assertTrue(clean_path2.exists())
        
        # Vérifier que les rapports sont cohérents
        self.assertEqual(report1['removed_count'], 0)
        self.assertEqual(report2['removed_count'], 0)


if __name__ == '__main__':
    unittest.main()
