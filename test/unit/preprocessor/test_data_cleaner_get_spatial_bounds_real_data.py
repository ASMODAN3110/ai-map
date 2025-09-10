"""
Tests unitaires pour la méthode _get_spatial_bounds de GeophysicalDataCleaner.
Utilise des données réelles pour tester l'obtention des limites spatiales.
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


class TestDataCleanerGetSpatialBoundsRealData(unittest.TestCase):
    """Tests pour la méthode _get_spatial_bounds avec données réelles."""
    
    def setUp(self):
        """Configuration des tests."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.test_dir) / "processed"
        self.processed_dir.mkdir()
        
        # Créer des données de test avec différentes configurations spatiales
        self.df_full_coordinates = pd.DataFrame({
            'x': [500000, 500100, 500200, 500300, 500400],
            'y': [450000, 450100, 450200, 450300, 450400],
            'z': [500, 510, 520, 530, 540],
            'resistivity': [100, 150, 200, 250, 300],
            'chargeability': [10, 15, 20, 25, 30]
        })
        
        self.df_xy_only = pd.DataFrame({
            'x': [501000, 501100, 501200, 501300, 501400],
            'y': [451000, 451100, 451200, 451300, 451400],
            'resistivity': [120, 170, 220, 270, 320],
            'chargeability': [12, 17, 22, 27, 32]
        })
        
        self.df_x_only = pd.DataFrame({
            'x': [502000, 502100, 502200, 502300, 502400],
            'resistivity': [130, 180, 230, 280, 330],
            'chargeability': [13, 18, 23, 28, 33]
        })
        
        self.df_y_only = pd.DataFrame({
            'y': [452000, 452100, 452200, 452300, 452400],
            'resistivity': [140, 190, 240, 290, 340],
            'chargeability': [14, 19, 24, 29, 34]
        })
        
        self.df_z_only = pd.DataFrame({
            'z': [600, 610, 620, 630, 640],
            'resistivity': [150, 200, 250, 300, 350],
            'chargeability': [15, 20, 25, 30, 35]
        })
        
        self.df_no_coordinates = pd.DataFrame({
            'resistivity': [160, 210, 260, 310, 360],
            'chargeability': [16, 21, 26, 31, 36]
        })
        
        self.df_single_point = pd.DataFrame({
            'x': [500000],
            'y': [450000],
            'z': [500],
            'resistivity': [100],
            'chargeability': [10]
        })
        
        self.df_empty = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        
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
    def test_get_spatial_bounds_full_coordinates(self, mock_config):
        """Test avec toutes les coordonnées (x, y, z)."""
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
        
        # Tester avec toutes les coordonnées
        bounds = cleaner._get_spatial_bounds(self.df_full_coordinates)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertIn('x', bounds)
        self.assertIn('y', bounds)
        self.assertIn('z', bounds)
        
        # Vérifier les limites x
        self.assertIn('min', bounds['x'])
        self.assertIn('max', bounds['x'])
        self.assertEqual(bounds['x']['min'], 500000)
        self.assertEqual(bounds['x']['max'], 500400)
        
        # Vérifier les limites y
        self.assertIn('min', bounds['y'])
        self.assertIn('max', bounds['y'])
        self.assertEqual(bounds['y']['min'], 450000)
        self.assertEqual(bounds['y']['max'], 450400)
        
        # Vérifier les limites z
        self.assertIn('min', bounds['z'])
        self.assertIn('max', bounds['z'])
        self.assertEqual(bounds['z']['min'], 500)
        self.assertEqual(bounds['z']['max'], 540)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_xy_only(self, mock_config):
        """Test avec seulement les coordonnées x et y."""
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
        
        # Tester avec seulement x et y
        bounds = cleaner._get_spatial_bounds(self.df_xy_only)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertIn('x', bounds)
        self.assertIn('y', bounds)
        self.assertNotIn('z', bounds)
        
        # Vérifier les limites x
        self.assertEqual(bounds['x']['min'], 501000)
        self.assertEqual(bounds['x']['max'], 501400)
        
        # Vérifier les limites y
        self.assertEqual(bounds['y']['min'], 451000)
        self.assertEqual(bounds['y']['max'], 451400)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_x_only(self, mock_config):
        """Test avec seulement la coordonnée x."""
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
        
        # Tester avec seulement x
        bounds = cleaner._get_spatial_bounds(self.df_x_only)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertIn('x', bounds)
        self.assertNotIn('y', bounds)
        self.assertNotIn('z', bounds)
        
        # Vérifier les limites x
        self.assertEqual(bounds['x']['min'], 502000)
        self.assertEqual(bounds['x']['max'], 502400)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_y_only(self, mock_config):
        """Test avec seulement la coordonnée y."""
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
        
        # Tester avec seulement y
        bounds = cleaner._get_spatial_bounds(self.df_y_only)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertNotIn('x', bounds)
        self.assertIn('y', bounds)
        self.assertNotIn('z', bounds)
        
        # Vérifier les limites y
        self.assertEqual(bounds['y']['min'], 452000)
        self.assertEqual(bounds['y']['max'], 452400)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_z_only(self, mock_config):
        """Test avec seulement la coordonnée z."""
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
        
        # Tester avec seulement z
        bounds = cleaner._get_spatial_bounds(self.df_z_only)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertNotIn('x', bounds)
        self.assertNotIn('y', bounds)
        self.assertIn('z', bounds)
        
        # Vérifier les limites z
        self.assertEqual(bounds['z']['min'], 600)
        self.assertEqual(bounds['z']['max'], 640)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_no_coordinates(self, mock_config):
        """Test sans coordonnées."""
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
        
        # Tester sans coordonnées
        bounds = cleaner._get_spatial_bounds(self.df_no_coordinates)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        # La méthode retourne un dictionnaire avec les colonnes même si le DataFrame est vide
        self.assertEqual(len(bounds), 0, "Devrait retourner un dictionnaire vide pour un DataFrame sans coordonnées")
        self.assertNotIn('x', bounds)
        self.assertNotIn('y', bounds)
        self.assertNotIn('z', bounds)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_single_point(self, mock_config):
        """Test avec un seul point."""
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
        
        # Tester avec un seul point
        bounds = cleaner._get_spatial_bounds(self.df_single_point)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        self.assertIn('x', bounds)
        self.assertIn('y', bounds)
        self.assertIn('z', bounds)
        
        # Vérifier les limites (min = max pour un seul point)
        self.assertEqual(bounds['x']['min'], 500000)
        self.assertEqual(bounds['x']['max'], 500000)
        self.assertEqual(bounds['y']['min'], 450000)
        self.assertEqual(bounds['y']['max'], 450000)
        self.assertEqual(bounds['z']['min'], 500)
        self.assertEqual(bounds['z']['max'], 500)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_empty_dataframe(self, mock_config):
        """Test avec un DataFrame vide."""
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
        
        # Tester avec un DataFrame vide
        bounds = cleaner._get_spatial_bounds(self.df_empty)
        
        # Vérifications
        self.assertIsInstance(bounds, dict)
        # La méthode retourne un dictionnaire avec les colonnes même si le DataFrame est vide
        self.assertEqual(len(bounds), 3, "Devrait retourner un dictionnaire avec 3 colonnes pour un DataFrame vide avec colonnes de coordonnées")
        
        # Vérifier que les valeurs sont NaN pour un DataFrame vide
        for col in ['x', 'y', 'z']:
            self.assertIn(col, bounds)
            self.assertTrue(pd.isna(bounds[col]['min']))
            self.assertTrue(pd.isna(bounds[col]['max']))
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_data_types(self, mock_config):
        """Test des types de données retournés."""
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
        
        # Tester les types de données
        bounds = cleaner._get_spatial_bounds(self.df_full_coordinates)
        
        # Vérifier les types
        for coord in ['x', 'y', 'z']:
            self.assertIsInstance(bounds[coord]['min'], (int, float))
            self.assertIsInstance(bounds[coord]['max'], (int, float))
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_get_spatial_bounds_consistency(self, mock_config):
        """Test de la cohérence des résultats."""
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
        
        # Tester la cohérence
        bounds = cleaner._get_spatial_bounds(self.df_full_coordinates)
        
        # Vérifier que min <= max pour chaque coordonnée
        for coord in ['x', 'y', 'z']:
            self.assertLessEqual(bounds[coord]['min'], bounds[coord]['max'], f"min devrait être <= max pour {coord}")
        
        # Vérifier que les valeurs correspondent aux données
        self.assertEqual(bounds['x']['min'], self.df_full_coordinates['x'].min())
        self.assertEqual(bounds['x']['max'], self.df_full_coordinates['x'].max())
        self.assertEqual(bounds['y']['min'], self.df_full_coordinates['y'].min())
        self.assertEqual(bounds['y']['max'], self.df_full_coordinates['y'].max())
        self.assertEqual(bounds['z']['min'], self.df_full_coordinates['z'].min())
        self.assertEqual(bounds['z']['max'], self.df_full_coordinates['z'].max())


if __name__ == '__main__':
    unittest.main()
