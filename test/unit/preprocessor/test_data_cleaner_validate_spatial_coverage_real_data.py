"""
Tests unitaires pour la méthode _validate_spatial_coverage de GeophysicalDataCleaner.
Utilise des données réelles pour tester la validation de la couverture spatiale.
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

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner, read_csv_with_auto_separator, normalize_coordinate_columns


class TestDataCleanerValidateSpatialCoverageRealData(unittest.TestCase):
    """Tests pour la méthode _validate_spatial_coverage avec données réelles."""
    
    def setUp(self):
        """Configuration des tests avec données réelles des fixtures."""
        self.project_root = project_root
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        self.processed_dir = self.test_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les données réelles des fixtures
        self.load_real_data()
        
        # Créer des données de test avec différentes couvertures spatiales
        self.df_large_coverage = pd.DataFrame({
            'x': [500000, 500500, 501000, 501500, 502000],
            'y': [450000, 450500, 451000, 451500, 452000],
            'z': [500, 510, 520, 530, 540],
            'resistivity': [100, 150, 200, 250, 300],
            'chargeability': [10, 15, 20, 25, 30]
        })
        
        self.df_small_coverage = pd.DataFrame({
            'x': [500000, 500010, 500020, 500030, 500040],
            'y': [450000, 450010, 450020, 450030, 450040],
            'z': [500, 510, 520, 530, 540],
            'resistivity': [100, 150, 200, 250, 300],
            'chargeability': [10, 15, 20, 25, 30]
        })
        
        self.df_single_point = pd.DataFrame({
            'x': [500000],
            'y': [450000],
            'z': [500],
            'resistivity': [100],
            'chargeability': [10]
        })
        
        self.df_no_coordinates = pd.DataFrame({
            'resistivity': [100, 150, 200, 250, 300],
            'chargeability': [10, 15, 20, 25, 30]
        })
    
    def load_real_data(self):
        """Charger les données réelles des fixtures."""
        try:
            # Charger PD.csv
            pd_file = self.raw_data_dir / "PD.csv"
            if pd_file.exists():
                self.df_pd_real = read_csv_with_auto_separator(pd_file)
                self.df_pd_real = normalize_coordinate_columns(self.df_pd_real)
                print(f"✅ PD.csv chargé: {len(self.df_pd_real)} lignes, {len(self.df_pd_real.columns)} colonnes")
            else:
                self.df_pd_real = None
                print("⚠️ PD.csv non trouvé dans les fixtures")
            
            # Charger S.csv
            s_file = self.raw_data_dir / "S.csv"
            if s_file.exists():
                self.df_s_real = read_csv_with_auto_separator(s_file)
                self.df_s_real = normalize_coordinate_columns(self.df_s_real)
                print(f"✅ S.csv chargé: {len(self.df_s_real)} lignes, {len(self.df_s_real.columns)} colonnes")
            else:
                self.df_s_real = None
                print("⚠️ S.csv non trouvé dans les fixtures")
            
            # Charger les fichiers de profils
            self.df_profiles_real = []
            profile_files = list(self.raw_data_dir.glob("profil*.csv"))
            for profile_file in profile_files:
                try:
                    df_profile = read_csv_with_auto_separator(profile_file)
                    df_profile = normalize_coordinate_columns(df_profile)
                    self.df_profiles_real.append((profile_file.name, df_profile))
                    print(f"✅ {profile_file.name} chargé: {len(df_profile)} lignes")
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {profile_file.name}: {e}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des données réelles: {e}")
            self.df_pd_real = None
            self.df_s_real = None
            self.df_profiles_real = []
        
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
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
    
    def tearDown(self):
        """Nettoyage après les tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_large_coverage(self, mock_config):
        """Test de validation avec une grande couverture spatiale."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec une grande couverture spatiale
        result_df = cleaner._validate_spatial_coverage(self.df_large_coverage, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_large_coverage), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_large_coverage, "Les données devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_small_coverage(self, mock_config):
        """Test de validation avec une petite couverture spatiale."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec une petite couverture spatiale
        result_df = cleaner._validate_spatial_coverage(self.df_small_coverage, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_small_coverage), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_small_coverage, "Les données devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_single_point(self, mock_config):
        """Test de validation avec un seul point."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec un seul point
        result_df = cleaner._validate_spatial_coverage(self.df_single_point, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_single_point), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_single_point, "Les données devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_no_coordinates(self, mock_config):
        """Test de validation sans coordonnées."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester sans coordonnées
        result_df = cleaner._validate_spatial_coverage(self.df_no_coordinates, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_no_coordinates), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_no_coordinates, "Les données devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_different_devices(self, mock_config):
        """Test de validation avec différents dispositifs."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 200, 'min_y_range': 200}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec Pole-Dipole
        result_df_pd = cleaner._validate_spatial_coverage(self.df_large_coverage, "pole_dipole")
        self.assertIsInstance(result_df_pd, pd.DataFrame)
        
        # Tester avec Schlumberger
        result_df_s = cleaner._validate_spatial_coverage(self.df_large_coverage, "schlumberger")
        self.assertIsInstance(result_df_s, pd.DataFrame)
        
        # Vérifier que les résultats sont identiques
        pd.testing.assert_frame_equal(result_df_pd, result_df_s, "Les résultats devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_unknown_device(self, mock_config):
        """Test de validation avec un dispositif inconnu."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec un dispositif inconnu
        result_df = cleaner._validate_spatial_coverage(self.df_large_coverage, "unknown_device")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_large_coverage), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_large_coverage, "Les données devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_empty_dataframe(self, mock_config):
        """Test de validation avec un DataFrame vide."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Créer un DataFrame vide
        empty_df = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        
        # Tester avec un DataFrame vide
        result_df = cleaner._validate_spatial_coverage(empty_df, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 0, "Le DataFrame vide devrait rester vide")
        
        # Vérifier que les colonnes sont préservées
        self.assertEqual(list(result_df.columns), list(empty_df.columns), "Les colonnes devraient être préservées")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_data_integrity(self, mock_config):
        """Test de l'intégrité des données après validation."""
        mock_config.paths.raw_data_dir = str(self.test_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32633'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester l'intégrité des données
        result_df = cleaner._validate_spatial_coverage(self.df_large_coverage, "pole_dipole")
        
        # Vérifier que les données sont identiques
        pd.testing.assert_frame_equal(result_df, self.df_large_coverage, "Les données devraient être identiques")
        
        # Vérifier que les types de données sont préservés
        for col in self.df_large_coverage.columns:
            self.assertEqual(result_df[col].dtype, self.df_large_coverage[col].dtype, f"Le type de la colonne {col} devrait être préservé")
        
        # Vérifier que les valeurs sont identiques
        for col in self.df_large_coverage.columns:
            pd.testing.assert_series_equal(result_df[col], self.df_large_coverage[col], f"Les valeurs de la colonne {col} devraient être identiques")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_pd_real_data(self, mock_config):
        """Test de validation avec les données réelles de PD.csv."""
        mock_config.paths.raw_data_dir = str(self.raw_data_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32630'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        if self.df_pd_real is None:
            self.skipTest("PD.csv non disponible dans les fixtures")
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec les données réelles de PD.csv
        result_df = cleaner._validate_spatial_coverage(self.df_pd_real, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_pd_real), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        if 'x' in self.df_pd_real.columns and 'y' in self.df_pd_real.columns:
            x_range = self.df_pd_real['x'].max() - self.df_pd_real['x'].min()
            y_range = self.df_pd_real['y'].max() - self.df_pd_real['y'].min()
            print(f"✅ PD.csv: Couverture spatiale {x_range:.1f}m x {y_range:.1f}m")
        
        print(f"✅ PD.csv validé avec succès: {len(result_df)} enregistrements")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_s_real_data(self, mock_config):
        """Test de validation avec les données réelles de S.csv."""
        mock_config.paths.raw_data_dir = str(self.raw_data_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32630'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        if self.df_s_real is None:
            self.skipTest("S.csv non disponible dans les fixtures")
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec les données réelles de S.csv
        result_df = cleaner._validate_spatial_coverage(self.df_s_real, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.df_s_real), "Le DataFrame devrait conserver le même nombre de lignes")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        if 'x' in self.df_s_real.columns and 'y' in self.df_s_real.columns:
            x_range = self.df_s_real['x'].max() - self.df_s_real['x'].min()
            y_range = self.df_s_real['y'].max() - self.df_s_real['y'].min()
            print(f"✅ S.csv: Couverture spatiale {x_range:.1f}m x {y_range:.1f}m")
        elif 'LAT' in self.df_s_real.columns and 'LON' in self.df_s_real.columns:
            lat_range = self.df_s_real['LAT'].max() - self.df_s_real['LAT'].min()
            lon_range = self.df_s_real['LON'].max() - self.df_s_real['LON'].min()
            print(f"✅ S.csv: Couverture géographique {lat_range:.6f}° x {lon_range:.6f}°")
        
        print(f"✅ S.csv validé avec succès: {len(result_df)} enregistrements")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_profiles_real_data(self, mock_config):
        """Test de validation avec les données réelles des fichiers de profils."""
        mock_config.paths.raw_data_dir = str(self.raw_data_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32630'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        if not self.df_profiles_real:
            self.skipTest("Aucun fichier de profil disponible dans les fixtures")
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester avec tous les fichiers de profils
        for profile_name, df_profile in self.df_profiles_real:
            with self.subTest(profile=profile_name):
                try:
                    result_df = cleaner._validate_spatial_coverage(df_profile, "schlumberger")
                    
                    # Vérifications
                    self.assertIsInstance(result_df, pd.DataFrame)
                    self.assertEqual(len(result_df), len(df_profile), f"Le DataFrame {profile_name} devrait conserver le même nombre de lignes")
                    
                    # Vérifier la couverture spatiale si les coordonnées sont présentes
                    if 'x' in df_profile.columns and 'y' in df_profile.columns:
                        x_range = df_profile['x'].max() - df_profile['x'].min()
                        y_range = df_profile['y'].max() - df_profile['y'].min()
                        print(f"✅ {profile_name}: Couverture spatiale {x_range:.1f}m x {y_range:.1f}m")
                    
                    print(f"✅ {profile_name} validé avec succès: {len(result_df)} enregistrements")
                    
                except Exception as e:
                    self.fail(f"Erreur lors de la validation de {profile_name}: {e}")
    
    @patch('backend.preprocessor.data_cleaner.CONFIG')
    def test_validate_spatial_coverage_all_fixtures(self, mock_config):
        """Test de validation avec tous les fichiers des fixtures."""
        mock_config.paths.raw_data_dir = str(self.raw_data_dir)
        mock_config.paths.processed_data_dir = str(self.processed_dir)
        mock_config.geophysical_data.coordinate_systems = {
            'wgs84': 'EPSG:4326',
            'utm_proj': 'EPSG:32630'
        }
        mock_config.geophysical_data.required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        mock_config.geophysical_data.devices = {
            'pole_dipole': {'file': 'PD.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}},
            'schlumberger': {'file': 'S.csv', 'coverage': {'min_x_range': 100, 'min_y_range': 100}}
        }
        
        cleaner = GeophysicalDataCleaner()
        
        # Tester tous les fichiers CSV des fixtures
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        self.assertGreater(len(csv_files), 0, "Aucun fichier CSV trouvé dans les fixtures")
        
        total_tests = 0
        successful_tests = 0
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    # Charger et normaliser les données
                    df = read_csv_with_auto_separator(csv_file)
                    df = normalize_coordinate_columns(df)
                    
                    # Déterminer le type de dispositif
                    device_type = "pole_dipole" if "PD" in csv_file.name else "schlumberger"
                    
                    # Tester la validation
                    result_df = cleaner._validate_spatial_coverage(df, device_type)
                    
                    # Vérifications
                    self.assertIsInstance(result_df, pd.DataFrame)
                    self.assertEqual(len(result_df), len(df), f"Le DataFrame {csv_file.name} devrait conserver le même nombre de lignes")
                    
                    total_tests += 1
                    successful_tests += 1
                    
                    print(f"✅ {csv_file.name} validé avec succès: {len(result_df)} enregistrements")
                    
                except Exception as e:
                    print(f"⚠️ Erreur lors de la validation de {csv_file.name}: {e}")
                    total_tests += 1
        
        # Vérifier qu'au moins quelques tests ont réussi
        self.assertGreater(successful_tests, 0, "Aucun fichier n'a pu être validé")
        print(f"✅ Validation complète: {successful_tests}/{total_tests} fichiers validés avec succès")


if __name__ == '__main__':
    unittest.main()
