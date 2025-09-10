#!/usr/bin/env python3
"""
Test unitaire pour la fonction __init__ de GeophysicalDataCleaner

Ce test vérifie que l'initialisation de la classe fonctionne correctement
avec des données réelles (PD.csv, S.csv) et des chemins réels.
"""

import sys
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestGeophysicalDataCleanerInit(unittest.TestCase):
    """Tests pour la fonction __init__ de GeophysicalDataCleaner avec données réelles"""
    
    def setUp(self):
        """Configuration avant chaque test avec données réelles"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Vérifier que les fichiers de données réels existent
        self.pd_file = self.raw_data_dir / "PD.csv"
        self.s_file = self.raw_data_dir / "S.csv"
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer QUE le dossier processed temporaire, PAS le dossier fixtures
        import shutil
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_import_class(self):
        """Test que la classe peut être importée"""
        self.assertIsNotNone(GeophysicalDataCleaner)
        self.assertTrue(hasattr(GeophysicalDataCleaner, '__init__'))
    
    def test_create_instance_with_real_data_paths(self):
        """Test de création d'instance avec des chemins de données réels"""
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'LAT', 'LON', 'El-array'
            ]
            mock_config.geophysical_data.devices = {
                'pole_dipole': {'coverage': 1000},
                'schlumberger': {'coverage': 1000}
            }
            
            cleaner = GeophysicalDataCleaner()
            
            # Vérifications
            self.assertIsInstance(cleaner, GeophysicalDataCleaner)
            self.assertIsNotNone(cleaner)
            self.assertEqual(cleaner.raw_data_dir, self.raw_data_dir)
            self.assertEqual(cleaner.processed_data_dir, self.test_dir / "processed")
            
            print(f"✅ Instance créée avec chemins réels: {cleaner.raw_data_dir}")
    
    def test_initialization_with_real_csv_files(self):
        """Test d'initialisation avec vérification des fichiers CSV réels"""
        # Vérifier que les fichiers de données réels existent
        self.assertTrue(self.pd_file.exists(), f"Fichier PD.csv manquant: {self.pd_file}")
        self.assertTrue(self.s_file.exists(), f"Fichier S.csv manquant: {self.s_file}")
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'LAT', 'LON', 'El-array'
            ]
            mock_config.geophysical_data.devices = {
                'pole_dipole': {'coverage': 1000},
                'schlumberger': {'coverage': 1000}
            }
            
            cleaner = GeophysicalDataCleaner()
            
            # Vérifier que les fichiers peuvent être lus
            pd_df = pd.read_csv(self.pd_file, sep=';')
            s_df = pd.read_csv(self.s_file, sep=';')
            
            self.assertGreater(len(pd_df), 0, "PD.csv ne devrait pas être vide")
            self.assertGreater(len(s_df), 0, "S.csv ne devrait pas être vide")
            
            print(f"✅ Fichiers réels vérifiés: PD.csv ({len(pd_df)} lignes), S.csv ({len(s_df)} lignes)")
    
    def test_supported_devices_initialization(self):
        """Test d'initialisation des dispositifs supportés avec données réelles"""
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'LAT', 'LON', 'El-array'
            ]
            mock_config.geophysical_data.devices = {
                'pole_dipole': {'coverage': 1000},
                'schlumberger': {'coverage': 1000}
            }
            
            cleaner = GeophysicalDataCleaner()
            
            # Vérifier les dispositifs supportés
            self.assertIn('pole_dipole', cleaner.supported_devices)
            self.assertIn('schlumberger', cleaner.supported_devices)
            self.assertEqual(len(cleaner.supported_devices), 2)
            
            # Vérifier les informations des dispositifs
            pd_info = cleaner.supported_devices['pole_dipole']
            self.assertEqual(pd_info['name'], 'Pole-Dipole')
            self.assertIn('electrodes', pd_info)
            self.assertIn('measurements', pd_info)
            
            schl_info = cleaner.supported_devices['schlumberger']
            self.assertEqual(schl_info['name'], 'Schlumberger')
            self.assertIn('electrodes', schl_info)
            self.assertIn('measurements', schl_info)
            
            print(f"✅ Dispositifs supportés initialisés: {list(cleaner.supported_devices.keys())}")
    
    def test_generator_config_initialization(self):
        """Test d'initialisation de la configuration des générateurs"""
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'LAT', 'LON', 'El-array'
            ]
            mock_config.geophysical_data.devices = {
                'pole_dipole': {'coverage': 1000},
                'schlumberger': {'coverage': 1000}
            }
            
            cleaner = GeophysicalDataCleaner()
            
            # Vérifier la configuration des générateurs
            self.assertIn('unet_2d', cleaner.generator_config)
            self.assertIn('voxnet_3d', cleaner.generator_config)
            
            # Vérifier la configuration U-Net 2D
            unet_config = cleaner.generator_config['unet_2d']
            self.assertEqual(unet_config['input_size'], (64, 64, 4))
            self.assertEqual(unet_config['output_channels'], 2)
            self.assertEqual(unet_config['spatial_resolution'], 1.0)
            
            # Vérifier la configuration VoxNet 3D
            voxnet_config = cleaner.generator_config['voxnet_3d']
            self.assertEqual(voxnet_config['input_size'], (32, 32, 32, 4))
            self.assertEqual(voxnet_config['output_channels'], 1)
            self.assertEqual(voxnet_config['spatial_resolution'], 2.0)
            
            print(f"✅ Configuration des générateurs initialisée: U-Net 2D et VoxNet 3D")
    
    def test_attributes_exist(self):
        """Test que tous les attributs requis existent"""
        cleaner = GeophysicalDataCleaner()
        
        # Attributs attendus
        expected_attributes = [
            'report',
            'raw_data_dir', 
            'processed_data_dir',
            'coord_transformer'
        ]
        
        for attr in expected_attributes:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(cleaner, attr), f"Attribut '{attr}' manquant")
    
    def test_report_attribute(self):
        """Test de l'attribut report"""
        cleaner = GeophysicalDataCleaner()
        self.assertIsInstance(cleaner.report, dict)
        self.assertEqual(len(cleaner.report), 0)  # Rapport vide au début
    
    def test_raw_data_dir_attribute(self):
        """Test de l'attribut raw_data_dir"""
        cleaner = GeophysicalDataCleaner()
        self.assertIsInstance(cleaner.raw_data_dir, Path)
        # Vérifier que le chemin existe et est un répertoire
        self.assertTrue(cleaner.raw_data_dir.exists(), "Le répertoire raw_data_dir devrait exister")
        self.assertTrue(cleaner.raw_data_dir.is_dir(), "raw_data_dir devrait être un répertoire")
    
    def test_processed_data_dir_attribute(self):
        """Test de l'attribut processed_data_dir"""
        cleaner = GeophysicalDataCleaner()
        self.assertIsInstance(cleaner.processed_data_dir, Path)
        # Compatible Windows et Unix
        processed_path_str = str(cleaner.processed_data_dir)
        self.assertTrue(processed_path_str.endswith("data\\processed") or processed_path_str.endswith("data/processed") or "data/processed" in processed_path_str)
    
    def test_coord_transformer_attribute(self):
        """Test de l'attribut coord_transformer"""
        cleaner = GeophysicalDataCleaner()
        self.assertIsNotNone(cleaner.coord_transformer)
        # Vérifier que c'est un transformateur pyproj
        self.assertTrue(hasattr(cleaner.coord_transformer, 'transform'))
    
    def test_processed_dir_creation(self):
        """Test que le dossier processed est créé automatiquement"""
        # Supprimer le dossier s'il existe
        processed_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
        
        # Créer une instance (doit créer le dossier)
        cleaner = GeophysicalDataCleaner()
        
        # Vérifier que le dossier a été créé
        self.assertTrue(cleaner.processed_data_dir.exists())
    
    def test_coordinate_transformation(self):
        """Test de la transformation de coordonnées"""
        cleaner = GeophysicalDataCleaner()
        
        # Coordonnées de test (Paris)
        lat_test = 48.8566
        lon_test = 2.3522
        
        # Transformer les coordonnées
        x, y = cleaner.coord_transformer.transform(lon_test, lat_test)
        
        # Vérifications
        self.assertIsInstance(x, (int, float))
        self.assertIsInstance(y, (int, float))
        self.assertGreater(x, 0)  # Coordonnées UTM positives
        self.assertGreater(y, 0)
        
        # Vérifier que les coordonnées sont dans des plages raisonnables
        # Zone UTM 30N (Europe de l'Ouest)
        self.assertGreater(x, 500000)  # X > 500km
        self.assertGreater(y, 5000000)  # Y > 5000km
    
    def test_multiple_instances(self):
        """Test que plusieurs instances peuvent être créées"""
        cleaner1 = GeophysicalDataCleaner()
        cleaner2 = GeophysicalDataCleaner()
        
        self.assertIsNot(cleaner1, cleaner2)
        self.assertNotEqual(id(cleaner1), id(cleaner2))
        
        # Vérifier que les attributs sont indépendants
        self.assertIsNot(cleaner1.report, cleaner2.report)
    
    def test_public_methods_available(self):
        """Test que les méthodes publiques sont disponibles"""
        cleaner = GeophysicalDataCleaner()
        
        # Méthodes publiques attendues
        expected_methods = [
            'clean_all_devices',
            'prepare_data_for_generators',
            'generate_synthetic_data_for_training',
            'prepare_data_for_generators_from_df'
        ]
        
        for method in expected_methods:
            with self.subTest(method=method):
                self.assertTrue(hasattr(cleaner, method), f"Méthode '{method}' manquante")
                method_obj = getattr(cleaner, method)
                self.assertTrue(callable(method_obj), f"'{method}' n'est pas appelable")


class TestGeophysicalDataCleanerMethods(unittest.TestCase):
    """Tests pour les méthodes principales de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.cleaner = GeophysicalDataCleaner()
    
    def test_clean_all_devices_with_real_files(self):
        """Test de clean_all_devices avec les vrais fichiers CSV"""
        results = self.cleaner.clean_all_devices()
        
        # Devrait retourner un dictionnaire avec les dispositifs traités
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0, "Aucun dispositif traité")
        
        # Vérifier que les dispositifs attendus sont présents
        expected_devices = ['pole_dipole', 'schlumberger']
        for device in expected_devices:
            self.assertIn(device, results, f"Dispositif {device} manquant dans les résultats")
        
        # Vérifier la structure des résultats
        for device_name, (clean_path, report) in results.items():
            self.assertIsInstance(clean_path, Path)
            self.assertIsInstance(report, dict)
            # Vérifier les clés communes du rapport
            self.assertIn('original_count', report)
            self.assertIn('cleaned_count', report)
            self.assertIn('removed_count', report)
    
    def test_prepare_data_for_generators(self):
        """Test de prepare_data_for_generators"""
        # Créer un fichier CSV de test
        test_file = Path("test_data.csv")
        test_data = pd.DataFrame({
            'x': [500000, 500100, 500200],
            'y': [450000, 450100, 450200],
            'z': [500, 510, 520],
            'resistivity': [100, 150, 200],
            'chargeability': [10, 15, 20]
        })
        test_data.to_csv(test_file, index=False)
        
        try:
            # Tester la méthode
            result = self.cleaner.prepare_data_for_generators(test_file, "pole_dipole")
            
            # Vérifications
            self.assertIsInstance(result, dict)
            self.assertIn('unet_2d', result)
            self.assertIn('voxnet_3d', result)
            self.assertIn('metadata', result)
        finally:
            # Nettoyer
            if test_file.exists():
                test_file.unlink()
    
    def test_methods_return_types(self):
        """Test des types de retour des méthodes"""
        # clean_all_devices
        results = self.cleaner.clean_all_devices()
        self.assertIsInstance(results, dict)
        
        # generate_synthetic_data_for_training
        synthetic_data = self.cleaner.generate_synthetic_data_for_training(10, "pole_dipole")
        self.assertIsInstance(synthetic_data, dict)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
