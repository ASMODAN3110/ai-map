#!/usr/bin/env python3
"""
Test unitaire pour la fonction __init__ de GeophysicalDataCleaner

Ce test vérifie que l'initialisation de la classe fonctionne correctement.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestGeophysicalDataCleanerInit(unittest.TestCase):
    """Tests pour la fonction __init__ de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser le dossier fixtures existant (ne pas le recréer)
        self.test_data_dir = Path(__file__).parent.parent.parent.parent / "test" / "fixtures"
        
        # Créer seulement le dossier processed s'il n'existe pas
        self.test_processed_dir = self.test_data_dir / "processed"
        self.test_processed_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer QUE le dossier processed temporaire, PAS le dossier fixtures
        import shutil
        if self.test_processed_dir.exists():
            shutil.rmtree(self.test_processed_dir)
    
    def test_import_class(self):
        """Test que la classe peut être importée"""
        self.assertIsNotNone(GeophysicalDataCleaner)
        self.assertTrue(hasattr(GeophysicalDataCleaner, '__init__'))
    
    def test_create_instance(self):
        """Test de création d'instance"""
        cleaner = GeophysicalDataCleaner()
        self.assertIsInstance(cleaner, GeophysicalDataCleaner)
        self.assertIsNotNone(cleaner)
    
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
        # Compatible Windows et Unix
        raw_path_str = str(cleaner.raw_data_dir)
        self.assertTrue(raw_path_str.endswith("data\\raw") or raw_path_str.endswith("data/raw") or "data/raw" in raw_path_str)
    
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
            'get_cleaning_summary'
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
            self.assertIn('device', report)
            self.assertEqual(report['device'], device_name)
    
    def test_get_cleaning_summary_empty(self):
        """Test de get_cleaning_summary avec aucun nettoyage"""
        summary = self.cleaner.get_cleaning_summary()
        
        # Devrait retourner un dictionnaire vide au début
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary), 0)
    
    def test_methods_return_types(self):
        """Test des types de retour des méthodes"""
        # clean_all_devices
        results = self.cleaner.clean_all_devices()
        self.assertIsInstance(results, dict)
        
        # get_cleaning_summary
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
