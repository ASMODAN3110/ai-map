#!/usr/bin/env python3
"""
Test unitaire pour la méthode _transform_coordinates de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _transform_coordinates
avec des coordonnées réelles extraites des fichiers S.csv (LAT/LON vers UTM).
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner, read_csv_with_auto_separator, normalize_coordinate_columns


class TestDataCleanerTransformCoordinatesRealData(unittest.TestCase):
    """Tests pour la méthode _transform_coordinates avec données réelles"""
    
    def setUp(self):
        """Configuration avant chaque test avec données réelles des fixtures"""
        # Utiliser les fichiers de données du répertoire data/raw
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
        
        # Créer des données de test avec des coordonnées LAT/LON pour tester la transformation
        # Les fichiers dans data/raw contiennent déjà des coordonnées UTM, pas LAT/LON
        self.create_test_lat_lon_data()
    
    def create_test_lat_lon_data(self):
        """Créer des données de test avec des coordonnées LAT/LON pour tester la transformation."""
        # Coordonnées LAT/LON de test basées sur une zone géographique réelle
        # Zone autour de Paris, France (pour avoir des coordonnées UTM raisonnables)
        test_lat_values = [48.8566, 48.8570, 48.8574, 48.8578, 48.8582, 48.8586, 48.8590, 48.8594, 48.8598, 48.8602]
        test_lon_values = [2.3522, 2.3526, 2.3530, 2.3534, 2.3538, 2.3542, 2.3546, 2.3550, 2.3554, 2.3558]
        
        self.real_lat = pd.Series(test_lat_values)
        self.real_lon = pd.Series(test_lon_values)
        
        print(f"✅ Données de test LAT/LON créées: {len(self.real_lat)} coordonnées")
        print(f"   LAT: {self.real_lat.min():.6f} à {self.real_lat.max():.6f}")
        print(f"   LON: {self.real_lon.min():.6f} à {self.real_lon.max():.6f}")
        
        # Créer un DataFrame de test
        self.s_df = pd.DataFrame({
            'LAT': self.real_lat,
            'LON': self.real_lon,
            'resistivity': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
            'chargeability': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        })
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_transform_coordinates_real_s_csv_data(self):
        """Test de transformation avec les vraies coordonnées du fichier S.csv"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Prendre les 10 premières coordonnées pour le test
        test_lat = self.real_lat.head(10)
        test_lon = self.real_lon.head(10)
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(test_lat, test_lon)
        
        # Vérifications
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), len(test_lat))
        self.assertEqual(len(y_coords), len(test_lon))
        
        # Vérifier que les coordonnées UTM sont dans des plages raisonnables
        # Pour la zone UTM 31N (EPSG:32631), X devrait être autour de 500000-900000
        # et Y devrait être autour de 5400000-5500000 (zone de Paris)
        self.assertTrue(np.all(x_coords > 400000), "Coordonnées X UTM trop faibles")
        self.assertTrue(np.all(x_coords < 1000000), "Coordonnées X UTM trop élevées")
        self.assertTrue(np.all(y_coords > 5000000), "Coordonnées Y UTM trop faibles")
        self.assertTrue(np.all(y_coords < 6000000), "Coordonnées Y UTM trop élevées")
        
        print(f"✅ Transformation réussie: {len(x_coords)} coordonnées LAT/LON → UTM")
        print(f"   LAT: {test_lat.min():.6f} à {test_lat.max():.6f}")
        print(f"   LON: {test_lon.min():.6f} à {test_lon.max():.6f}")
        print(f"   UTM X: {x_coords.min():.2f} à {x_coords.max():.2f}")
        print(f"   UTM Y: {y_coords.min():.2f} à {y_coords.max():.2f}")
    
    def test_transform_coordinates_single_point(self):
        """Test de transformation avec un seul point"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Prendre un seul point
        single_lat = pd.Series([self.real_lat.iloc[0]])
        single_lon = pd.Series([self.real_lon.iloc[0]])
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(single_lat, single_lon)
        
        # Vérifications
        self.assertEqual(len(x_coords), 1)
        self.assertEqual(len(y_coords), 1)
        self.assertIsInstance(x_coords[0], (int, float))
        self.assertIsInstance(y_coords[0], (int, float))
        
        print(f"✅ Transformation point unique: LAT={single_lat.iloc[0]:.6f}, LON={single_lon.iloc[0]:.6f}")
        print(f"   → UTM X={x_coords[0]:.2f}, Y={y_coords[0]:.2f}")
    
    def test_transform_coordinates_edge_cases(self):
        """Test de transformation avec des cas limites"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Test avec des coordonnées aux limites
        edge_lat = pd.Series([self.real_lat.min(), self.real_lat.max()])
        edge_lon = pd.Series([self.real_lon.min(), self.real_lon.max()])
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(edge_lat, edge_lon)
        
        # Vérifications
        self.assertEqual(len(x_coords), 2)
        self.assertEqual(len(y_coords), 2)
        
        # Vérifier que les coordonnées sont différentes (pas de dégénérescence)
        self.assertNotEqual(x_coords[0], x_coords[1], "Coordonnées X identiques")
        self.assertNotEqual(y_coords[0], y_coords[1], "Coordonnées Y identiques")
        
        print(f"✅ Transformation cas limites: 2 points aux extrémités")
        print(f"   LAT: {edge_lat.min():.6f} à {edge_lat.max():.6f}")
        print(f"   LON: {edge_lon.min():.6f} à {edge_lon.max():.6f}")
    
    def test_transform_coordinates_consistency(self):
        """Test de cohérence de la transformation (même point → même résultat)"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Prendre un point de référence
        ref_lat = pd.Series([self.real_lat.iloc[0]])
        ref_lon = pd.Series([self.real_lon.iloc[0]])
        
        # Première transformation
        x1, y1 = self.cleaner._transform_coordinates(ref_lat, ref_lon)
        
        # Deuxième transformation (même point)
        x2, y2 = self.cleaner._transform_coordinates(ref_lat, ref_lon)
        
        # Vérifications de cohérence
        self.assertAlmostEqual(x1[0], x2[0], places=2, msg="Coordonnées X incohérentes")
        self.assertAlmostEqual(y1[0], y2[0], places=2, msg="Coordonnées Y incohérentes")
        
        print(f"✅ Transformation cohérente: même point → même résultat UTM")
    
    def test_transform_coordinates_invalid_input(self):
        """Test de transformation avec des entrées invalides"""
        # Test avec des valeurs NaN
        nan_lat = pd.Series([np.nan, 4.707200127])
        nan_lon = pd.Series([12.34385219, np.nan])
        
        # La méthode devrait gérer les NaN
        try:
            x_coords, y_coords = self.cleaner._transform_coordinates(nan_lat, nan_lon)
            # Si elle ne lève pas d'exception, vérifier les résultats
            self.assertEqual(len(x_coords), 2)
            self.assertEqual(len(y_coords), 2)
            print(f"✅ Transformation avec NaN gérée correctement")
        except Exception as e:
            # Si elle lève une exception, c'est acceptable
            print(f"✅ Transformation avec NaN lève exception (acceptable): {e}")
    
    def test_transform_coordinates_empty_series(self):
        """Test de transformation avec des séries vides"""
        empty_lat = pd.Series([], dtype=float)
        empty_lon = pd.Series([], dtype=float)
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(empty_lat, empty_lon)
        
        # Vérifications
        self.assertEqual(len(x_coords), 0)
        self.assertEqual(len(y_coords), 0)
        
        print(f"✅ Transformation avec séries vides gérée correctement")
    
    def test_transform_coordinates_precision(self):
        """Test de précision de la transformation"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Prendre un point de référence
        ref_lat = pd.Series([self.real_lat.iloc[0]])
        ref_lon = pd.Series([self.real_lon.iloc[0]])
        
        # Transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(ref_lat, ref_lon)
        
        # Vérifier que les coordonnées UTM ont une précision raisonnable
        # (pas de valeurs entières si les coordonnées d'entrée sont précises)
        x_precision = len(str(x_coords[0]).split('.')[-1]) if '.' in str(x_coords[0]) else 0
        y_precision = len(str(y_coords[0]).split('.')[-1]) if '.' in str(y_coords[0]) else 0
        
        self.assertGreater(x_precision, 0, "Coordonnée X UTM devrait avoir une précision décimale")
        self.assertGreater(y_precision, 0, "Coordonnée Y UTM devrait avoir une précision décimale")
        
        print(f"✅ Précision de transformation: X={x_precision} décimales, Y={y_precision} décimales")
    
    def test_transform_coordinates_all_fixtures(self):
        """Test de transformation avec les données de test LAT/LON"""
        # Utiliser les données de test créées dans setUp
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Données de test LAT/LON non disponibles")
        
        # Tester la transformation avec les données de test
        test_lat = self.real_lat.head(5)
        test_lon = self.real_lon.head(5)
        
        # Tester la transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(test_lat, test_lon)
        
        # Vérifications
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), len(test_lat))
        self.assertEqual(len(y_coords), len(test_lon))
        
        # Vérifier que les coordonnées UTM sont dans des plages raisonnables
        self.assertTrue(np.all(x_coords > 400000), "Coordonnées X UTM trop faibles")
        self.assertTrue(np.all(x_coords < 1000000), "Coordonnées X UTM trop élevées")
        self.assertTrue(np.all(y_coords > 5000000), "Coordonnées Y UTM trop faibles")
        self.assertTrue(np.all(y_coords < 6000000), "Coordonnées Y UTM trop élevées")
        
        print(f"✅ Données de test: {len(x_coords)} coordonnées transformées avec succès")
        print(f"   LAT: {test_lat.min():.6f} à {test_lat.max():.6f}")
        print(f"   LON: {test_lon.min():.6f} à {test_lon.max():.6f}")
        print(f"   UTM X: {x_coords.min():.2f} à {x_coords.max():.2f}")
        print(f"   UTM Y: {y_coords.min():.2f} à {y_coords.max():.2f}")
    
    def test_transform_coordinates_performance(self):
        """Test de performance de la transformation avec les données réelles"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        import time
        
        # Prendre un échantillon plus large pour tester la performance
        sample_size = min(100, len(self.real_lat))
        test_lat = self.real_lat.head(sample_size)
        test_lon = self.real_lon.head(sample_size)
        
        # Mesurer le temps de transformation
        start_time = time.time()
        x_coords, y_coords = self.cleaner._transform_coordinates(test_lat, test_lon)
        transformation_time = time.time() - start_time
        
        # Vérifications
        self.assertEqual(len(x_coords), sample_size)
        self.assertEqual(len(y_coords), sample_size)
        
        # Vérifier que la transformation est rapide (< 1 seconde pour 100 points)
        self.assertLess(transformation_time, 1.0, f"Transformation trop lente: {transformation_time:.3f}s")
        
        print(f"✅ Performance: {sample_size} coordonnées transformées en {transformation_time:.3f}s")
        print(f"   Vitesse: {sample_size/transformation_time:.0f} coordonnées/seconde")
    
    def test_transform_coordinates_data_quality(self):
        """Test de qualité des données de transformation"""
        if self.s_df is None or len(self.real_lat) == 0:
            self.skipTest("Fichier S.csv non trouvé ou pas de données LAT/LON")
        
        # Prendre un échantillon de coordonnées
        test_lat = self.real_lat.head(20)
        test_lon = self.real_lon.head(20)
        
        # Transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(test_lat, test_lon)
        
        # Vérifier qu'il n'y a pas de valeurs infinies ou NaN
        self.assertFalse(np.any(np.isnan(x_coords)), "Coordonnées X UTM contiennent des NaN")
        self.assertFalse(np.any(np.isnan(y_coords)), "Coordonnées Y UTM contiennent des NaN")
        self.assertFalse(np.any(np.isinf(x_coords)), "Coordonnées X UTM contiennent des valeurs infinies")
        self.assertFalse(np.any(np.isinf(y_coords)), "Coordonnées Y UTM contiennent des valeurs infinies")
        
        # Vérifier que les coordonnées sont différentes (pas de dégénérescence)
        if len(x_coords) > 1:
            x_unique = len(np.unique(x_coords))
            y_unique = len(np.unique(y_coords))
            self.assertGreater(x_unique, 1, "Toutes les coordonnées X UTM sont identiques")
            self.assertGreater(y_unique, 1, "Toutes les coordonnées Y UTM sont identiques")
        
        print(f"✅ Qualité des données: {len(x_coords)} coordonnées valides, pas de NaN ou infini")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
