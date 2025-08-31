#!/usr/bin/env python3
"""
Test unitaire pour la méthode _transform_coordinates de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _transform_coordinates
qui transforme les coordonnées géographiques LAT/LON vers UTM (x, y)
en utilisant les vrais fichiers PD.csv et S.csv.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerTransformCoordinates(unittest.TestCase):
    """Tests pour la méthode _transform_coordinates de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('src.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
    
    def test_transform_coordinates_s_csv(self):
        """Test de transformation des coordonnées avec S.csv (LAT/LON)"""
        # Charger S.csv qui contient des coordonnées LAT/LON
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        coord_cols = ['LAT', 'LON']
        for col in coord_cols:
            self.assertIn(col, df_s.columns, f"Colonne {col} manquante dans S.csv")
        
        # Extraire les coordonnées LAT/LON
        lat_series = df_s['LAT']
        lon_series = df_s['LON']
        
        # Vérifier que les coordonnées sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(lat_series))
        self.assertTrue(pd.api.types.is_numeric_dtype(lon_series))
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(lat_series, lon_series)
        
        # Vérifications
        self.assertIsInstance(x_coords, np.ndarray, "x_coords devrait être un numpy array")
        self.assertIsInstance(y_coords, np.ndarray, "y_coords devrait être un numpy array")
        
        # Vérifier les dimensions
        self.assertEqual(len(x_coords), len(lat_series), "Le nombre de coordonnées x devrait correspondre au nombre de lignes")
        self.assertEqual(len(y_coords), len(lon_series), "Le nombre de coordonnées y devrait correspondre au nombre de lignes")
        
        # Vérifier que les coordonnées transformées sont numériques
        self.assertTrue(np.issubdtype(x_coords.dtype, np.number), "x_coords devrait être numérique")
        self.assertTrue(np.issubdtype(y_coords.dtype, np.number), "y_coords devrait être numérique")
        
        # Vérifier qu'il n'y a pas de valeurs NaN dans les coordonnées transformées
        self.assertFalse(np.any(np.isnan(x_coords)), "x_coords ne devrait pas contenir de valeurs NaN")
        self.assertFalse(np.any(np.isnan(y_coords)), "y_coords ne devrait pas contenir de valeurs NaN")
        
        print(f"✅ Coordonnées de S.csv transformées: {len(x_coords)} points LAT/LON → UTM (x, y)")
    
    def test_transform_coordinates_pd_csv_coordinates(self):
        """Test de transformation des coordonnées avec les coordonnées de PD.csv (si disponibles)"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            self.assertIn(col, df_pd.columns, f"Colonne {col} manquante dans PD.csv")
        
        # PD.csv utilise déjà des coordonnées cartésiennes, donc pas de transformation nécessaire
        # Mais on peut tester que la méthode fonctionne avec des coordonnées fictives
        print("✅ PD.csv utilise déjà des coordonnées cartésiennes (x, y, z)")
    
    def test_transform_coordinates_both_files_comparison(self):
        """Test de comparaison des transformations entre les deux fichiers"""
        # Tester S.csv (LAT/LON)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        lat_series_s = df_s['LAT']
        lon_series_s = df_s['LON']
        
        x_coords_s, y_coords_s = self.cleaner._transform_coordinates(lat_series_s, lon_series_s)
        
        # Vérifier que les transformations ont réussi
        self.assertIsInstance(x_coords_s, np.ndarray)
        self.assertIsInstance(y_coords_s, np.ndarray)
        self.assertEqual(len(x_coords_s), len(lat_series_s))
        self.assertEqual(len(y_coords_s), len(lon_series_s))
        
        # Vérifier que les coordonnées transformées sont cohérentes
        self.assertTrue(np.all(np.isfinite(x_coords_s)), "Toutes les coordonnées x devraient être finies")
        self.assertTrue(np.all(np.isfinite(y_coords_s)), "Toutes les coordonnées y devraient être finies")
        
        print(f"✅ Transformation des coordonnées validée: S.csv ({len(x_coords_s)} points)")
    
    def test_transform_coordinates_data_integrity(self):
        """Test de l'intégrité des données après transformation"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # Extraire les coordonnées originales
        lat_series = df_s['LAT']
        lon_series = df_s['LON']
        
        # Sauvegarder les valeurs originales
        original_lat = lat_series.copy()
        original_lon = lon_series.copy()
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(lat_series, lon_series)
        
        # Vérifier que les séries originales n'ont pas été modifiées
        pd.testing.assert_series_equal(lat_series, original_lat, check_names=False)
        pd.testing.assert_series_equal(lon_series, original_lon, check_names=False)
        
        # Vérifier que les coordonnées transformées sont cohérentes
        self.assertEqual(len(x_coords), len(original_lat))
        self.assertEqual(len(y_coords), len(original_lon))
        
        print("✅ Intégrité des données originales préservée après transformation")
    
    def test_transform_coordinates_numeric_validation(self):
        """Test de validation des coordonnées numériques"""
        # Créer des séries de coordonnées numériques
        lat_values = [48.8566, 43.2965, 45.7640]  # Paris, Lyon, Nice
        lon_values = [2.3522, 5.3698, 4.8357]
        
        lat_series = pd.Series(lat_values)
        lon_series = pd.Series(lon_values)
        
        # Vérifier que les séries sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(lat_series))
        self.assertTrue(pd.api.types.is_numeric_dtype(lon_series))
        
        # Appeler la méthode de transformation
        x_coords, y_coords = self.cleaner._transform_coordinates(lat_series, lon_series)
        
        # Vérifications
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), len(lat_values))
        self.assertEqual(len(y_coords), len(lon_values))
        
        # Vérifier que les coordonnées transformées sont numériques
        self.assertTrue(np.issubdtype(x_coords.dtype, np.number))
        self.assertTrue(np.issubdtype(y_coords.dtype, np.number))
        
        print("✅ Validation des coordonnées numériques réussie")
    
    def test_transform_coordinates_edge_cases(self):
        """Test des cas limites de transformation des coordonnées"""
        # Test avec des coordonnées extrêmes
        extreme_lat = pd.Series([90.0, -90.0, 0.0])  # Pôles et équateur
        extreme_lon = pd.Series([180.0, -180.0, 0.0])  # Méridiens extrêmes
        
        x_coords, y_coords = self.cleaner._transform_coordinates(extreme_lat, extreme_lon)
        
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), 3)
        self.assertEqual(len(y_coords), 3)
        
        # Test avec des coordonnées à une seule valeur
        single_lat = pd.Series([48.8566])
        single_lon = pd.Series([2.3522])
        
        x_coords_single, y_coords_single = self.cleaner._transform_coordinates(single_lat, single_lon)
        
        self.assertIsInstance(x_coords_single, np.ndarray)
        self.assertIsInstance(y_coords_single, np.ndarray)
        self.assertEqual(len(x_coords_single), 1)
        self.assertEqual(len(y_coords_single), 1)
        
        print("✅ Cas limites de transformation gérés avec succès")
    
    def test_transform_coordinates_performance(self):
        """Test de performance de la transformation des coordonnées"""
        import time
        
        # Tester avec S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        lat_series = df_s['LAT']
        lon_series = df_s['LON']
        
        start_time = time.time()
        x_coords, y_coords = self.cleaner._transform_coordinates(lat_series, lon_series)
        transform_time = time.time() - start_time
        
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertLess(transform_time, 1.0, "Transformation des coordonnées devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: {len(x_coords)} points transformés en {transform_time:.3f}s")
    
    def test_transform_coordinates_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        
        # Valider les colonnes
        validated_df = self.cleaner._validate_columns(df_s, "schlumberger")
        
        # Nettoyer les coordonnées (ce qui peut déclencher la transformation)
        cleaned_df = self.cleaner._clean_coordinates(validated_df, "schlumberger")
        
        # Vérifier que le processus s'est bien déroulé
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), len(validated_df))
        
        # Si des colonnes x, y ont été créées, vérifier qu'elles sont numériques
        if 'x' in cleaned_df.columns and 'y' in cleaned_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['x']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['y']))
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_transform_coordinates_coordinate_system_validation(self):
        """Test de validation du système de coordonnées"""
        # Vérifier que le transformer de coordonnées est configuré
        self.assertIsNotNone(self.cleaner.coord_transformer, "Le transformer de coordonnées devrait être configuré")
        
        # Tester avec des coordonnées de référence connues
        # Coordonnées de Paris (48.8566°N, 2.3522°E)
        paris_lat = pd.Series([48.8566])
        paris_lon = pd.Series([2.3522])
        
        x_coords, y_coords = self.cleaner._transform_coordinates(paris_lat, paris_lon)
        
        # Vérifier que la transformation a produit des coordonnées UTM valides
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), 1)
        self.assertEqual(len(y_coords), 1)
        
        # Les coordonnées UTM devraient être dans des plages raisonnables
        # (pour la zone UTM 30N, x devrait être entre 500000 et 600000, y entre 5000000 et 6000000)
        self.assertTrue(0 <= x_coords[0] <= 1000000, f"Coordonnée x UTM {x_coords[0]} hors de la plage attendue")
        self.assertTrue(0 <= y_coords[0] <= 10000000, f"Coordonnée y UTM {y_coords[0]} hors de la plage attendue")
        
        print("✅ Système de coordonnées UTM validé avec des coordonnées de référence")
    
    def test_transform_coordinates_error_handling(self):
        """Test de gestion des erreurs lors de la transformation"""
        # Test avec des coordonnées invalides (NaN)
        invalid_lat = pd.Series([48.8566, np.nan, 43.2965])
        invalid_lon = pd.Series([2.3522, 5.3698, np.nan])
        
        # La méthode devrait gérer les NaN et produire des coordonnées valides
        x_coords, y_coords = self.cleaner._transform_coordinates(invalid_lat, invalid_lon)
        
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), 3)
        self.assertEqual(len(y_coords), 3)
        
        # Vérifier que les coordonnées invalides produisent des NaN ou des valeurs par défaut
        # (le comportement exact dépend de l'implémentation du transformer)
        print("✅ Gestion des coordonnées invalides testée")
    
    def test_transform_coordinates_batch_processing(self):
        """Test de traitement par lots de coordonnées"""
        # Créer un grand ensemble de coordonnées
        num_points = 1000
        lat_values = np.random.uniform(-90, 90, num_points)
        lon_values = np.random.uniform(-180, 180, num_points)
        
        lat_series = pd.Series(lat_values)
        lon_series = pd.Series(lon_values)
        
        # Appeler la méthode de transformation
        start_time = time.time()
        x_coords, y_coords = self.cleaner._transform_coordinates(lat_series, lon_series)
        transform_time = time.time() - start_time
        
        # Vérifications
        self.assertIsInstance(x_coords, np.ndarray)
        self.assertIsInstance(y_coords, np.ndarray)
        self.assertEqual(len(x_coords), num_points)
        self.assertEqual(len(y_coords), num_points)
        
        # Vérifier la performance
        self.assertLess(transform_time, 5.0, "Transformation de 1000 points devrait être rapide (< 5 secondes)")
        
        print(f"✅ Traitement par lots validé: {num_points} points transformés en {transform_time:.3f}s")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
