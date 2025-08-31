#!/usr/bin/env python3
"""
Test unitaire pour la méthode _calculate_coverage_area de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _calculate_coverage_area
qui calcule la zone de couverture spatiale des données
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


class TestDataCleanerCalculateCoverageArea(unittest.TestCase):
    """Tests pour la méthode _calculate_coverage_area de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
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
    
    def test_calculate_coverage_area_pd_csv(self):
        """Test de calcul de la zone de couverture avec PD.csv"""
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
        
        # Appeler la méthode de calcul de la zone de couverture
        coverage_area = self.cleaner._calculate_coverage_area(df_pd)
        
        # Vérifications
        self.assertIsInstance(coverage_area, dict)
        self.assertGreater(len(coverage_area), 0, "La zone de couverture ne devrait pas être vide")
        
        # Vérifier que toutes les clés sont présentes
        expected_keys = ['x_min', 'x_max', 'y_min', 'y_max', 'width', 'height']
        for key in expected_keys:
            self.assertIn(key, coverage_area, f"Clé {key} manquante dans la zone de couverture")
        
        # Vérifier que les valeurs sont numériques
        for key in expected_keys:
            self.assertTrue(np.issubdtype(type(coverage_area[key]), np.number), f"Valeur {key} devrait être numérique")
        
        # Vérifier que les calculs sont cohérents
        calculated_width = coverage_area['x_max'] - coverage_area['x_min']
        calculated_height = coverage_area['y_max'] - coverage_area['y_min']
        
        self.assertAlmostEqual(coverage_area['width'], calculated_width, places=6, 
                              msg="La largeur calculée devrait correspondre à x_max - x_min")
        self.assertAlmostEqual(coverage_area['height'], calculated_height, places=6, 
                              msg="La hauteur calculée devrait correspondre à y_max - y_min")
        
        print(f"✅ Zone de couverture de PD.csv calculée: {coverage_area['width']:.1f}m x {coverage_area['height']:.1f}m")
    
    def test_calculate_coverage_area_s_csv(self):
        """Test de calcul de la zone de couverture avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        coord_cols = ['LAT', 'LON']
        for col in coord_cols:
            self.assertIn(col, df_s.columns, f"Colonne {col} manquante dans S.csv")
        
        # S.csv n'a pas de colonnes x, y, donc la méthode devrait retourner un dictionnaire vide
        coverage_area = self.cleaner._calculate_coverage_area(df_s)
        
        # Vérifications
        self.assertIsInstance(coverage_area, dict)
        self.assertEqual(len(coverage_area), 0, "S.csv n'a pas de colonnes x, y, donc la zone de couverture devrait être vide")
        
        print("✅ Zone de couverture de S.csv validée (pas de colonnes x, y)")
    
    def test_calculate_coverage_area_both_files(self):
        """Test de calcul de la zone de couverture des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        coverage_pd = self.cleaner._calculate_coverage_area(df_pd)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        coverage_s = self.cleaner._calculate_coverage_area(df_s)
        
        # Vérifier que les deux calculs ont réussi
        self.assertIsInstance(coverage_pd, dict)
        self.assertIsInstance(coverage_s, dict)
        
        # PD.csv devrait avoir une zone de couverture
        self.assertGreater(len(coverage_pd), 0, "PD.csv devrait avoir une zone de couverture calculée")
        
        # S.csv ne devrait pas avoir de zone de couverture (pas de colonnes x, y)
        self.assertEqual(len(coverage_s), 0, "S.csv ne devrait pas avoir de zone de couverture (pas de colonnes x, y)")
        
        print(f"✅ Zones de couverture calculées: PD.csv ({len(coverage_pd)} clés), S.csv ({len(coverage_s)} clés)")
    
    def test_calculate_coverage_area_coordinate_validation(self):
        """Test de validation des coordonnées et calcul de la zone de couverture"""
        # Créer un DataFrame avec des coordonnées connues
        test_df = pd.DataFrame({
            'x': [0, 10, 20, 30, 40, 50],
            'y': [0, 5, 10, 15, 20, 25],
            'z': [0, 1, 2, 3, 4, 5],
            'value': ['A', 'B', 'C', 'D', 'E', 'F']
        })
        
        # Appeler la méthode de calcul de la zone de couverture
        coverage_area = self.cleaner._calculate_coverage_area(test_df)
        
        # Vérifier que la zone de couverture a été calculée
        self.assertIsInstance(coverage_area, dict)
        self.assertGreater(len(coverage_area), 0)
        
        # Vérifier les valeurs attendues
        self.assertEqual(coverage_area['x_min'], 0)
        self.assertEqual(coverage_area['x_max'], 50)
        self.assertEqual(coverage_area['y_min'], 0)
        self.assertEqual(coverage_area['y_max'], 25)
        self.assertEqual(coverage_area['width'], 50)
        self.assertEqual(coverage_area['height'], 25)
        
        print(f"✅ Validation des coordonnées et calcul de la zone de couverture: {coverage_area['width']} x {coverage_area['height']}")
    
    def test_calculate_coverage_area_no_coordinate_columns(self):
        """Test avec un DataFrame sans colonnes de coordonnées x, y"""
        # Créer un DataFrame sans colonnes de coordonnées x, y
        no_coord_df = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [6, 7, 8, 9, 10],
            'value3': [11, 12, 13, 14, 15]
        })
        
        # Appeler la méthode de calcul de la zone de couverture
        coverage_area = self.cleaner._calculate_coverage_area(no_coord_df)
        
        # Vérifier que la zone de couverture est vide
        self.assertIsInstance(coverage_area, dict)
        self.assertEqual(len(coverage_area), 0, "La zone de couverture devrait être vide sans colonnes x, y")
        
        print("✅ DataFrame sans colonnes de coordonnées x, y traité correctement")
    
    def test_calculate_coverage_area_data_integrity(self):
        """Test de l'intégrité des données après calcul de la zone de couverture"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        
        # Sauvegarder les valeurs originales
        original_x_min = df_pd_original['x'].min()
        original_x_max = df_pd_original['x'].max()
        original_y_min = df_pd_original['y'].min()
        original_y_max = df_pd_original['y'].max()
        
        # Appeler la méthode de calcul de la zone de couverture
        coverage_area = self.cleaner._calculate_coverage_area(df_pd_original)
        
        # Vérifier que les données originales n'ont pas été modifiées
        self.assertEqual(df_pd_original['x'].min(), original_x_min, "x_min ne devrait pas être modifié")
        self.assertEqual(df_pd_original['x'].max(), original_x_max, "x_max ne devrait pas être modifié")
        self.assertEqual(df_pd_original['y'].min(), original_y_min, "y_min ne devrait pas être modifié")
        self.assertEqual(df_pd_original['y'].max(), original_y_max, "y_max ne devrait pas être modifié")
        
        # Vérifier que la zone de couverture correspond aux données originales
        self.assertEqual(coverage_area['x_min'], original_x_min)
        self.assertEqual(coverage_area['x_max'], original_x_max)
        self.assertEqual(coverage_area['y_min'], original_y_min)
        self.assertEqual(coverage_area['y_max'], original_y_max)
        
        print("✅ Intégrité des données préservée après calcul de la zone de couverture")
    
    def test_calculate_coverage_area_coordinate_ranges(self):
        """Test des plages de coordonnées et calcul de la zone de couverture"""
        # Tester PD.csv avec coordonnées cartésiennes
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier que les coordonnées sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['x']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['y']))
        
        # Calculer manuellement les plages de coordonnées
        manual_x_min = df_pd['x'].min()
        manual_x_max = df_pd['x'].max()
        manual_y_min = df_pd['y'].min()
        manual_y_max = df_pd['y'].max()
        manual_width = manual_x_max - manual_x_min
        manual_height = manual_y_max - manual_y_min
        
        # Appeler la méthode de calcul de la zone de couverture
        coverage_area = self.cleaner._calculate_coverage_area(df_pd)
        
        # Vérifier que les calculs correspondent
        self.assertEqual(coverage_area['x_min'], manual_x_min)
        self.assertEqual(coverage_area['x_max'], manual_x_max)
        self.assertEqual(coverage_area['y_min'], manual_y_min)
        self.assertEqual(coverage_area['y_max'], manual_y_max)
        self.assertEqual(coverage_area['width'], manual_width)
        self.assertEqual(coverage_area['height'], manual_height)
        
        print(f"✅ Plages de coordonnées validées: x={manual_width:.1f}m, y={manual_height:.1f}m")
    
    def test_calculate_coverage_area_performance(self):
        """Test de performance du calcul de la zone de couverture"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        coverage_pd = self.cleaner._calculate_coverage_area(df_pd)
        pd_time = time.time() - start_time
        
        self.assertIsInstance(coverage_pd, dict)
        self.assertLess(pd_time, 1.0, "Calcul de la zone de couverture de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        coverage_s = self.cleaner._calculate_coverage_area(df_s)
        s_time = time.time() - start_time
        
        self.assertIsInstance(coverage_s, dict)
        self.assertLess(s_time, 1.0, "Calcul de la zone de couverture de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_calculate_coverage_area_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        coverage_pd = self.cleaner._calculate_coverage_area(validated_pd)
        
        self.assertIsInstance(coverage_pd, dict)
        self.assertGreater(len(coverage_pd), 0)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        coverage_s = self.cleaner._calculate_coverage_area(validated_s)
        
        self.assertIsInstance(coverage_s, dict)
        self.assertEqual(len(coverage_s), 0)
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_calculate_coverage_area_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        coverage_empty = self.cleaner._calculate_coverage_area(empty_df)
        self.assertIsInstance(coverage_empty, dict)
        self.assertEqual(len(coverage_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'x': [100.0],
            'y': [200.0]
        })
        coverage_single = self.cleaner._calculate_coverage_area(single_row_df)
        self.assertIsInstance(coverage_single, dict)
        self.assertGreater(len(coverage_single), 0)
        
        # Vérifier que la largeur et hauteur sont 0 pour un seul point
        self.assertEqual(coverage_single['width'], 0)
        self.assertEqual(coverage_single['height'], 0)
        
        # Test avec DataFrame avec des coordonnées identiques
        identical_coords_df = pd.DataFrame({
            'x': [100.0, 100.0, 100.0],
            'y': [200.0, 200.0, 200.0]
        })
        coverage_identical = self.cleaner._calculate_coverage_area(identical_coords_df)
        self.assertIsInstance(coverage_identical, dict)
        self.assertGreater(len(coverage_identical), 0)
        
        # Vérifier que la largeur et hauteur sont 0 pour des coordonnées identiques
        self.assertEqual(coverage_identical['width'], 0)
        self.assertEqual(coverage_identical['height'], 0)
        
        print("✅ Cas limites gérés avec succès")
    
    def test_calculate_coverage_area_coordinate_systems(self):
        """Test de calcul avec différents systèmes de coordonnées"""
        # Tester PD.csv (coordonnées cartésiennes UTM)
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # PD.csv utilise des coordonnées cartésiennes (x, y, z)
        self.assertIn('x', df_pd.columns)
        self.assertIn('y', df_pd.columns)
        
        coverage_pd = self.cleaner._calculate_coverage_area(df_pd)
        self.assertIsInstance(coverage_pd, dict)
        self.assertGreater(len(coverage_pd), 0)
        
        # Tester S.csv (coordonnées géographiques LAT/LON)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # S.csv utilise des coordonnées géographiques (LAT, LON)
        self.assertIn('LAT', df_s.columns)
        self.assertIn('LON', df_s.columns)
        
        # S.csv n'a pas de colonnes x, y, donc pas de zone de couverture calculée
        coverage_s = self.cleaner._calculate_coverage_area(df_s)
        self.assertIsInstance(coverage_s, dict)
        self.assertEqual(len(coverage_s), 0)
        
        print("✅ Calcul avec différents systèmes de coordonnées réussi")
    
    def test_calculate_coverage_area_return_structure(self):
        """Test de la structure de retour de la zone de couverture"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        coverage_area = self.cleaner._calculate_coverage_area(df_pd)
        
        # Vérifier la structure du dictionnaire retourné
        self.assertIsInstance(coverage_area, dict)
        self.assertGreater(len(coverage_area), 0)
        
        # Vérifier que toutes les clés requises sont présentes
        required_keys = ['x_min', 'x_max', 'y_min', 'y_max', 'width', 'height']
        for key in required_keys:
            self.assertIn(key, coverage_area, f"Clé {key} manquante dans la zone de couverture")
        
        # Vérifier que les valeurs sont cohérentes
        self.assertLessEqual(coverage_area['x_min'], coverage_area['x_max'], "x_min devrait être <= x_max")
        self.assertLessEqual(coverage_area['y_min'], coverage_area['y_max'], "y_min devrait être <= y_max")
        self.assertGreaterEqual(coverage_area['width'], 0, "La largeur devrait être >= 0")
        self.assertGreaterEqual(coverage_area['height'], 0, "La hauteur devrait être >= 0")
        
        print("✅ Structure de retour de la zone de couverture validée")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
