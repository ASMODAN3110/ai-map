#!/usr/bin/env python3
"""
Test unitaire pour la méthode _calculate_coverage_area de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _calculate_coverage_area
avec des données réelles extraites des fichiers PD.csv et S.csv.
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

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerCalculateCoverageAreaRealData(unittest.TestCase):
    """Tests pour la méthode _calculate_coverage_area avec données réelles"""
    
    def setUp(self):
        """Configuration avant chaque test avec données réelles"""
        # Utiliser les vrais fichiers de données du projet
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
        
        # Charger les vraies données pour les tests
        self.pd_file = self.raw_data_dir / "PD.csv"
        self.s_file = self.raw_data_dir / "S.csv"
        
        if self.pd_file.exists():
            self.pd_df = pd.read_csv(self.pd_file, sep=',')
        else:
            self.pd_df = None
            
        if self.s_file.exists():
            self.s_df = pd.read_csv(self.s_file, sep=',')
        else:
            self.s_df = None
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_calculate_coverage_area_pd_csv_real(self):
        """Test de calcul de couverture avec les vraies données PD.csv"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Vérifier que les colonnes de coordonnées existent
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in self.pd_df.columns]
        if missing_cols:
            self.skipTest(f"Colonnes manquantes dans PD.csv: {missing_cols}")
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(self.pd_df)
        
        # Vérifications
        self.assertIsInstance(coverage, dict)
        self.assertIn('x_min', coverage)
        self.assertIn('x_max', coverage)
        self.assertIn('y_min', coverage)
        self.assertIn('y_max', coverage)
        self.assertIn('width', coverage)
        self.assertIn('height', coverage)
        
        # Vérifier les valeurs
        self.assertLess(coverage['x_min'], coverage['x_max'], "x_min devrait être < x_max")
        self.assertLess(coverage['y_min'], coverage['y_max'], "y_min devrait être < y_max")
        self.assertGreater(coverage['width'], 0, "Largeur devrait être > 0")
        self.assertGreater(coverage['height'], 0, "Hauteur devrait être > 0")
        
        # Vérifier la cohérence
        expected_width = coverage['x_max'] - coverage['x_min']
        expected_height = coverage['y_max'] - coverage['y_min']
        self.assertAlmostEqual(coverage['width'], expected_width, places=2)
        self.assertAlmostEqual(coverage['height'], expected_height, places=2)
        
        # Vérifier que les valeurs correspondent aux données réelles
        real_x_min = self.pd_df['x'].min()
        real_x_max = self.pd_df['x'].max()
        real_y_min = self.pd_df['y'].min()
        real_y_max = self.pd_df['y'].max()
        
        self.assertAlmostEqual(coverage['x_min'], real_x_min, places=2)
        self.assertAlmostEqual(coverage['x_max'], real_x_max, places=2)
        self.assertAlmostEqual(coverage['y_min'], real_y_min, places=2)
        self.assertAlmostEqual(coverage['y_max'], real_y_max, places=2)
        
        print(f"✅ Couverture PD.csv calculée:")
        print(f"   X: {coverage['x_min']:.2f} à {coverage['x_max']:.2f} (largeur: {coverage['width']:.2f}m)")
        print(f"   Y: {coverage['y_min']:.2f} à {coverage['y_max']:.2f} (hauteur: {coverage['height']:.2f}m)")
        print(f"   Surface: {coverage['width'] * coverage['height']:.2f} m²")
    
    def test_calculate_coverage_area_s_csv_real(self):
        """Test de calcul de couverture avec les vraies données S.csv"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Vérifier que les colonnes de coordonnées existent
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in self.s_df.columns]
        if missing_cols:
            self.skipTest(f"Colonnes manquantes dans S.csv: {missing_cols}")
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(self.s_df)
        
        # Vérifications
        self.assertIsInstance(coverage, dict)
        self.assertIn('x_min', coverage)
        self.assertIn('x_max', coverage)
        self.assertIn('y_min', coverage)
        self.assertIn('y_max', coverage)
        self.assertIn('width', coverage)
        self.assertIn('height', coverage)
        
        # Vérifier les valeurs
        self.assertLess(coverage['x_min'], coverage['x_max'], "x_min devrait être < x_max")
        self.assertLess(coverage['y_min'], coverage['y_max'], "y_min devrait être < y_max")
        self.assertGreater(coverage['width'], 0, "Largeur devrait être > 0")
        self.assertGreater(coverage['height'], 0, "Hauteur devrait être > 0")
        
        print(f"✅ Couverture S.csv calculée:")
        print(f"   X: {coverage['x_min']:.2f} à {coverage['x_max']:.2f} (largeur: {coverage['width']:.2f}m)")
        print(f"   Y: {coverage['y_min']:.2f} à {coverage['y_max']:.2f} (hauteur: {coverage['height']:.2f}m)")
        print(f"   Surface: {coverage['width'] * coverage['height']:.2f} m²")
    
    def test_calculate_coverage_area_single_point(self):
        """Test de calcul de couverture avec un seul point"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Vérifier que les colonnes de coordonnées existent
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in self.pd_df.columns]
        if missing_cols:
            self.skipTest(f"Colonnes manquantes dans PD.csv: {missing_cols}")
        
        # Créer un DataFrame avec un seul point
        single_point_df = pd.DataFrame({
            'x': [self.pd_df['x'].iloc[0]],
            'y': [self.pd_df['y'].iloc[0]]
        })
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(single_point_df)
        
        # Vérifications
        self.assertIsInstance(coverage, dict)
        self.assertIn('x_min', coverage)
        self.assertIn('x_max', coverage)
        self.assertIn('y_min', coverage)
        self.assertIn('y_max', coverage)
        self.assertIn('width', coverage)
        self.assertIn('height', coverage)
        
        # Pour un seul point, min = max et width = height = 0
        self.assertEqual(coverage['x_min'], coverage['x_max'])
        self.assertEqual(coverage['y_min'], coverage['y_max'])
        self.assertEqual(coverage['width'], 0)
        self.assertEqual(coverage['height'], 0)
        
        print(f"✅ Couverture point unique: X={coverage['x_min']:.2f}, Y={coverage['y_min']:.2f}")
    
    def test_calculate_coverage_area_empty_dataframe(self):
        """Test de calcul de couverture avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['x', 'y'])
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(empty_df)
        
        # Vérifications
        self.assertIsInstance(coverage, dict)
        # Pour un DataFrame vide, la méthode retourne un dictionnaire avec des valeurs NaN
        self.assertEqual(len(coverage), 6, "Couverture devrait contenir 6 clés même pour un DataFrame vide")
        
        # Vérifier que les valeurs sont NaN pour un DataFrame vide
        self.assertTrue(pd.isna(coverage['x_min']), "x_min devrait être NaN pour un DataFrame vide")
        self.assertTrue(pd.isna(coverage['x_max']), "x_max devrait être NaN pour un DataFrame vide")
        self.assertTrue(pd.isna(coverage['y_min']), "y_min devrait être NaN pour un DataFrame vide")
        self.assertTrue(pd.isna(coverage['y_max']), "y_max devrait être NaN pour un DataFrame vide")
        self.assertTrue(pd.isna(coverage['width']), "width devrait être NaN pour un DataFrame vide")
        self.assertTrue(pd.isna(coverage['height']), "height devrait être NaN pour un DataFrame vide")
        
        print(f"✅ Couverture DataFrame vide gérée correctement")
    
    def test_calculate_coverage_area_missing_columns(self):
        """Test de calcul de couverture avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame sans colonnes de coordonnées
        df_no_coords = self.pd_df.drop(columns=['x', 'y'], errors='ignore')
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(df_no_coords)
        
        # Vérifications
        self.assertIsInstance(coverage, dict)
        self.assertEqual(len(coverage), 0, "Couverture devrait être vide sans colonnes de coordonnées")
        
        print(f"✅ Couverture sans colonnes de coordonnées gérée correctement")
    
    def test_calculate_coverage_area_with_z_coordinates(self):
        """Test de calcul de couverture avec coordonnées Z (3D)"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Vérifier que la colonne z existe
        if 'z' not in self.pd_df.columns:
            self.skipTest("Colonne z manquante dans PD.csv")
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(self.pd_df)
        
        # Vérifications (la méthode ne devrait traiter que x et y)
        self.assertIsInstance(coverage, dict)
        self.assertIn('x_min', coverage)
        self.assertIn('x_max', coverage)
        self.assertIn('y_min', coverage)
        self.assertIn('y_max', coverage)
        
        # Vérifier que z n'est pas inclus dans la couverture 2D
        self.assertNotIn('z_min', coverage)
        self.assertNotIn('z_max', coverage)
        
        print(f"✅ Couverture 2D calculée (z ignoré): {coverage['width']:.2f}m x {coverage['height']:.2f}m")
    
    def test_calculate_coverage_area_precision(self):
        """Test de précision du calcul de couverture"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode de calcul de couverture
        coverage = self.cleaner._calculate_coverage_area(self.pd_df)
        
        # Vérifier que les calculs sont précis
        calculated_width = coverage['x_max'] - coverage['x_min']
        calculated_height = coverage['y_max'] - coverage['y_min']
        
        self.assertAlmostEqual(coverage['width'], calculated_width, places=10)
        self.assertAlmostEqual(coverage['height'], calculated_height, places=10)
        
        # Vérifier que les valeurs sont cohérentes avec les données
        real_x_min = self.pd_df['x'].min()
        real_x_max = self.pd_df['x'].max()
        real_y_min = self.pd_df['y'].min()
        real_y_max = self.pd_df['y'].max()
        
        self.assertAlmostEqual(coverage['x_min'], real_x_min, places=10)
        self.assertAlmostEqual(coverage['x_max'], real_x_max, places=10)
        self.assertAlmostEqual(coverage['y_min'], real_y_min, places=10)
        self.assertAlmostEqual(coverage['y_max'], real_y_max, places=10)
        
        print(f"✅ Précision du calcul de couverture vérifiée (10 décimales)")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
