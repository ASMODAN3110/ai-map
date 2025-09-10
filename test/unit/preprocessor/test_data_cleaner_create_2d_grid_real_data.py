#!/usr/bin/env python3
"""
Test unitaire pour la méthode _create_2d_grid de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _create_2d_grid
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


class TestDataCleanerCreate2dGridRealData(unittest.TestCase):
    """Tests pour la méthode _create_2d_grid avec données réelles"""
    
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
            self.pd_df = pd.read_csv(self.pd_file)
        else:
            self.pd_df = None
            
        if self.s_file.exists():
            self.s_df = pd.read_csv(self.s_file)
        else:
            self.s_df = None
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_create_2d_grid_pd_csv_real(self):
        """Test de création de grille 2D avec les vraies données PD.csv"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Paramètres de la grille
        height, width, channels = 64, 64, 4
        
        # Appeler la méthode privée via reflection
        grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(grid_2d, np.ndarray)
        self.assertEqual(grid_2d.shape, (height, width, channels))
        self.assertEqual(grid_2d.dtype, np.float64)
        
        # Vérifier que la grille n'est pas vide
        self.assertGreater(np.abs(grid_2d).sum(), 0, "La grille ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(np.isnan(grid_2d).any(), "La grille ne devrait pas contenir de NaN")
        self.assertFalse(np.isinf(grid_2d).any(), "La grille ne devrait pas contenir d'infini")
        
        print(f"✅ Grille 2D créée avec PD.csv:")
        print(f"   Shape: {grid_2d.shape}")
        print(f"   Min: {grid_2d.min():.4f}")
        print(f"   Max: {grid_2d.max():.4f}")
        print(f"   Mean: {grid_2d.mean():.4f}")
    
    def test_create_2d_grid_s_csv_real(self):
        """Test de création de grille 2D avec les vraies données S.csv"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Paramètres de la grille
        height, width, channels = 64, 64, 4
        
        # Appeler la méthode privée via reflection
        grid_2d = self.cleaner._create_2d_grid(self.s_df, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(grid_2d, np.ndarray)
        self.assertEqual(grid_2d.shape, (height, width, channels))
        self.assertEqual(grid_2d.dtype, np.float64)
        
        # Vérifier que la grille n'est pas vide
        self.assertGreater(np.abs(grid_2d).sum(), 0, "La grille ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(np.isnan(grid_2d).any(), "La grille ne devrait pas contenir de NaN")
        self.assertFalse(np.isinf(grid_2d).any(), "La grille ne devrait pas contenir d'infini")
        
        print(f"✅ Grille 2D créée avec S.csv:")
        print(f"   Shape: {grid_2d.shape}")
        print(f"   Min: {grid_2d.min():.4f}")
        print(f"   Max: {grid_2d.max():.4f}")
        print(f"   Mean: {grid_2d.mean():.4f}")
    
    def test_create_2d_grid_different_sizes(self):
        """Test de création de grille 2D avec différentes tailles"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Tester différentes tailles de grille
        test_sizes = [
            (32, 32, 4),   # Petite grille
            (64, 64, 4),   # Grille standard
            (128, 128, 4), # Grande grille
            (16, 32, 4),   # Grille rectangulaire
        ]
        
        for height, width, channels in test_sizes:
            with self.subTest(size=(height, width, channels)):
                grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
                
                # Vérifications
                self.assertIsInstance(grid_2d, np.ndarray)
                self.assertEqual(grid_2d.shape, (height, width, channels))
                self.assertGreater(np.abs(grid_2d).sum(), 0)
                
                print(f"✅ Grille {height}x{width}x{channels} créée correctement")
    
    def test_create_2d_grid_empty_dataframe(self):
        """Test de création de grille 2D avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        height, width, channels = 64, 64, 4
        
        # La méthode devrait gérer un DataFrame vide
        grid_2d = self.cleaner._create_2d_grid(empty_df, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(grid_2d, np.ndarray)
        self.assertEqual(grid_2d.shape, (height, width, channels))
        
        # La grille devrait être remplie de zéros pour un DataFrame vide
        self.assertEqual(np.abs(grid_2d).sum(), 0, "La grille devrait être zéro pour un DataFrame vide")
        
        print(f"✅ Grille 2D avec DataFrame vide gérée correctement")
    
    def test_create_2d_grid_missing_columns(self):
        """Test de création de grille 2D avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame avec des colonnes manquantes
        df_missing = self.pd_df.drop(columns=['resistivity', 'chargeability'], errors='ignore')
        height, width, channels = 64, 64, 4
        
        # La méthode devrait gérer les colonnes manquantes
        grid_2d = self.cleaner._create_2d_grid(df_missing, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(grid_2d, np.ndarray)
        self.assertEqual(grid_2d.shape, (height, width, channels))
        
        print(f"✅ Grille 2D avec colonnes manquantes gérée correctement")
    
    def test_create_2d_grid_channel_mapping(self):
        """Test du mapping des canaux dans la grille 2D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        height, width, channels = 64, 64, 4
        grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifier le mapping des canaux
        # Canal 0: x, Canal 1: y, Canal 2: resistivity, Canal 3: chargeability
        x_channel = grid_2d[:, :, 0]
        y_channel = grid_2d[:, :, 1]
        resistivity_channel = grid_2d[:, :, 2]
        chargeability_channel = grid_2d[:, :, 3]
        
        # Vérifier que les canaux contiennent des données
        self.assertGreater(np.abs(x_channel).sum(), 0, "Canal x ne devrait pas être vide")
        self.assertGreater(np.abs(y_channel).sum(), 0, "Canal y ne devrait pas être vide")
        self.assertGreater(np.abs(resistivity_channel).sum(), 0, "Canal résistivité ne devrait pas être vide")
        self.assertGreater(np.abs(chargeability_channel).sum(), 0, "Canal chargeabilité ne devrait pas être vide")
        
        # Vérifier que les coordonnées correspondent aux données réelles
        real_x_min, real_x_max = self.pd_df['x'].min(), self.pd_df['x'].max()
        real_y_min, real_y_max = self.pd_df['y'].min(), self.pd_df['y'].max()
        
        grid_x_min, grid_x_max = x_channel.min(), x_channel.max()
        grid_y_min, grid_y_max = y_channel.min(), y_channel.max()
        
        # Vérifier que les valeurs de la grille sont dans la plage des données réelles (avec tolérance)
        self.assertGreaterEqual(grid_x_min, real_x_min - 1.0, msg="X min devrait être proche des données réelles")
        self.assertLessEqual(grid_x_max, real_x_max + 1.0, msg="X max devrait être proche des données réelles")
        self.assertGreaterEqual(grid_y_min, real_y_min - 1.0, msg="Y min devrait être proche des données réelles")
        self.assertLessEqual(grid_y_max, real_y_max + 1.0, msg="Y max devrait être proche des données réelles")
        
        print(f"✅ Mapping des canaux vérifié:")
        print(f"   Canal X: {grid_x_min:.2f} à {grid_x_max:.2f}")
        print(f"   Canal Y: {grid_y_min:.2f} à {grid_y_max:.2f}")
        print(f"   Canal Résistivité: {resistivity_channel.min():.4f} à {resistivity_channel.max():.4f}")
        print(f"   Canal Chargeabilité: {chargeability_channel.min():.4f} à {chargeability_channel.max():.4f}")
    
    def test_create_2d_grid_interpolation_quality(self):
        """Test de la qualité de l'interpolation sur la grille 2D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        height, width, channels = 64, 64, 4
        grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifier que l'interpolation a bien réparti les données
        # La grille ne devrait pas avoir trop de zéros
        zero_ratio = np.sum(grid_2d == 0) / grid_2d.size
        self.assertLess(zero_ratio, 0.5, f"Trop de zéros dans la grille: {zero_ratio:.2%}")
        
        # Vérifier que les valeurs sont réparties de manière cohérente
        for channel in range(channels):
            channel_data = grid_2d[:, :, channel]
            
            # Vérifier que les valeurs ne sont pas toutes identiques
            unique_values = np.unique(channel_data)
            self.assertGreater(len(unique_values), 1, f"Canal {channel} ne devrait pas avoir qu'une seule valeur unique")
            
            # Vérifier la variance (doit être > 0)
            variance = np.var(channel_data)
            self.assertGreater(variance, 0, f"Canal {channel} devrait avoir une variance > 0")
        
        print(f"✅ Qualité de l'interpolation vérifiée:")
        print(f"   Ratio de zéros: {zero_ratio:.2%}")
        print(f"   Variance par canal: {[np.var(grid_2d[:, :, i]) for i in range(channels)]}")
    
    def test_create_2d_grid_spatial_coverage(self):
        """Test de la couverture spatiale de la grille 2D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        height, width, channels = 64, 64, 4
        grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifier que la grille couvre bien l'espace des données
        x_channel = grid_2d[:, :, 0]
        y_channel = grid_2d[:, :, 1]
        
        # Calculer les limites de la grille
        grid_x_min, grid_x_max = x_channel.min(), x_channel.max()
        grid_y_min, grid_y_max = y_channel.min(), y_channel.max()
        
        # Calculer les limites des données réelles
        real_x_min, real_x_max = self.pd_df['x'].min(), self.pd_df['x'].max()
        real_y_min, real_y_max = self.pd_df['y'].min(), self.pd_df['y'].max()
        
        # Vérifier que la grille couvre au moins l'espace des données
        self.assertLessEqual(grid_x_min, real_x_min, "La grille devrait couvrir au moins l'espace des données (X min)")
        self.assertGreaterEqual(grid_x_max, real_x_max, "La grille devrait couvrir au moins l'espace des données (X max)")
        self.assertLessEqual(grid_y_min, real_y_min, "La grille devrait couvrir au moins l'espace des données (Y min)")
        self.assertGreaterEqual(grid_y_max, real_y_max, "La grille devrait couvrir au moins l'espace des données (Y max)")
        
        # Calculer la résolution spatiale
        x_resolution = (grid_x_max - grid_x_min) / (width - 1)
        y_resolution = (grid_y_max - grid_y_min) / (height - 1)
        
        print(f"✅ Couverture spatiale vérifiée:")
        print(f"   Grille X: {grid_x_min:.2f} à {grid_x_max:.2f} (résolution: {x_resolution:.2f}m)")
        print(f"   Grille Y: {grid_y_min:.2f} à {grid_y_max:.2f} (résolution: {y_resolution:.2f}m)")
        print(f"   Données X: {real_x_min:.2f} à {real_x_max:.2f}")
        print(f"   Données Y: {real_y_min:.2f} à {real_y_max:.2f}")
    
    def test_create_2d_grid_consistency(self):
        """Test de cohérence de la création de grille 2D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        height, width, channels = 64, 64, 4
        
        # Créer la grille deux fois
        grid1 = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        grid2 = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifier la cohérence
        self.assertEqual(grid1.shape, grid2.shape)
        
        # Les grilles devraient être identiques (même données d'entrée)
        self.assertTrue(np.allclose(grid1, grid2, atol=1e-6), "Les grilles devraient être identiques")
        
        print(f"✅ Cohérence de la création de grille 2D vérifiée")
    
    def test_create_2d_grid_memory_efficiency(self):
        """Test de l'efficacité mémoire de la grille 2D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        height, width, channels = 64, 64, 4
        grid_2d = self.cleaner._create_2d_grid(self.pd_df, height, width, channels)
        
        # Vérifier la taille de la grille
        expected_size = height * width * channels
        actual_size = grid_2d.size
        self.assertEqual(actual_size, expected_size, f"Taille de la grille incorrecte: {actual_size} vs {expected_size}")
        
        # Vérifier la taille en mémoire (float64 = 8 bytes)
        expected_memory = expected_size * 8  # bytes
        actual_memory = grid_2d.nbytes
        self.assertEqual(actual_memory, expected_memory, f"Mémoire utilisée incorrecte: {actual_memory} vs {expected_memory} bytes")
        
        print(f"✅ Efficacité mémoire de la grille 2D:")
        print(f"   Éléments: {actual_size:,}")
        print(f"   Mémoire: {actual_memory / (1024*1024):.2f} MB")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
