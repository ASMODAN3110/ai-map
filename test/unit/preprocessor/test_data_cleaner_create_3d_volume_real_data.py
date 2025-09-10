#!/usr/bin/env python3
"""
Test unitaire pour la méthode _create_3d_volume de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _create_3d_volume
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


class TestDataCleanerCreate3dVolumeRealData(unittest.TestCase):
    """Tests pour la méthode _create_3d_volume avec données réelles"""
    
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
    
    def test_create_3d_volume_pd_csv_real(self):
        """Test de création de volume 3D avec les vraies données PD.csv"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Paramètres du volume
        depth, height, width, channels = 16, 16, 16, 4
        
        # Appeler la méthode privée via reflection
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(volume_3d, np.ndarray)
        self.assertEqual(volume_3d.shape, (depth, height, width, channels))
        self.assertEqual(volume_3d.dtype, np.float64)
        
        # Vérifier que le volume n'est pas vide
        self.assertGreater(np.abs(volume_3d).sum(), 0, "Le volume ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(np.isnan(volume_3d).any(), "Le volume ne devrait pas contenir de NaN")
        self.assertFalse(np.isinf(volume_3d).any(), "Le volume ne devrait pas contenir d'infini")
        
        print(f"✅ Volume 3D créé avec PD.csv:")
        print(f"   Shape: {volume_3d.shape}")
        print(f"   Min: {volume_3d.min():.4f}")
        print(f"   Max: {volume_3d.max():.4f}")
        print(f"   Mean: {volume_3d.mean():.4f}")
    
    def test_create_3d_volume_s_csv_real(self):
        """Test de création de volume 3D avec les vraies données S.csv"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Paramètres du volume
        depth, height, width, channels = 16, 16, 16, 4
        
        # Appeler la méthode privée via reflection
        volume_3d = self.cleaner._create_3d_volume(self.s_df, depth, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(volume_3d, np.ndarray)
        self.assertEqual(volume_3d.shape, (depth, height, width, channels))
        self.assertEqual(volume_3d.dtype, np.float64)
        
        # Vérifier que le volume n'est pas vide
        self.assertGreater(np.abs(volume_3d).sum(), 0, "Le volume ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(np.isnan(volume_3d).any(), "Le volume ne devrait pas contenir de NaN")
        self.assertFalse(np.isinf(volume_3d).any(), "Le volume ne devrait pas contenir d'infini")
        
        print(f"✅ Volume 3D créé avec S.csv:")
        print(f"   Shape: {volume_3d.shape}")
        print(f"   Min: {volume_3d.min():.4f}")
        print(f"   Max: {volume_3d.max():.4f}")
        print(f"   Mean: {volume_3d.mean():.4f}")
    
    def test_create_3d_volume_different_sizes(self):
        """Test de création de volume 3D avec différentes tailles"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Tester différentes tailles de volume
        test_sizes = [
            (16, 16, 16, 4),   # Petit volume
            (32, 32, 32, 4),   # Volume standard
            (64, 64, 64, 4),   # Grand volume
            (16, 32, 16, 4),   # Volume rectangulaire
        ]
        
        for depth, height, width, channels in test_sizes:
            with self.subTest(size=(depth, height, width, channels)):
                volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
                
                # Vérifications
                self.assertIsInstance(volume_3d, np.ndarray)
                self.assertEqual(volume_3d.shape, (depth, height, width, channels))
                self.assertGreater(np.abs(volume_3d).sum(), 0)
                
                print(f"✅ Volume {depth}x{height}x{width}x{channels} créé correctement")
    
    def test_create_3d_volume_empty_dataframe(self):
        """Test de création de volume 3D avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        depth, height, width, channels = 16, 16, 16, 4
        
        # La méthode devrait gérer un DataFrame vide
        volume_3d = self.cleaner._create_3d_volume(empty_df, depth, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(volume_3d, np.ndarray)
        self.assertEqual(volume_3d.shape, (depth, height, width, channels))
        
        # Le volume devrait être rempli de zéros pour un DataFrame vide
        self.assertEqual(np.abs(volume_3d).sum(), 0, "Le volume devrait être zéro pour un DataFrame vide")
        
        print(f"✅ Volume 3D avec DataFrame vide géré correctement")
    
    def test_create_3d_volume_missing_columns(self):
        """Test de création de volume 3D avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame avec des colonnes manquantes
        df_missing = self.pd_df.drop(columns=['z', 'chargeability'], errors='ignore')
        depth, height, width, channels = 16, 16, 16, 4
        
        # La méthode devrait gérer les colonnes manquantes
        volume_3d = self.cleaner._create_3d_volume(df_missing, depth, height, width, channels)
        
        # Vérifications
        self.assertIsInstance(volume_3d, np.ndarray)
        self.assertEqual(volume_3d.shape, (depth, height, width, channels))
        
        print(f"✅ Volume 3D avec colonnes manquantes géré correctement")
    
    def test_create_3d_volume_channel_mapping(self):
        """Test du mapping des canaux dans le volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier le mapping des canaux
        # Canal 0: x, Canal 1: y, Canal 2: z, Canal 3: chargeability
        x_channel = volume_3d[:, :, :, 0]
        y_channel = volume_3d[:, :, :, 1]
        z_channel = volume_3d[:, :, :, 2]
        chargeability_channel = volume_3d[:, :, :, 3]
        
        # Vérifier que les canaux contiennent des données
        self.assertGreater(np.abs(x_channel).sum(), 0, "Canal x ne devrait pas être vide")
        self.assertGreater(np.abs(y_channel).sum(), 0, "Canal y ne devrait pas être vide")
        self.assertGreater(np.abs(z_channel).sum(), 0, "Canal z ne devrait pas être vide")
        self.assertGreater(np.abs(chargeability_channel).sum(), 0, "Canal chargeabilité ne devrait pas être vide")
        
        # Vérifier que les coordonnées correspondent aux données réelles
        real_x_min, real_x_max = self.pd_df['x'].min(), self.pd_df['x'].max()
        real_y_min, real_y_max = self.pd_df['y'].min(), self.pd_df['y'].max()
        real_z_min, real_z_max = self.pd_df['z'].min(), self.pd_df['z'].max()
        
        volume_x_min, volume_x_max = x_channel.min(), x_channel.max()
        volume_y_min, volume_y_max = y_channel.min(), y_channel.max()
        volume_z_min, volume_z_max = z_channel.min(), z_channel.max()
        
        # Vérifier que les valeurs du volume sont dans la plage des données réelles (avec tolérance)
        self.assertGreaterEqual(volume_x_min, real_x_min - 1.0, msg="X min devrait être proche des données réelles")
        self.assertLessEqual(volume_x_max, real_x_max + 1.0, msg="X max devrait être proche des données réelles")
        self.assertGreaterEqual(volume_y_min, real_y_min - 1.0, msg="Y min devrait être proche des données réelles")
        self.assertLessEqual(volume_y_max, real_y_max + 1.0, msg="Y max devrait être proche des données réelles")
        self.assertGreaterEqual(volume_z_min, real_z_min - 1.0, msg="Z min devrait être proche des données réelles")
        self.assertLessEqual(volume_z_max, real_z_max + 1.0, msg="Z max devrait être proche des données réelles")
        
        print(f"✅ Mapping des canaux vérifié:")
        print(f"   Canal X: {volume_x_min:.2f} à {volume_x_max:.2f}")
        print(f"   Canal Y: {volume_y_min:.2f} à {volume_y_max:.2f}")
        print(f"   Canal Z: {volume_z_min:.2f} à {volume_z_max:.2f}")
        print(f"   Canal Chargeabilité: {chargeability_channel.min():.4f} à {chargeability_channel.max():.4f}")
    
    def test_create_3d_volume_interpolation_quality(self):
        """Test de la qualité de l'interpolation sur le volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier que l'interpolation a bien réparti les données
        # Le volume ne devrait pas avoir trop de zéros
        zero_ratio = np.sum(volume_3d == 0) / volume_3d.size
        self.assertLess(zero_ratio, 0.7, f"Trop de zéros dans le volume: {zero_ratio:.2%}")
        
        # Vérifier que les valeurs sont réparties de manière cohérente
        for channel in range(channels):
            channel_data = volume_3d[:, :, :, channel]
            
            # Vérifier que les valeurs ne sont pas toutes identiques
            unique_values = np.unique(channel_data)
            self.assertGreater(len(unique_values), 1, f"Canal {channel} ne devrait pas avoir qu'une seule valeur unique")
            
            # Vérifier la variance (doit être > 0)
            variance = np.var(channel_data)
            self.assertGreater(variance, 0, f"Canal {channel} devrait avoir une variance > 0")
        
        print(f"✅ Qualité de l'interpolation vérifiée:")
        print(f"   Ratio de zéros: {zero_ratio:.2%}")
        print(f"   Variance par canal: {[np.var(volume_3d[:, :, :, i]) for i in range(channels)]}")
    
    def test_create_3d_volume_spatial_coverage(self):
        """Test de la couverture spatiale du volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier que le volume couvre bien l'espace des données
        x_channel = volume_3d[:, :, :, 0]
        y_channel = volume_3d[:, :, :, 1]
        z_channel = volume_3d[:, :, :, 2]
        
        # Calculer les limites du volume
        volume_x_min, volume_x_max = x_channel.min(), x_channel.max()
        volume_y_min, volume_y_max = y_channel.min(), y_channel.max()
        volume_z_min, volume_z_max = z_channel.min(), z_channel.max()
        
        # Calculer les limites des données réelles
        real_x_min, real_x_max = self.pd_df['x'].min(), self.pd_df['x'].max()
        real_y_min, real_y_max = self.pd_df['y'].min(), self.pd_df['y'].max()
        real_z_min, real_z_max = self.pd_df['z'].min(), self.pd_df['z'].max()
        
        # Vérifier que le volume couvre au moins l'espace des données
        self.assertLessEqual(volume_x_min, real_x_min, "Le volume devrait couvrir au moins l'espace des données (X min)")
        self.assertGreaterEqual(volume_x_max, real_x_max, "Le volume devrait couvrir au moins l'espace des données (X max)")
        self.assertLessEqual(volume_y_min, real_y_min, "Le volume devrait couvrir au moins l'espace des données (Y min)")
        self.assertGreaterEqual(volume_y_max, real_y_max, "Le volume devrait couvrir au moins l'espace des données (Y max)")
        self.assertLessEqual(volume_z_min, real_z_min, "Le volume devrait couvrir au moins l'espace des données (Z min)")
        self.assertGreaterEqual(volume_z_max, real_z_max, "Le volume devrait couvrir au moins l'espace des données (Z max)")
        
        # Calculer la résolution spatiale
        x_resolution = (volume_x_max - volume_x_min) / (width - 1)
        y_resolution = (volume_y_max - volume_y_min) / (height - 1)
        z_resolution = (volume_z_max - volume_z_min) / (depth - 1)
        
        print(f"✅ Couverture spatiale vérifiée:")
        print(f"   Volume X: {volume_x_min:.2f} à {volume_x_max:.2f} (résolution: {x_resolution:.2f}m)")
        print(f"   Volume Y: {volume_y_min:.2f} à {volume_y_max:.2f} (résolution: {y_resolution:.2f}m)")
        print(f"   Volume Z: {volume_z_min:.2f} à {volume_z_max:.2f} (résolution: {z_resolution:.2f}m)")
        print(f"   Données X: {real_x_min:.2f} à {real_x_max:.2f}")
        print(f"   Données Y: {real_y_min:.2f} à {real_y_max:.2f}")
        print(f"   Données Z: {real_z_min:.2f} à {real_z_max:.2f}")
    
    def test_create_3d_volume_consistency(self):
        """Test de cohérence de la création de volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        
        # Créer le volume deux fois
        volume1 = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        volume2 = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier la cohérence
        self.assertEqual(volume1.shape, volume2.shape)
        
        # Les volumes devraient être identiques (même données d'entrée)
        self.assertTrue(np.allclose(volume1, volume2, atol=1e-6), "Les volumes devraient être identiques")
        
        print(f"✅ Cohérence de la création de volume 3D vérifiée")
    
    def test_create_3d_volume_memory_efficiency(self):
        """Test de l'efficacité mémoire du volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier la taille du volume
        expected_size = depth * height * width * channels
        actual_size = volume_3d.size
        self.assertEqual(actual_size, expected_size, f"Taille du volume incorrecte: {actual_size} vs {expected_size}")
        
        # Vérifier la taille en mémoire (float64 = 8 bytes)
        expected_memory = expected_size * 8  # bytes
        actual_memory = volume_3d.nbytes
        self.assertEqual(actual_memory, expected_memory, f"Mémoire utilisée incorrecte: {actual_memory} vs {expected_memory} bytes")
        
        print(f"✅ Efficacité mémoire du volume 3D:")
        print(f"   Éléments: {actual_size:,}")
        print(f"   Mémoire: {actual_memory / (1024*1024):.2f} MB")
    
    def test_create_3d_volume_3d_structure(self):
        """Test de la structure 3D du volume"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        depth, height, width, channels = 16, 16, 16, 4
        volume_3d = self.cleaner._create_3d_volume(self.pd_df, depth, height, width, channels)
        
        # Vérifier que le volume a bien une structure 3D
        self.assertEqual(len(volume_3d.shape), 4, "Le volume devrait avoir 4 dimensions")
        self.assertEqual(volume_3d.shape[0], depth, "La première dimension devrait être la profondeur")
        self.assertEqual(volume_3d.shape[1], height, "La deuxième dimension devrait être la hauteur")
        self.assertEqual(volume_3d.shape[2], width, "La troisième dimension devrait être la largeur")
        self.assertEqual(volume_3d.shape[3], channels, "La quatrième dimension devrait être les canaux")
        
        # Vérifier que chaque dimension a des données
        for dim in range(3):  # depth, height, width
            # Prendre une tranche dans chaque dimension
            if dim == 0:  # depth
                slice_data = volume_3d[0, :, :, :]
            elif dim == 1:  # height
                slice_data = volume_3d[:, 0, :, :]
            else:  # width
                slice_data = volume_3d[:, :, 0, :]
            
            self.assertGreater(np.abs(slice_data).sum(), 0, f"La dimension {dim} devrait contenir des données")
        
        print(f"✅ Structure 3D du volume vérifiée:")
        print(f"   Dimensions: {volume_3d.shape}")
        print(f"   Chaque dimension contient des données")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
