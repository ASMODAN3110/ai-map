#!/usr/bin/env python3
"""
Test unitaire pour la méthode _prepare_voxnet_3d_data de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _prepare_voxnet_3d_data
avec des données réelles extraites des fichiers PD.csv et S.csv.
"""

import sys
import unittest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerPrepareVoxnet3dDataRealData(unittest.TestCase):
    """Tests pour la méthode _prepare_voxnet_3d_data avec données réelles"""
    
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
    
    def test_prepare_voxnet_3d_data_pd_csv_real(self):
        """Test de préparation VoxNet 3D avec les vraies données PD.csv"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode privée via reflection
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))  # (depth, height, width, channels)
        self.assertEqual(voxnet_3d_tensor.dtype, torch.float32)
        
        # Vérifier que le tenseur n'est pas vide
        self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0, "Le tenseur ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(torch.isnan(voxnet_3d_tensor).any(), "Le tenseur ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(voxnet_3d_tensor).any(), "Le tenseur ne devrait pas contenir d'infini")
        
        # Vérifier les canaux
        # Canal 0: x, Canal 1: y, Canal 2: z, Canal 3: chargeability
        for channel in range(4):
            channel_data = voxnet_3d_tensor[:, :, :, channel]
            self.assertGreater(channel_data.abs().sum().item(), 0, f"Canal {channel} ne devrait pas être entièrement zéro")
        
        print(f"✅ VoxNet 3D préparé avec PD.csv:")
        print(f"   Shape: {voxnet_3d_tensor.shape}")
        print(f"   Min: {voxnet_3d_tensor.min().item():.4f}")
        print(f"   Max: {voxnet_3d_tensor.max().item():.4f}")
        print(f"   Mean: {voxnet_3d_tensor.mean().item():.4f}")
    
    def test_prepare_voxnet_3d_data_s_csv_real(self):
        """Test de préparation VoxNet 3D avec les vraies données S.csv"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Appeler la méthode privée via reflection
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.s_df, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        self.assertEqual(voxnet_3d_tensor.dtype, torch.float32)
        
        # Vérifier que le tenseur n'est pas vide
        self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0, "Le tenseur ne devrait pas être entièrement zéro")
        
        # Vérifier qu'il n'y a pas de NaN ou d'infini
        self.assertFalse(torch.isnan(voxnet_3d_tensor).any(), "Le tenseur ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(voxnet_3d_tensor).any(), "Le tenseur ne devrait pas contenir d'infini")
        
        print(f"✅ VoxNet 3D préparé avec S.csv:")
        print(f"   Shape: {voxnet_3d_tensor.shape}")
        print(f"   Min: {voxnet_3d_tensor.min().item():.4f}")
        print(f"   Max: {voxnet_3d_tensor.max().item():.4f}")
        print(f"   Mean: {voxnet_3d_tensor.mean().item():.4f}")
    
    def test_prepare_voxnet_3d_data_different_device_types(self):
        """Test de préparation VoxNet 3D avec différents types de dispositifs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        device_types = ["pole_dipole", "schlumberger"]
        
        for device_type in device_types:
            with self.subTest(device_type=device_type):
                voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, device_type)
                
                # Vérifications
                self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
                self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
                self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0)
                
                print(f"✅ VoxNet 3D préparé pour {device_type}: {voxnet_3d_tensor.shape}")
    
    def test_prepare_voxnet_3d_data_empty_dataframe(self):
        """Test de préparation VoxNet 3D avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        
        # La méthode devrait gérer un DataFrame vide
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(empty_df, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        
        # Le tenseur devrait être rempli de zéros pour un DataFrame vide
        self.assertEqual(voxnet_3d_tensor.abs().sum().item(), 0, "Le tenseur devrait être zéro pour un DataFrame vide")
        
        print(f"✅ VoxNet 3D avec DataFrame vide géré correctement")
    
    def test_prepare_voxnet_3d_data_missing_columns(self):
        """Test de préparation VoxNet 3D avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame avec des colonnes manquantes
        df_missing = self.pd_df.drop(columns=['z', 'chargeability'], errors='ignore')
        
        # La méthode devrait gérer les colonnes manquantes
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(df_missing, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        
        print(f"✅ VoxNet 3D avec colonnes manquantes géré correctement")
    
    def test_prepare_voxnet_3d_data_tensor_properties(self):
        """Test des propriétés détaillées du tenseur VoxNet 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Vérifier les propriétés du tenseur
        self.assertEqual(voxnet_3d_tensor.device.type, 'cpu')  # Device par défaut
        self.assertEqual(voxnet_3d_tensor.requires_grad, False)  # Pas de gradient requis
        
        # Vérifier la distribution des valeurs par canal
        for channel in range(4):
            channel_data = voxnet_3d_tensor[:, :, :, channel]
            
            # Vérifier que les valeurs sont dans des plages raisonnables
            if channel in [0, 1, 2]:  # Coordonnées x, y, z
                self.assertGreaterEqual(channel_data.min().item(), 0, f"Canal {channel} (coordonnées) devrait être >= 0")
            elif channel == 3:  # Chargeabilité
                self.assertGreaterEqual(channel_data.min().item(), 0, f"Canal {channel} (chargeabilité) devrait être >= 0")
            
            print(f"   Canal {channel}: Min={channel_data.min().item():.4f}, Max={channel_data.max().item():.4f}, Mean={channel_data.mean().item():.4f}")
    
    def test_prepare_voxnet_3d_data_consistency(self):
        """Test de cohérence de la préparation VoxNet 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Préparer les données deux fois
        tensor1 = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        tensor2 = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Vérifier la cohérence
        self.assertEqual(tensor1.shape, tensor2.shape)
        
        # Les tenseurs devraient être identiques (même données d'entrée)
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=1e-6), "Les tenseurs devraient être identiques")
        
        print(f"✅ Cohérence de la préparation VoxNet 3D vérifiée")
    
    def test_prepare_voxnet_3d_data_volume_interpolation(self):
        """Test de l'interpolation sur le volume 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Vérifier que l'interpolation a fonctionné
        # Les coordonnées devraient être réparties sur le volume
        x_channel = voxnet_3d_tensor[:, :, :, 0]  # Canal x
        y_channel = voxnet_3d_tensor[:, :, :, 1]  # Canal y
        z_channel = voxnet_3d_tensor[:, :, :, 2]  # Canal z
        
        # Vérifier que les coordonnées couvrent le volume
        x_min, x_max = x_channel.min().item(), x_channel.max().item()
        y_min, y_max = y_channel.min().item(), y_channel.max().item()
        z_min, z_max = z_channel.min().item(), z_channel.max().item()
        
        # Les coordonnées devraient correspondre aux données d'entrée
        real_x_min, real_x_max = self.pd_df['x'].min(), self.pd_df['x'].max()
        real_y_min, real_y_max = self.pd_df['y'].min(), self.pd_df['y'].max()
        real_z_min, real_z_max = self.pd_df['z'].min(), self.pd_df['z'].max()
        
        self.assertAlmostEqual(x_min, real_x_min, places=2, msg="X min devrait correspondre aux données réelles")
        self.assertAlmostEqual(x_max, real_x_max, places=2, msg="X max devrait correspondre aux données réelles")
        self.assertAlmostEqual(y_min, real_y_min, places=2, msg="Y min devrait correspondre aux données réelles")
        self.assertAlmostEqual(y_max, real_y_max, places=2, msg="Y max devrait correspondre aux données réelles")
        self.assertAlmostEqual(z_min, real_z_min, places=2, msg="Z min devrait correspondre aux données réelles")
        self.assertAlmostEqual(z_max, real_z_max, places=2, msg="Z max devrait correspondre aux données réelles")
        
        print(f"✅ Interpolation sur volume 3D vérifiée:")
        print(f"   X: {x_min:.2f} à {x_max:.2f} (réel: {real_x_min:.2f} à {real_x_max:.2f})")
        print(f"   Y: {y_min:.2f} à {y_max:.2f} (réel: {real_y_min:.2f} à {real_y_max:.2f})")
        print(f"   Z: {z_min:.2f} à {z_max:.2f} (réel: {real_z_min:.2f} à {real_z_max:.2f})")
    
    def test_prepare_voxnet_3d_data_memory_usage(self):
        """Test de l'utilisation mémoire du tenseur VoxNet 3D"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Vérifier la taille du tenseur
        expected_size = 32 * 32 * 32 * 4  # depth * height * width * channels
        actual_size = voxnet_3d_tensor.numel()
        self.assertEqual(actual_size, expected_size, f"Taille du tenseur incorrecte: {actual_size} vs {expected_size}")
        
        # Vérifier la taille en mémoire (float32 = 4 bytes)
        expected_memory = expected_size * 4  # bytes
        actual_memory = voxnet_3d_tensor.element_size() * voxnet_3d_tensor.numel()
        self.assertEqual(actual_memory, expected_memory, f"Mémoire utilisée incorrecte: {actual_memory} vs {expected_memory} bytes")
        
        print(f"✅ Utilisation mémoire VoxNet 3D:")
        print(f"   Éléments: {actual_size:,}")
        print(f"   Mémoire: {actual_memory / (1024*1024):.2f} MB")
    
    def test_prepare_voxnet_3d_data_channel_distribution(self):
        """Test de la distribution des données par canal"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        voxnet_3d_tensor = self.cleaner._prepare_voxnet_3d_data(self.pd_df, "pole_dipole")
        
        # Analyser la distribution par canal
        for channel in range(4):
            channel_data = voxnet_3d_tensor[:, :, :, channel]
            
            # Calculer les statistiques
            min_val = channel_data.min().item()
            max_val = channel_data.max().item()
            mean_val = channel_data.mean().item()
            std_val = channel_data.std().item()
            
            # Vérifier que les statistiques sont raisonnables
            self.assertGreaterEqual(min_val, 0, f"Canal {channel}: min devrait être >= 0")
            self.assertGreater(max_val, min_val, f"Canal {channel}: max devrait être > min")
            self.assertGreater(std_val, 0, f"Canal {channel}: std devrait être > 0")
            
            print(f"   Canal {channel}: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f}")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
