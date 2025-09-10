#!/usr/bin/env python3
"""
Test unitaire pour la méthode prepare_data_for_generators_from_df de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode prepare_data_for_generators_from_df
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


class TestDataCleanerPrepareDataForGeneratorsFromDfRealData(unittest.TestCase):
    """Tests pour la méthode prepare_data_for_generators_from_df avec données réelles"""
    
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
    
    def test_prepare_data_for_generators_from_df_pd_csv_real(self):
        """Test de préparation des données pour les générateurs avec DataFrame PD.csv réel"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Vérifier les tenseurs U-Net 2D
        unet_2d_tensor = result['unet_2d']
        self.assertIsInstance(unet_2d_tensor, torch.Tensor)
        self.assertEqual(unet_2d_tensor.shape, (64, 64, 4))
        self.assertEqual(unet_2d_tensor.dtype, torch.float32)
        
        # Vérifier les tenseurs VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        self.assertEqual(voxnet_3d_tensor.dtype, torch.float32)
        
        # Vérifier les métadonnées
        metadata = result['metadata']
        self.assertIn('device_type', metadata)
        self.assertIn('num_points', metadata)
        self.assertIn('spatial_bounds', metadata)
        self.assertIn('value_ranges', metadata)
        
        self.assertEqual(metadata['device_type'], 'pole_dipole')
        self.assertEqual(metadata['num_points'], len(self.pd_df))
        
        # Vérifier que les tenseurs ne sont pas vides
        self.assertGreater(unet_2d_tensor.abs().sum().item(), 0)
        self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0)
        
        print(f"✅ Données PD.csv préparées pour les générateurs depuis DataFrame:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points de données: {metadata['num_points']}")
    
    def test_prepare_data_for_generators_from_df_s_csv_real(self):
        """Test de préparation des données pour les générateurs avec DataFrame S.csv réel"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators_from_df(self.s_df, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Vérifier les tenseurs U-Net 2D
        unet_2d_tensor = result['unet_2d']
        self.assertIsInstance(unet_2d_tensor, torch.Tensor)
        self.assertEqual(unet_2d_tensor.shape, (64, 64, 4))
        
        # Vérifier les tenseurs VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        
        # Vérifier les métadonnées
        metadata = result['metadata']
        self.assertEqual(metadata['device_type'], 'schlumberger')
        self.assertEqual(metadata['num_points'], len(self.s_df))
        
        print(f"✅ Données S.csv préparées pour les générateurs depuis DataFrame:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points de données: {metadata['num_points']}")
    
    def test_prepare_data_for_generators_from_df_different_device_types(self):
        """Test de préparation avec différents types de dispositifs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        device_types = ["pole_dipole", "schlumberger"]
        
        for device_type in device_types:
            with self.subTest(device_type=device_type):
                result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, device_type)
                
                # Vérifications
                self.assertIsInstance(result, dict)
                self.assertIn('unet_2d', result)
                self.assertIn('voxnet_3d', result)
                self.assertEqual(result['metadata']['device_type'], device_type)
                
                print(f"✅ Dispositif {device_type} traité correctement depuis DataFrame")
    
    def test_prepare_data_for_generators_from_df_empty_dataframe(self):
        """Test de préparation avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z', 'resistivity', 'chargeability'])
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators_from_df(empty_df, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Les tenseurs devraient être remplis de zéros
        unet_2d_tensor = result['unet_2d']
        voxnet_3d_tensor = result['voxnet_3d']
        
        self.assertEqual(unet_2d_tensor.abs().sum().item(), 0, "U-Net 2D devrait être zéro pour un DataFrame vide")
        self.assertEqual(voxnet_3d_tensor.abs().sum().item(), 0, "VoxNet 3D devrait être zéro pour un DataFrame vide")
        
        # Les métadonnées devraient indiquer 0 points
        self.assertEqual(result['metadata']['num_points'], 0)
        
        print(f"✅ DataFrame vide géré correctement")
    
    def test_prepare_data_for_generators_from_df_missing_columns(self):
        """Test de préparation avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame avec des colonnes manquantes
        df_missing = self.pd_df.drop(columns=['resistivity', 'chargeability'], errors='ignore')
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators_from_df(df_missing, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Les tenseurs devraient être créés même avec des colonnes manquantes
        unet_2d_tensor = result['unet_2d']
        voxnet_3d_tensor = result['voxnet_3d']
        
        self.assertEqual(unet_2d_tensor.shape, (64, 64, 4))
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        
        print(f"✅ Colonnes manquantes gérées correctement")
    
    def test_prepare_data_for_generators_from_df_tensor_properties(self):
        """Test des propriétés des tenseurs générés"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Vérifier les propriétés des tenseurs U-Net 2D
        unet_2d_tensor = result['unet_2d']
        self.assertFalse(torch.isnan(unet_2d_tensor).any(), "U-Net 2D ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(unet_2d_tensor).any(), "U-Net 2D ne devrait pas contenir d'infini")
        self.assertEqual(unet_2d_tensor.device.type, 'cpu')
        self.assertEqual(unet_2d_tensor.requires_grad, False)
        
        # Vérifier les propriétés des tenseurs VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        self.assertFalse(torch.isnan(voxnet_3d_tensor).any(), "VoxNet 3D ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(voxnet_3d_tensor).any(), "VoxNet 3D ne devrait pas contenir d'infini")
        self.assertEqual(voxnet_3d_tensor.device.type, 'cpu')
        self.assertEqual(voxnet_3d_tensor.requires_grad, False)
        
        print(f"✅ Propriétés des tenseurs vérifiées:")
        print(f"   U-Net 2D - Min: {unet_2d_tensor.min().item():.4f}, Max: {unet_2d_tensor.max().item():.4f}")
        print(f"   VoxNet 3D - Min: {voxnet_3d_tensor.min().item():.4f}, Max: {voxnet_3d_tensor.max().item():.4f}")
    
    def test_prepare_data_for_generators_from_df_metadata_accuracy(self):
        """Test de l'exactitude des métadonnées"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        metadata = result['metadata']
        
        # Vérifier l'exactitude des métadonnées
        self.assertEqual(metadata['num_points'], len(self.pd_df))
        
        # Vérifier les limites spatiales
        spatial_bounds = metadata['spatial_bounds']
        if 'x' in self.pd_df.columns:
            self.assertAlmostEqual(spatial_bounds['x']['min'], self.pd_df['x'].min(), places=2)
            self.assertAlmostEqual(spatial_bounds['x']['max'], self.pd_df['x'].max(), places=2)
        
        if 'y' in self.pd_df.columns:
            self.assertAlmostEqual(spatial_bounds['y']['min'], self.pd_df['y'].min(), places=2)
            self.assertAlmostEqual(spatial_bounds['y']['max'], self.pd_df['y'].max(), places=2)
        
        # Vérifier les plages de valeurs
        value_ranges = metadata['value_ranges']
        if 'resistivity' in value_ranges:
            resistivity_range = value_ranges['resistivity']
            self.assertAlmostEqual(resistivity_range['min'], self.pd_df['resistivity'].min(), places=2)
            self.assertAlmostEqual(resistivity_range['max'], self.pd_df['resistivity'].max(), places=2)
        
        print(f"✅ Métadonnées exactes vérifiées:")
        print(f"   Points: {metadata['num_points']}")
        print(f"   Limites spatiales: {spatial_bounds}")
        print(f"   Plages de valeurs: {value_ranges}")
    
    def test_prepare_data_for_generators_from_df_consistency(self):
        """Test de cohérence de la préparation"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Préparer les données deux fois
        result1 = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        result2 = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Vérifier la cohérence
        self.assertEqual(result1['unet_2d'].shape, result2['unet_2d'].shape)
        self.assertEqual(result1['voxnet_3d'].shape, result2['voxnet_3d'].shape)
        
        # Les tenseurs devraient être identiques (même données d'entrée)
        self.assertTrue(torch.allclose(result1['unet_2d'], result2['unet_2d'], atol=1e-6), 
                       "Les tenseurs U-Net 2D devraient être identiques")
        self.assertTrue(torch.allclose(result1['voxnet_3d'], result2['voxnet_3d'], atol=1e-6), 
                       "Les tenseurs VoxNet 3D devraient être identiques")
        
        # Les métadonnées devraient être identiques
        self.assertEqual(result1['metadata']['num_points'], result2['metadata']['num_points'])
        
        print(f"✅ Cohérence de la préparation vérifiée")
    
    def test_prepare_data_for_generators_from_df_data_integrity(self):
        """Test de l'intégrité des données"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Vérifier que les données d'entrée n'ont pas été modifiées
        original_length = len(self.pd_df)
        self.assertEqual(len(self.pd_df), original_length, "Le DataFrame d'entrée ne devrait pas être modifié")
        
        # Vérifier que les colonnes d'entrée sont préservées
        original_columns = list(self.pd_df.columns)
        self.assertEqual(list(self.pd_df.columns), original_columns, "Les colonnes d'entrée devraient être préservées")
        
        # Vérifier que les données d'entrée sont préservées
        if 'x' in self.pd_df.columns:
            original_x_min = self.pd_df['x'].min()
            self.assertEqual(self.pd_df['x'].min(), original_x_min, "Les données X d'entrée devraient être préservées")
        
        print(f"✅ Intégrité des données d'entrée vérifiée")
    
    def test_prepare_data_for_generators_from_df_memory_efficiency(self):
        """Test de l'efficacité mémoire"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Calculer l'utilisation mémoire
        unet_2d_memory = result['unet_2d'].element_size() * result['unet_2d'].numel()
        voxnet_3d_memory = result['voxnet_3d'].element_size() * result['voxnet_3d'].numel()
        total_memory = unet_2d_memory + voxnet_3d_memory
        
        # Vérifier que l'utilisation mémoire est raisonnable
        self.assertLess(total_memory, 100 * 1024 * 1024, "L'utilisation mémoire devrait être < 100 MB")
        
        print(f"✅ Efficacité mémoire vérifiée:")
        print(f"   U-Net 2D: {unet_2d_memory / (1024*1024):.2f} MB")
        print(f"   VoxNet 3D: {voxnet_3d_memory / (1024*1024):.2f} MB")
        print(f"   Total: {total_memory / (1024*1024):.2f} MB")
    
    def test_prepare_data_for_generators_from_df_channel_mapping(self):
        """Test du mapping des canaux dans les tenseurs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        result = self.cleaner.prepare_data_for_generators_from_df(self.pd_df, "pole_dipole")
        
        # Vérifier le mapping des canaux U-Net 2D
        unet_2d_tensor = result['unet_2d']
        # Canal 0: x, Canal 1: y, Canal 2: resistivity, Canal 3: chargeability
        for channel in range(4):
            channel_data = unet_2d_tensor[:, :, channel]
            self.assertGreater(channel_data.abs().sum().item(), 0, f"Canal U-Net 2D {channel} ne devrait pas être vide")
        
        # Vérifier le mapping des canaux VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        # Canal 0: x, Canal 1: y, Canal 2: z, Canal 3: chargeability
        for channel in range(4):
            channel_data = voxnet_3d_tensor[:, :, :, channel]
            self.assertGreater(channel_data.abs().sum().item(), 0, f"Canal VoxNet 3D {channel} ne devrait pas être vide")
        
        print(f"✅ Mapping des canaux vérifié:")
        print(f"   U-Net 2D: 4 canaux (x, y, resistivity, chargeability)")
        print(f"   VoxNet 3D: 4 canaux (x, y, z, chargeability)")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
