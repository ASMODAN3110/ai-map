#!/usr/bin/env python3
"""
Test unitaire pour la méthode prepare_data_for_generators de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode prepare_data_for_generators
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


class TestDataCleanerPrepareDataForGeneratorsRealData(unittest.TestCase):
    """Tests pour la méthode prepare_data_for_generators avec données réelles"""
    
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
    
    def test_prepare_data_for_generators_pd_csv_real(self):
        """Test de préparation des données pour les générateurs avec PD.csv réel"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un fichier CSV temporaire avec les données réelles
        temp_csv = self.test_dir / "processed" / "pd_test.csv"
        self.pd_df.to_csv(temp_csv, index=False)
        
        # Appeler la méthode de préparation des données
        result = self.cleaner.prepare_data_for_generators(temp_csv, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Vérifier les tenseurs U-Net 2D
        unet_2d_tensor = result['unet_2d']
        self.assertIsInstance(unet_2d_tensor, torch.Tensor)
        self.assertEqual(unet_2d_tensor.shape, (64, 64, 4))  # (height, width, channels)
        self.assertEqual(unet_2d_tensor.dtype, torch.float32)
        
        # Vérifier les tenseurs VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        self.assertIsInstance(voxnet_3d_tensor, torch.Tensor)
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))  # (depth, height, width, channels)
        self.assertEqual(voxnet_3d_tensor.dtype, torch.float32)
        
        # Vérifier les métadonnées
        metadata = result['metadata']
        self.assertIn('device_type', metadata)
        self.assertIn('num_points', metadata)
        self.assertIn('spatial_bounds', metadata)
        self.assertIn('value_ranges', metadata)
        
        self.assertEqual(metadata['device_type'], 'pole_dipole')
        self.assertEqual(metadata['num_points'], len(self.pd_df))
        
        print(f"✅ Données PD.csv préparées pour les générateurs:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points de données: {metadata['num_points']}")
    
    def test_prepare_data_for_generators_s_csv_real(self):
        """Test de préparation des données pour les générateurs avec S.csv réel"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Créer un fichier CSV temporaire avec les données réelles
        temp_csv = self.test_dir / "processed" / "s_test.csv"
        self.s_df.to_csv(temp_csv, index=False)
        
        # Appeler la méthode de préparation des données
        result = self.cleaner.prepare_data_for_generators(temp_csv, "schlumberger")
        
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
        
        print(f"✅ Données S.csv préparées pour les générateurs:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points de données: {metadata['num_points']}")
    
    def test_prepare_data_for_generators_invalid_file(self):
        """Test de préparation avec un fichier invalide"""
        # Créer un fichier CSV invalide
        invalid_csv = self.test_dir / "processed" / "invalid.csv"
        with open(invalid_csv, 'w') as f:
            f.write("invalid,data\n")
            f.write("not,enough,columns\n")
        
        # La méthode devrait lever une exception
        with self.assertRaises(Exception):
            self.cleaner.prepare_data_for_generators(invalid_csv, "pole_dipole")
        
        print(f"✅ Fichier invalide correctement rejeté")
    
    def test_prepare_data_for_generators_missing_file(self):
        """Test de préparation avec un fichier inexistant"""
        missing_csv = self.test_dir / "processed" / "missing.csv"
        
        # La méthode devrait lever une exception
        with self.assertRaises(Exception):
            self.cleaner.prepare_data_for_generators(missing_csv, "pole_dipole")
        
        print(f"✅ Fichier manquant correctement géré")
    
    def test_prepare_data_for_generators_different_device_types(self):
        """Test de préparation avec différents types de dispositifs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un fichier CSV temporaire
        temp_csv = self.test_dir / "processed" / "device_test.csv"
        self.pd_df.to_csv(temp_csv, index=False)
        
        # Tester avec différents types de dispositifs
        device_types = ["pole_dipole", "schlumberger"]
        
        for device_type in device_types:
            with self.subTest(device_type=device_type):
                result = self.cleaner.prepare_data_for_generators(temp_csv, device_type)
                
                # Vérifications
                self.assertIsInstance(result, dict)
                self.assertIn('unet_2d', result)
                self.assertIn('voxnet_3d', result)
                self.assertEqual(result['metadata']['device_type'], device_type)
                
                print(f"✅ Dispositif {device_type} traité correctement")
    
    def test_prepare_data_for_generators_tensor_properties(self):
        """Test des propriétés des tenseurs générés"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un fichier CSV temporaire
        temp_csv = self.test_dir / "processed" / "tensor_test.csv"
        self.pd_df.to_csv(temp_csv, index=False)
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators(temp_csv, "pole_dipole")
        
        # Vérifier les propriétés des tenseurs U-Net 2D
        unet_2d_tensor = result['unet_2d']
        self.assertFalse(torch.isnan(unet_2d_tensor).any(), "U-Net 2D ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(unet_2d_tensor).any(), "U-Net 2D ne devrait pas contenir d'infini")
        
        # Vérifier les propriétés des tenseurs VoxNet 3D
        voxnet_3d_tensor = result['voxnet_3d']
        self.assertFalse(torch.isnan(voxnet_3d_tensor).any(), "VoxNet 3D ne devrait pas contenir de NaN")
        self.assertFalse(torch.isinf(voxnet_3d_tensor).any(), "VoxNet 3D ne devrait pas contenir d'infini")
        
        # Vérifier que les tenseurs ne sont pas tous zéros
        self.assertGreater(unet_2d_tensor.abs().sum().item(), 0, "U-Net 2D ne devrait pas être entièrement zéro")
        self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0, "VoxNet 3D ne devrait pas être entièrement zéro")
        
        print(f"✅ Propriétés des tenseurs vérifiées:")
        print(f"   U-Net 2D - Min: {unet_2d_tensor.min().item():.4f}, Max: {unet_2d_tensor.max().item():.4f}")
        print(f"   VoxNet 3D - Min: {voxnet_3d_tensor.min().item():.4f}, Max: {voxnet_3d_tensor.max().item():.4f}")
    
    def test_prepare_data_for_generators_metadata_accuracy(self):
        """Test de l'exactitude des métadonnées"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un fichier CSV temporaire
        temp_csv = self.test_dir / "processed" / "metadata_test.csv"
        self.pd_df.to_csv(temp_csv, index=False)
        
        # Appeler la méthode
        result = self.cleaner.prepare_data_for_generators(temp_csv, "pole_dipole")
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


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
