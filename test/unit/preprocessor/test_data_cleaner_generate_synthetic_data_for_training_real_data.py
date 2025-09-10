#!/usr/bin/env python3
"""
Test unitaire pour la méthode generate_synthetic_data_for_training de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode generate_synthetic_data_for_training
avec des paramètres réalistes et des données synthétiques.
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


class TestDataCleanerGenerateSyntheticDataForTrainingRealData(unittest.TestCase):
    """Tests pour la méthode generate_synthetic_data_for_training avec données réalistes"""
    
    def setUp(self):
        """Configuration avant chaque test"""
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
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_generate_synthetic_data_for_training_pole_dipole(self):
        """Test de génération de données synthétiques pour Pole-Dipole"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        # Appeler la méthode
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
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
        self.assertEqual(metadata['device_type'], device_type)
        self.assertEqual(metadata['num_points'], num_samples)
        
        # Vérifier que les tenseurs ne sont pas vides
        self.assertGreater(unet_2d_tensor.abs().sum().item(), 0)
        self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0)
        
        print(f"✅ Données synthétiques Pole-Dipole générées:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points: {metadata['num_points']}")
    
    def test_generate_synthetic_data_for_training_schlumberger(self):
        """Test de génération de données synthétiques pour Schlumberger"""
        num_samples = 500
        device_type = "schlumberger"
        
        # Appeler la méthode
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('unet_2d', result)
        self.assertIn('voxnet_3d', result)
        self.assertIn('metadata', result)
        
        # Vérifier les tenseurs
        unet_2d_tensor = result['unet_2d']
        voxnet_3d_tensor = result['voxnet_3d']
        
        self.assertEqual(unet_2d_tensor.shape, (64, 64, 4))
        self.assertEqual(voxnet_3d_tensor.shape, (32, 32, 32, 4))
        
        # Vérifier les métadonnées
        metadata = result['metadata']
        self.assertEqual(metadata['device_type'], device_type)
        self.assertEqual(metadata['num_points'], num_samples)
        
        print(f"✅ Données synthétiques Schlumberger générées:")
        print(f"   U-Net 2D: {unet_2d_tensor.shape}")
        print(f"   VoxNet 3D: {voxnet_3d_tensor.shape}")
        print(f"   Points: {metadata['num_points']}")
    
    def test_generate_synthetic_data_for_training_different_sample_sizes(self):
        """Test de génération avec différentes tailles d'échantillons"""
        device_type = "pole_dipole"
        sample_sizes = [100, 500, 1000, 2000]
        
        for num_samples in sample_sizes:
            with self.subTest(num_samples=num_samples):
                result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
                
                # Vérifications
                self.assertIsInstance(result, dict)
                self.assertIn('unet_2d', result)
                self.assertIn('voxnet_3d', result)
                self.assertIn('metadata', result)
                
                # Vérifier que le nombre de points correspond
                metadata = result['metadata']
                self.assertEqual(metadata['num_points'], num_samples)
                
                print(f"✅ {num_samples} échantillons générés correctement")
    
    def test_generate_synthetic_data_for_training_tensor_properties(self):
        """Test des propriétés des tenseurs générés"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
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
    
    def test_generate_synthetic_data_for_training_metadata_accuracy(self):
        """Test de l'exactitude des métadonnées"""
        num_samples = 1000
        device_type = "schlumberger"
        
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        metadata = result['metadata']
        
        # Vérifier les métadonnées
        self.assertEqual(metadata['device_type'], device_type)
        self.assertEqual(metadata['num_points'], num_samples)
        self.assertIn('spatial_bounds', metadata)
        self.assertIn('value_ranges', metadata)
        
        # Vérifier les limites spatiales
        spatial_bounds = metadata['spatial_bounds']
        self.assertIn('x', spatial_bounds)
        self.assertIn('y', spatial_bounds)
        self.assertIn('z', spatial_bounds)
        
        # Vérifier les plages de valeurs
        value_ranges = metadata['value_ranges']
        self.assertIn('resistivity', value_ranges)
        self.assertIn('chargeability', value_ranges)
        
        print(f"✅ Métadonnées exactes vérifiées:")
        print(f"   Type de dispositif: {metadata['device_type']}")
        print(f"   Nombre de points: {metadata['num_points']}")
        print(f"   Limites spatiales: {spatial_bounds}")
        print(f"   Plages de valeurs: {value_ranges}")
    
    def test_generate_synthetic_data_for_training_consistency(self):
        """Test de cohérence de la génération"""
        num_samples = 500
        device_type = "pole_dipole"
        
        # Générer les données deux fois
        result1 = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        result2 = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
        # Vérifier que les structures sont identiques
        self.assertEqual(result1.keys(), result2.keys())
        self.assertEqual(result1['unet_2d'].shape, result2['unet_2d'].shape)
        self.assertEqual(result1['voxnet_3d'].shape, result2['voxnet_3d'].shape)
        
        # Vérifier que les métadonnées sont identiques
        self.assertEqual(result1['metadata']['device_type'], result2['metadata']['device_type'])
        self.assertEqual(result1['metadata']['num_points'], result2['metadata']['num_points'])
        
        # Les données devraient être différentes (génération aléatoire)
        self.assertFalse(torch.allclose(result1['unet_2d'], result2['unet_2d'], atol=1e-6), 
                        "Les données synthétiques devraient être différentes")
        
        print(f"✅ Cohérence de la génération vérifiée")
    
    def test_generate_synthetic_data_for_training_device_specific_characteristics(self):
        """Test des caractéristiques spécifiques par dispositif"""
        num_samples = 1000
        
        # Tester les deux types de dispositifs
        device_types = ["pole_dipole", "schlumberger"]
        
        for device_type in device_types:
            with self.subTest(device_type=device_type):
                result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
                
                # Vérifier que le type de dispositif est correct
                metadata = result['metadata']
                self.assertEqual(metadata['device_type'], device_type)
                
                # Vérifier que les données sont générées
                unet_2d_tensor = result['unet_2d']
                voxnet_3d_tensor = result['voxnet_3d']
                
                self.assertGreater(unet_2d_tensor.abs().sum().item(), 0)
                self.assertGreater(voxnet_3d_tensor.abs().sum().item(), 0)
                
                print(f"✅ Caractéristiques {device_type} vérifiées")
    
    def test_generate_synthetic_data_for_training_edge_cases(self):
        """Test des cas limites"""
        device_type = "pole_dipole"
        
        # Test avec un seul échantillon
        result_single = self.cleaner.generate_synthetic_data_for_training(1, device_type)
        self.assertEqual(result_single['metadata']['num_points'], 1)
        
        # Test avec un grand nombre d'échantillons
        result_large = self.cleaner.generate_synthetic_data_for_training(10000, device_type)
        self.assertEqual(result_large['metadata']['num_points'], 10000)
        
        print(f"✅ Cas limites gérés correctement:")
        print(f"   1 échantillon: {result_single['unet_2d'].shape}")
        print(f"   10000 échantillons: {result_large['unet_2d'].shape}")
    
    def test_generate_synthetic_data_for_training_memory_usage(self):
        """Test de l'utilisation mémoire"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
        # Calculer l'utilisation mémoire
        unet_2d_memory = result['unet_2d'].element_size() * result['unet_2d'].numel()
        voxnet_3d_memory = result['voxnet_3d'].element_size() * result['voxnet_3d'].numel()
        total_memory = unet_2d_memory + voxnet_3d_memory
        
        # Vérifier que l'utilisation mémoire est raisonnable
        self.assertLess(total_memory, 100 * 1024 * 1024, "L'utilisation mémoire devrait être < 100 MB")
        
        print(f"✅ Utilisation mémoire vérifiée:")
        print(f"   U-Net 2D: {unet_2d_memory / (1024*1024):.2f} MB")
        print(f"   VoxNet 3D: {voxnet_3d_memory / (1024*1024):.2f} MB")
        print(f"   Total: {total_memory / (1024*1024):.2f} MB")
    
    def test_generate_synthetic_data_for_training_statistical_properties(self):
        """Test des propriétés statistiques des données générées"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        result = self.cleaner.generate_synthetic_data_for_training(num_samples, device_type)
        
        # Analyser les propriétés statistiques des tenseurs
        unet_2d_tensor = result['unet_2d']
        voxnet_3d_tensor = result['voxnet_3d']
        
        # Vérifier que les valeurs sont dans des plages raisonnables
        self.assertGreater(unet_2d_tensor.min().item(), -1000, "U-Net 2D: valeurs trop faibles")
        self.assertLess(unet_2d_tensor.max().item(), 1000, "U-Net 2D: valeurs trop élevées")
        
        self.assertGreater(voxnet_3d_tensor.min().item(), -1000, "VoxNet 3D: valeurs trop faibles")
        self.assertLess(voxnet_3d_tensor.max().item(), 1000, "VoxNet 3D: valeurs trop élevées")
        
        # Vérifier que les tenseurs ont une variance > 0
        self.assertGreater(unet_2d_tensor.var().item(), 0, "U-Net 2D devrait avoir une variance > 0")
        self.assertGreater(voxnet_3d_tensor.var().item(), 0, "VoxNet 3D devrait avoir une variance > 0")
        
        print(f"✅ Propriétés statistiques vérifiées:")
        print(f"   U-Net 2D - Mean: {unet_2d_tensor.mean().item():.4f}, Std: {unet_2d_tensor.std().item():.4f}")
        print(f"   VoxNet 3D - Mean: {voxnet_3d_tensor.mean().item():.4f}, Std: {voxnet_3d_tensor.std().item():.4f}")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
