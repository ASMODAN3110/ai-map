#!/usr/bin/env python3
"""
Tests unitaires pour la classe GeophysicalTrainer.

Teste l'intégration entre l'entraîneur et l'augmenteur de données.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.model.geophysical_trainer import (
    GeophysicalTrainer, 
    GeophysicalCNN2D, 
    GeophysicalCNN3D, 
    GeophysicalDataFrameNet
)
from src.utils.logger import logger


class TestGeophysicalTrainer(unittest.TestCase):
    """Tests pour la classe GeophysicalTrainer."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.augmenter = GeophysicalDataAugmenter(random_seed=42)
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Créer des données de test
        self.grids_2d = [
            np.random.randn(4, 16, 16) for _ in range(5)
        ]
        self.labels_2d = [0, 1, 0, 1, 0]
        
        self.volumes_3d = [
            np.random.randn(4, 8, 8, 8) for _ in range(3)
        ]
        self.labels_3d = [0, 1, 0]
        
        self.dataframes = [
            pd.DataFrame({
                'x': np.random.randn(10),
                'y': np.random.randn(10),
                'resistivity': np.random.randn(10),
                'chargeability': np.random.randn(10)
            }) for _ in range(4)
        ]
        self.labels_df = [0, 1, 0, 1]
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Tester l'initialisation de l'entraîneur."""
        self.assertIsNotNone(self.trainer.augmenter)
        self.assertEqual(self.trainer.device.type, "cpu")
        self.assertIn("train_loss", self.trainer.training_history)
        self.assertIn("val_loss", self.trainer.training_history)
        self.assertIn("train_accuracy", self.trainer.training_history)
        self.assertIn("val_accuracy", self.trainer.training_history)
        self.assertIn("epochs", self.trainer.training_history)
    
    def test_trainer_device_auto_detection(self):
        """Tester la détection automatique du device."""
        # Test avec device "auto" (devrait détecter CPU sur la plupart des systèmes)
        trainer_auto = GeophysicalTrainer(self.augmenter, device="auto")
        self.assertIsNotNone(trainer_auto.device)
        
        # Test avec device explicite
        trainer_cpu = GeophysicalTrainer(self.augmenter, device="cpu")
        self.assertEqual(trainer_cpu.device.type, "cpu")
    
    def test_prepare_data_2d(self):
        """Tester la préparation des données 2D avec augmentation."""
        train_loader, val_loader = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            augmentations=["flip_horizontal", "gaussian_noise"],
            num_augmentations=2,
            test_size=0.2
        )
        
        # Vérifier que les DataLoaders sont créés
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Vérifier que les données sont chargées
        for batch_idx, (data, target) in enumerate(train_loader):
            self.assertEqual(data.shape[0], target.shape[0])  # Batch size cohérent
            self.assertEqual(data.shape[1], 4)  # 4 canaux
            self.assertEqual(data.shape[2], 16)  # 16x16 grille
            break
    
    def test_prepare_data_3d(self):
        """Tester la préparation des données 3D avec augmentation."""
        train_loader, val_loader = self.trainer.prepare_data_3d(
            volumes=self.volumes_3d,
            labels=self.labels_3d,
            augmentations=["gaussian_noise"],
            num_augmentations=1,
            test_size=0.33
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Vérifier la forme des données 3D
        for batch_idx, (data, target) in enumerate(train_loader):
            self.assertEqual(data.shape[1], 4)  # 4 canaux
            self.assertEqual(data.shape[2], 8)  # 8x8x8 volume
            break
    
    def test_prepare_data_dataframe(self):
        """Tester la préparation des données DataFrame avec augmentation."""
        train_loader, val_loader = self.trainer.prepare_data_dataframe(
            dataframes=self.dataframes,
            labels=self.labels_df,
            augmentations=["gaussian_noise"],
            num_augmentations=2,
            test_size=0.25
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Vérifier la forme des données DataFrame
        for batch_idx, (data, target) in enumerate(train_loader):
            # Les données DataFrame ont la forme (batch_size, num_rows, num_features)
            # où num_features = 4 (x, y, resistivity, chargeability)
            self.assertEqual(data.shape[2], 4)  # 4 features (x, y, resistivity, chargeability)
            break
    
    def test_validation_errors(self):
        """Tester la gestion des erreurs de validation."""
        # Test avec des listes vides
        with self.assertRaises(ValueError):
            self.trainer.prepare_data_2d([], [], test_size=0.2)
        
        # Test avec des tailles différentes
        with self.assertRaises(ValueError):
            self.trainer.prepare_data_2d(self.grids_2d, [0, 1], test_size=0.2)
        
        # Test avec test_size invalide
        with self.assertRaises(ValueError):
            self.trainer.prepare_data_2d(self.grids_2d, self.labels_2d, test_size=1.5)
        
        # Test avec test_size invalide (négatif)
        with self.assertRaises(ValueError):
            self.trainer.prepare_data_2d(self.grids_2d, self.labels_2d, test_size=-0.1)
        
        # Test avec num_augmentations négatif
        with self.assertRaises(ValueError):
            self.trainer.prepare_data_2d(self.grids_2d, self.labels_2d, num_augmentations=-1)
    
    def test_augmentation_validation(self):
        """Tester la validation des techniques d'augmentation."""
        # Test avec des augmentations valides
        self.assertTrue(
            self.trainer.validate_augmentations_for_data_type(
                ["rotation", "flip_horizontal"], "2d"
            )
        )
        
        # Test avec des augmentations invalides
        self.assertFalse(
            self.trainer.validate_augmentations_for_data_type(
                ["invalid_technique"], "2d"
            )
        )
        
        # Test avec un type de données invalide
        with self.assertRaises(ValueError):
            self.trainer.validate_augmentations_for_data_type(
                ["rotation"], "invalid_type"
            )
    
    def test_augmentation_techniques_by_type(self):
        """Tester les techniques d'augmentation supportées par type de données."""
        # Test pour les données 2D
        valid_2d = self.trainer.validate_augmentations_for_data_type(
            ["rotation", "flip_horizontal", "gaussian_noise"], "2d"
        )
        self.assertTrue(valid_2d)
        
        # Test pour les données 3D
        valid_3d = self.trainer.validate_augmentations_for_data_type(
            ["rotation", "gaussian_noise"], "3d"
        )
        self.assertTrue(valid_3d)
        
        # Test pour les DataFrames
        valid_df = self.trainer.validate_augmentations_for_data_type(
            ["gaussian_noise", "value_variation"], "dataframe"
        )
        self.assertTrue(valid_df)
    
    def test_training_history_management(self):
        """Tester la gestion de l'historique d'entraînement."""
        # Vérifier l'état initial
        self.assertEqual(len(self.trainer.training_history["train_loss"]), 0)
        self.assertEqual(len(self.trainer.training_history["val_loss"]), 0)
        self.assertEqual(len(self.trainer.training_history["train_accuracy"]), 0)
        self.assertEqual(len(self.trainer.training_history["val_accuracy"]), 0)
        self.assertEqual(len(self.trainer.training_history["epochs"]), 0)
        
        # Simuler l'ajout d'historique
        self.trainer.training_history["train_loss"].append(0.5)
        self.trainer.training_history["val_loss"].append(0.6)
        self.trainer.training_history["train_accuracy"].append(75.0)
        self.trainer.training_history["val_accuracy"].append(70.0)
        self.trainer.training_history["epochs"].append(0)
        
        # Vérifier que l'historique est mis à jour
        self.assertEqual(len(self.trainer.training_history["train_loss"]), 1)
        self.assertEqual(self.trainer.training_history["train_loss"][0], 0.5)
        self.assertEqual(self.trainer.training_history["val_loss"][0], 0.6)
        self.assertEqual(self.trainer.training_history["train_accuracy"][0], 75.0)
        self.assertEqual(self.trainer.training_history["val_accuracy"][0], 70.0)
        self.assertEqual(self.trainer.training_history["epochs"][0], 0)
        
        # Tester la réinitialisation
        self.trainer.reset_training_history()
        self.assertEqual(len(self.trainer.training_history["train_loss"]), 0)
        self.assertEqual(len(self.trainer.training_history["val_loss"]), 0)
        self.assertEqual(len(self.trainer.training_history["train_accuracy"]), 0)
        self.assertEqual(len(self.trainer.training_history["val_accuracy"]), 0)
        self.assertEqual(len(self.trainer.training_history["epochs"]), 0)
    
    def test_augmentation_summary_integration(self):
        """Tester l'intégration avec l'augmenteur pour le résumé."""
        # Utiliser l'augmenteur pour créer des augmentations
        self.augmenter.augment_2d_grid(
            self.grids_2d[0], 
            ["flip_horizontal"], 
            2
        )
        
        # Récupérer le résumé via l'entraîneur
        summary = self.trainer.get_augmentation_summary()
        self.assertIsInstance(summary, dict)  # Le résumé est un dictionnaire
        self.assertIn("total_augmentations", summary)
        self.assertGreater(summary["total_augmentations"], 0)
    
    def test_data_preparation_with_different_augmentation_counts(self):
        """Tester la préparation des données avec différents nombres d'augmentations."""
        # Test sans augmentation
        train_loader, val_loader = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            num_augmentations=0,
            test_size=0.4  # Augmenter test_size pour éviter les erreurs de stratification
        )
        
        # Vérifier que les données originales sont préservées
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            total_samples += data.shape[0]
        
        # Avec test_size=0.4, on devrait avoir environ 3 échantillons d'entraînement
        self.assertGreaterEqual(total_samples, 3)
        
        # Test avec beaucoup d'augmentations
        train_loader_many, val_loader_many = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            num_augmentations=3,  # Réduire pour éviter les erreurs de stratification
            test_size=0.4
        )
        
        # Vérifier que plus de données sont générées
        total_samples_many = 0
        for batch_idx, (data, target) in enumerate(train_loader_many):
            total_samples_many += data.shape[0]
        
        self.assertGreater(total_samples_many, total_samples)
    
    def test_data_preparation_with_different_test_sizes(self):
        """Tester la préparation des données avec différentes tailles de test."""
        # Test avec test_size=0.3 (70% train, 30% val)
        train_loader_30, val_loader_30 = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            test_size=0.3,
            num_augmentations=1
        )
        
        # Test avec test_size=0.5 (50% train, 50% val)
        train_loader_50, val_loader_50 = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            test_size=0.5,
            num_augmentations=1
        )
        
        # Compter les échantillons dans chaque split
        train_samples_30 = sum(data.shape[0] for data, _ in train_loader_30)
        val_samples_30 = sum(data.shape[0] for data, _ in val_loader_30)
        
        train_samples_50 = sum(data.shape[0] for data, _ in train_loader_50)
        val_samples_50 = sum(data.shape[0] for data, _ in val_loader_50)
        
        # Avec test_size=0.3, on devrait avoir plus d'échantillons d'entraînement
        self.assertGreater(train_samples_30, val_samples_30)
        
        # Avec test_size=0.5, la différence devrait être plus petite
        self.assertLess(abs(train_samples_50 - val_samples_50), abs(train_samples_30 - val_samples_30))
    
    def test_batch_size_consistency(self):
        """Tester la cohérence des tailles de batch."""
        # Test 2D
        train_loader_2d, _ = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            num_augmentations=1
        )
        
        # Test DataFrame
        train_loader_df, _ = self.trainer.prepare_data_dataframe(
            dataframes=self.dataframes,
            labels=self.labels_df,
            num_augmentations=1
        )
        
        # Vérifier que les batch sizes sont cohérents dans chaque loader
        for loader in [train_loader_2d, train_loader_df]:
            batch_sizes = [data.shape[0] for data, _ in loader]
            if len(batch_sizes) > 1:
                # Tous les batches (sauf le dernier) devraient avoir la même taille
                self.assertEqual(len(set(batch_sizes[:-1])), 1)
    
    def test_label_preservation(self):
        """Tester que les labels sont correctement préservés lors de l'augmentation."""
        # Créer des labels uniques pour chaque grille
        unique_labels = [i for i in range(len(self.grids_2d))]
        
        train_loader, val_loader = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=unique_labels,
            num_augmentations=2,
            test_size=0.4  # Augmenter test_size pour éviter les erreurs de stratification
        )
        
        # Collecter tous les labels d'entraînement
        all_train_labels = []
        for data, target in train_loader:
            all_train_labels.extend(target.tolist())
        
        # Vérifier que tous les labels originaux sont présents
        for original_label in unique_labels:
            self.assertIn(original_label, all_train_labels)
        
        # Vérifier que les labels sont dans la bonne plage
        self.assertTrue(all(0 <= label < len(unique_labels) for label in all_train_labels))
    
    def test_augmentation_technique_combinations(self):
        """Tester les combinaisons de techniques d'augmentation."""
        # Test avec plusieurs techniques
        train_loader, val_loader = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            augmentations=["rotation", "flip_horizontal", "gaussian_noise"],
            num_augmentations=3,
            test_size=0.2
        )
        
        # Vérifier que les données sont bien générées
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test avec une seule technique
        train_loader_single, val_loader_single = self.trainer.prepare_data_2d(
            grids=self.grids_2d,
            labels=self.labels_2d,
            augmentations=["gaussian_noise"],
            num_augmentations=2,
            test_size=0.2
        )
        
        self.assertIsNotNone(train_loader_single)
        self.assertIsNotNone(val_loader_single)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
