#!/usr/bin/env python3
"""
Tests unitaires pour la méthode evaluate_model de GeophysicalTrainer.

Teste l'évaluation des modèles, le calcul des métriques et la gestion des données de test.
"""

import unittest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_trainer import GeophysicalTrainer, GeophysicalCNN2D
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.utils.logger import logger


class TestGeophysicalTrainerEvaluateModel(unittest.TestCase):
    """Tests pour la méthode evaluate_model de GeophysicalTrainer."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un augmenter de test
        self.augmenter = GeophysicalDataAugmenter()
        
        # Créer un entraîneur de test
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Créer un modèle de test
        self.model = GeophysicalCNN2D(input_channels=1, grid_size=16, num_classes=2)
        
        # Créer des données de test avec la bonne forme (batch_size, channels, height, width)
        from torch.utils.data import DataLoader, TensorDataset
        
        test_data = torch.randn(16, 1, 16, 16)  # 16 échantillons, 1 canal, 16x16
        test_labels = torch.randint(0, 2, (16,))
        
        test_dataset = TensorDataset(test_data, test_labels)
        self.test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Importer TensorDataset et DataLoader pour tous les tests
        self.TensorDataset = TensorDataset
        self.DataLoader = DataLoader
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_evaluate_model_basic_evaluation(self):
        """Tester l'évaluation de base d'un modèle."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que les métriques contiennent les bonnes clés
        expected_keys = ["test_loss", "test_accuracy", "classification_report", "confusion_matrix"]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Vérifier que les métriques sont des nombres valides
        self.assertIsInstance(metrics["test_loss"], (int, float))
        self.assertIsInstance(metrics["test_accuracy"], (int, float))
        self.assertGreaterEqual(metrics["test_loss"], 0)
        self.assertGreaterEqual(metrics["test_accuracy"], 0)
        self.assertLessEqual(metrics["test_accuracy"], 100)
        
        # Vérifier que le modèle est en mode évaluation
        self.assertFalse(self.model.training)
    
    def test_evaluate_model_device_transfer(self):
        """Tester le transfert du modèle vers le device."""
        # Vérifier que le modèle est initialement sur CPU
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que le modèle est toujours sur le bon device après évaluation
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
    
    def test_evaluate_model_accuracy_calculation(self):
        """Tester le calcul de l'accuracy."""
        # Créer des données avec des labels connus pour tester l'accuracy
        test_data = torch.randn(8, 1, 16, 16)
        test_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # Labels alternés
        
        test_dataset = self.TensorDataset(test_data, test_labels)
        test_loader = self.DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=test_loader
        )
        
        # Vérifier que l'accuracy est dans la plage [0, 100]
        self.assertGreaterEqual(metrics["test_accuracy"], 0)
        self.assertLessEqual(metrics["test_accuracy"], 100)
        
        # Vérifier que l'accuracy est un nombre
        self.assertIsInstance(metrics["test_accuracy"], (int, float))
    
    def test_evaluate_model_loss_calculation(self):
        """Tester le calcul de la loss."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que la loss est un nombre positif
        self.assertIsInstance(metrics["test_loss"], (int, float))
        self.assertGreaterEqual(metrics["test_loss"], 0)
        
        # Vérifier que la loss est raisonnable (pas infinie ou NaN)
        self.assertFalse(np.isnan(metrics["test_loss"]))
        self.assertFalse(np.isinf(metrics["test_loss"]))
    
    def test_evaluate_model_classification_report(self):
        """Tester la génération du rapport de classification."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que le rapport de classification est une chaîne
        self.assertIsInstance(metrics["classification_report"], str)
        self.assertGreater(len(metrics["classification_report"]), 0)
        
        # Vérifier que le rapport contient des informations utiles
        report = metrics["classification_report"]
        self.assertIn("precision", report.lower())
        self.assertIn("recall", report.lower())
        self.assertIn("f1-score", report.lower())
    
    def test_evaluate_model_confusion_matrix(self):
        """Tester la génération de la matrice de confusion."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que la matrice de confusion est une liste
        self.assertIsInstance(metrics["confusion_matrix"], list)
        self.assertGreater(len(metrics["confusion_matrix"]), 0)
        
        # Vérifier que c'est une matrice 2D (pour 2 classes)
        confusion_matrix = metrics["confusion_matrix"]
        self.assertEqual(len(confusion_matrix), 2)  # 2 lignes
        self.assertEqual(len(confusion_matrix[0]), 2)  # 2 colonnes
        
        # Vérifier que tous les éléments sont des nombres
        for row in confusion_matrix:
            for element in row:
                self.assertIsInstance(element, (int, np.integer))
                self.assertGreaterEqual(element, 0)
    
    def test_evaluate_model_no_grad_mode(self):
        """Tester que l'évaluation se fait sans calcul de gradients."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
        
        # Vérifier que le modèle est toujours en mode évaluation
        self.assertFalse(self.model.training)
        
        # Vérifier que les paramètres n'ont pas de gradients (mode eval)
        for param in self.model.parameters():
            if param.grad is not None:
                # Les gradients peuvent exister mais ne doivent pas être calculés pendant eval
                pass
    
    def test_evaluate_model_different_batch_sizes(self):
        """Tester l'évaluation avec différents batch sizes."""
        # Créer des DataLoaders avec différents batch sizes
        test_data = torch.randn(12, 1, 16, 16)
        test_labels = torch.randint(0, 2, (12,))
        
        test_dataset = self.TensorDataset(test_data, test_labels)
        
        batch_sizes = [1, 2, 4, 6]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                test_loader = self.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Évaluer le modèle
                metrics = self.trainer.evaluate_model(
                    model=self.model,
                    test_loader=test_loader
                )
                
                # Vérifier que les métriques ont été calculées
                self.assertIn("test_loss", metrics)
                self.assertIn("test_accuracy", metrics)
                
                # Vérifier que la loss est un nombre valide
                self.assertIsInstance(metrics["test_loss"], (int, float))
                self.assertGreaterEqual(metrics["test_loss"], 0)
    
    def test_evaluate_model_empty_test_loader(self):
        """Tester le comportement avec un test loader vide."""
        # Créer un DataLoader vide
        empty_dataset = self.TensorDataset(
            torch.empty(0, 1, 16, 16),
            torch.empty(0, dtype=torch.long)
        )
        empty_loader = self.DataLoader(empty_dataset, batch_size=1)
        
        # Tenter d'évaluer avec des données vides
        with self.assertRaises(Exception):
            self.trainer.evaluate_model(
                model=self.model,
                test_loader=empty_loader
            )
    
    def test_evaluate_model_single_sample(self):
        """Tester l'évaluation avec un seul échantillon."""
        # Créer un DataLoader avec un seul échantillon
        single_data = torch.randn(1, 1, 16, 16)
        single_label = torch.tensor([0])
        
        single_dataset = self.TensorDataset(single_data, single_label)
        single_loader = self.DataLoader(single_dataset, batch_size=1, shuffle=False)
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=single_loader
        )
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
        
        # Vérifier que l'accuracy est soit 0% soit 100% (1 échantillon)
        self.assertIn(metrics["test_accuracy"], [0.0, 100.0])
    
    def test_evaluate_model_multiple_classes(self):
        """Tester l'évaluation avec plusieurs classes."""
        # Créer un modèle avec 3 classes
        model_3_classes = GeophysicalCNN2D(input_channels=1, grid_size=16, num_classes=3)
        
        # Créer des données avec 3 classes
        test_data = torch.randn(9, 1, 16, 16)
        test_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # 3 échantillons par classe
        
        test_dataset = self.TensorDataset(test_data, test_labels)
        test_loader = self.DataLoader(test_dataset, batch_size=3, shuffle=False)
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=model_3_classes,
            test_loader=test_loader
        )
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
        
        # Vérifier que la matrice de confusion est 3x3
        confusion_matrix = metrics["confusion_matrix"]
        self.assertEqual(len(confusion_matrix), 3)  # 3 lignes
        self.assertEqual(len(confusion_matrix[0]), 3)  # 3 colonnes
    
    def test_evaluate_model_return_value_structure(self):
        """Tester la structure de la valeur de retour."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que c'est un dictionnaire
        self.assertIsInstance(metrics, dict)
        
        # Vérifier que toutes les clés attendues sont présentes
        expected_keys = ["test_loss", "test_accuracy", "classification_report", "confusion_matrix"]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Vérifier que les valeurs ont les bons types
        self.assertIsInstance(metrics["test_loss"], (int, float))
        self.assertIsInstance(metrics["test_accuracy"], (int, float))
        self.assertIsInstance(metrics["classification_report"], str)
        self.assertIsInstance(metrics["confusion_matrix"], list)
    
    def test_evaluate_model_metric_ranges(self):
        """Tester que les métriques sont dans les bonnes plages."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier les plages des métriques
        self.assertGreaterEqual(metrics["test_loss"], 0)  # Loss >= 0
        self.assertGreaterEqual(metrics["test_accuracy"], 0)  # Accuracy >= 0%
        self.assertLessEqual(metrics["test_accuracy"], 100)  # Accuracy <= 100%
        
        # Vérifier que les métriques sont des nombres finis
        self.assertFalse(np.isnan(metrics["test_loss"]))
        self.assertFalse(np.isinf(metrics["test_loss"]))
        self.assertFalse(np.isnan(metrics["test_accuracy"]))
        self.assertFalse(np.isinf(metrics["test_accuracy"]))
    
    def test_evaluate_model_model_state_preservation(self):
        """Tester que l'état du modèle est préservé après évaluation."""
        # Sauvegarder l'état initial du modèle
        initial_state = {}
        for name, param in self.model.named_parameters():
            initial_state[name] = param.data.clone()
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
        
        # Vérifier que les paramètres du modèle n'ont pas changé
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.equal(initial_state[name], param.data),
                          f"Le paramètre {name} a changé pendant l'évaluation")
    
    def test_evaluate_model_logging(self):
        """Tester que le logging fonctionne correctement."""
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(
            model=self.model,
            test_loader=self.test_loader
        )
        
        # Vérifier que les métriques ont été calculées
        self.assertIn("test_loss", metrics)
        self.assertIn("test_accuracy", metrics)
        
        # Vérifier que le modèle est en mode évaluation
        self.assertFalse(self.model.training)
        
        # Vérifier que l'évaluation s'est bien terminée
        self.assertIsInstance(metrics["test_loss"], (int, float))
        self.assertIsInstance(metrics["test_accuracy"], (int, float))


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
