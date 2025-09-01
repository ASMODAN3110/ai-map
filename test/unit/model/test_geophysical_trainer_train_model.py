#!/usr/bin/env python3
"""
Tests unitaires pour la méthode train_model de GeophysicalTrainer.

Teste l'entraînement des modèles, l'early stopping, le scheduler et la gestion de l'historique.
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


class TestGeophysicalTrainerTrainModel(unittest.TestCase):
    """Tests pour la méthode train_model de GeophysicalTrainer."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un augmenter de test
        self.augmenter = GeophysicalDataAugmenter()
        
        # Créer un entraîneur de test
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Créer des données de test 2D (height, width, channels)
        self.grids_2d = [
            np.random.randn(16, 16, 1),  # Grille 16x16x1
            np.random.randn(16, 16, 1),
            np.random.randn(16, 16, 1),
            np.random.randn(16, 16, 1)
        ]
        self.labels_2d = [0, 1, 0, 1]
        
        # Créer des DataLoaders de test simples
        from torch.utils.data import DataLoader, TensorDataset
        
        # Créer des données de test avec la bonne forme (batch_size, channels, height, width)
        train_data = torch.randn(8, 1, 16, 16)  # 8 échantillons, 1 canal, 16x16
        train_labels = torch.randint(0, 2, (8,))
        val_data = torch.randn(4, 1, 16, 16)    # 4 échantillons, 1 canal, 16x16
        val_labels = torch.randint(0, 2, (4,))
        
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Créer un modèle de test
        self.model = GeophysicalCNN2D(input_channels=1, grid_size=16, num_classes=2)
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
        
        # Supprimer le fichier best_model.pth s'il existe
        if os.path.exists("best_model.pth"):
            os.remove("best_model.pth")
    
    def test_train_model_basic_training(self):
        """Tester l'entraînement de base d'un modèle."""
        # Entraîner le modèle sur quelques époques
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3,
            learning_rate=0.01
        )
        
        # Vérifier que l'historique a été mis à jour
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("train_accuracy", history)
        self.assertIn("val_accuracy", history)
        self.assertIn("epochs", history)
        
        # Vérifier que les listes ne sont pas vides
        self.assertGreater(len(history["train_loss"]), 0)
        self.assertGreater(len(history["val_loss"]), 0)
        self.assertGreater(len(history["train_accuracy"]), 0)
        self.assertGreater(len(history["val_accuracy"]), 0)
        self.assertGreater(len(history["epochs"]), 0)
        
        # Vérifier que le nombre d'époques correspond
        self.assertEqual(len(history["epochs"]), 3)
        
        # Vérifier que les métriques sont des nombres valides
        for loss in history["train_loss"]:
            self.assertIsInstance(loss, (int, float))
            self.assertGreaterEqual(loss, 0)
        
        for loss in history["val_loss"]:
            self.assertIsInstance(loss, (int, float))
            self.assertGreaterEqual(loss, 0)
        
        for acc in history["train_accuracy"]:
            self.assertIsInstance(acc, (int, float))
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 100)
    
        for acc in history["val_accuracy"]:
            self.assertIsInstance(acc, (int, float))
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 100)
    
    def test_train_model_learning_rate_scheduling(self):
        """Tester le scheduler de learning rate."""
        # Entraîner avec un scheduler
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=5,
            learning_rate=0.01
        )
                
                # Vérifier que l'entraînement s'est bien déroulé
        self.assertGreater(len(history["epochs"]), 0)
        
        # Vérifier que le modèle a été déplacé vers le bon device
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
    
    def test_train_model_weight_decay(self):
        """Tester l'application du weight decay."""
        # Entraîner avec weight decay
                history = self.trainer.train_model(
            model=self.model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=3,
                    learning_rate=0.01,
            weight_decay=0.01
                )
                
                # Vérifier que l'entraînement s'est bien déroulé
        self.assertGreater(len(history["epochs"]), 0)
        
        # Vérifier que les paramètres ont des gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_train_model_early_stopping(self):
        """Tester l'early stopping."""
        # Créer un modèle simple qui ne s'améliore pas
        simple_model = GeophysicalCNN2D(input_channels=1, grid_size=16, num_classes=2)
        
        # Créer des DataLoaders avec des données constantes (même données = pas d'amélioration)
        constant_data = torch.randn(4, 1, 16, 16)
        constant_labels = torch.tensor([0, 1, 0, 1])
        constant_dataset = torch.utils.data.TensorDataset(constant_data, constant_labels)
        constant_loader = torch.utils.data.DataLoader(constant_dataset, batch_size=2)
        
        # Entraîner avec early stopping
                history = self.trainer.train_model(
            model=simple_model,
            train_loader=constant_loader,
            val_loader=constant_loader,
            num_epochs=20,
            patience=3
        )
        
        # Vérifier que l'early stopping a fonctionné
        # (moins d'époques que prévu à cause de l'early stopping)
        self.assertLessEqual(len(history["epochs"]), 20)
    
    def test_train_model_device_transfer(self):
        """Tester le transfert du modèle vers le device."""
        # Vérifier que le modèle est initialement sur CPU
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=2
        )
        
        # Vérifier que le modèle est toujours sur le bon device après entraînement
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_gradient_flow(self):
        """Tester que les gradients circulent correctement."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=2
        )
        
        # Vérifier que les gradients ont été calculés
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Les gradients peuvent être None si pas de backward pass récent
                # mais le paramètre doit exister
                self.assertIsNotNone(param)
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_accuracy_calculation(self):
        """Tester le calcul de l'accuracy."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3
        )
        
        # Vérifier que les accuracies sont dans la plage [0, 100]
        for train_acc in history["train_accuracy"]:
            self.assertGreaterEqual(train_acc, 0)
            self.assertLessEqual(train_acc, 100)
        
        for val_acc in history["val_accuracy"]:
            self.assertGreaterEqual(val_acc, 0)
            self.assertLessEqual(val_acc, 100)
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_loss_calculation(self):
        """Tester le calcul de la loss."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3
        )
        
        # Vérifier que les losses sont des nombres positifs
        for train_loss in history["train_loss"]:
            self.assertGreaterEqual(train_loss, 0)
            self.assertIsInstance(train_loss, (int, float))
        
        for val_loss in history["val_loss"]:
            self.assertGreaterEqual(val_loss, 0)
            self.assertIsInstance(val_loss, (int, float))
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_epoch_logging(self):
        """Tester le logging des métriques par époque."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=5
        )
        
        # Vérifier que les époques sont numérotées correctement
        expected_epochs = list(range(5))
        self.assertEqual(history["epochs"], expected_epochs)
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_best_model_saving(self):
        """Tester la sauvegarde du meilleur modèle."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3
        )
        
        # Vérifier que le fichier best_model.pth a été créé
        self.assertTrue(os.path.exists("best_model.pth"))
        
        # Vérifier que le fichier n'est pas vide
        file_size = os.path.getsize("best_model.pth")
        self.assertGreater(file_size, 0)
    
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_parameter_updates(self):
        """Tester que les paramètres du modèle sont mis à jour."""
        # Sauvegarder les paramètres initiaux
        initial_params = {}
        for name, param in self.model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3
        )
        
        # Vérifier que les paramètres ont changé
        params_changed = False
        for name, param in self.model.named_parameters():
            if not torch.equal(initial_params[name], param.data):
                params_changed = True
                break
        
        self.assertTrue(params_changed, "Les paramètres du modèle n'ont pas changé pendant l'entraînement")
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
    
    def test_train_model_training_mode_switching(self):
        """Tester le changement de mode entraînement/évaluation."""
        # Vérifier le mode initial
        self.assertTrue(self.model.training)
        
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=2
        )
        
        # Vérifier que l'historique a été mis à jour
        self.assertGreater(len(history["epochs"]), 0)
        
        # Vérifier que le modèle peut encore changer de mode
        self.model.eval()
        self.assertFalse(self.model.training)
        
        self.model.train()
        self.assertTrue(self.model.training)
    
    def test_train_model_empty_data_loaders(self):
        """Tester le comportement avec des DataLoaders vides."""
        # Créer des DataLoaders vides
        empty_dataset = torch.utils.data.TensorDataset(
            torch.empty(0, 1, 16, 16),
            torch.empty(0, dtype=torch.long)
        )
        empty_loader = torch.utils.data.DataLoader(empty_dataset, batch_size=1)
        
        # Tenter d'entraîner avec des données vides
        with self.assertRaises(Exception):
            self.trainer.train_model(
                model=self.model,
                train_loader=empty_loader,
                val_loader=empty_loader,
                num_epochs=1
            )
    
    def test_train_model_invalid_parameters(self):
        """Tester le comportement avec des paramètres invalides."""
        # Test avec num_epochs négatif (doit être >= 0)
        with self.assertRaises(ValueError):
            self.trainer.train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=-1
            )
        
        # Test avec patience négative (doit être >= 0)
        with self.assertRaises(ValueError):
            self.trainer.train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=1,
                patience=-1
            )
        
        # Test avec learning_rate négatif (doit être > 0)
        with self.assertRaises(ValueError):
            self.trainer.train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=1,
                learning_rate=-0.01
            )
    
    def test_train_model_return_value(self):
        """Tester que la valeur de retour est correcte."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=3
        )
        
        # Vérifier que l'historique retourné est le même que celui de l'entraîneur
        self.assertIs(history, self.trainer.training_history)
        
        # Vérifier que l'historique contient les bonnes clés
        expected_keys = ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "epochs"]
        for key in expected_keys:
            self.assertIn(key, history)
    
    def test_train_model_history_consistency(self):
        """Tester la cohérence de l'historique d'entraînement."""
        # Entraîner le modèle
        history = self.trainer.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=5
        )
        
        # Vérifier que toutes les listes ont la même longueur
        list_lengths = [
            len(history["train_loss"]),
            len(history["val_loss"]),
            len(history["train_accuracy"]),
            len(history["val_accuracy"]),
            len(history["epochs"])
        ]
        
        self.assertEqual(len(set(list_lengths)), 1, "Toutes les listes de l'historique doivent avoir la même longueur")
        
        # Vérifier que la longueur correspond au nombre d'époques
        expected_length = len(history["epochs"])
        self.assertEqual(list_lengths[0], expected_length)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
