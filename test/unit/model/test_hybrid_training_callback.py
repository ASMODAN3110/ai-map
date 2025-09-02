#!/usr/bin/env python3
"""
Tests unitaires pour HybridTrainingCallback
==========================================

Ce module teste toutes les méthodes de la classe HybridTrainingCallback
pour les callbacks d'entraînement de modèles hybrides.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.optim as optim
from datetime import datetime

# Ajouter le chemin du projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.model.geophysical_image_trainer import HybridTrainingCallback


class TestHybridTrainingCallback(unittest.TestCase):
    """Tests pour la classe HybridTrainingCallback."""
    
    def setUp(self):
        """Initialiser les tests avec des données d'entraînement réalistes."""
        # Créer un dossier temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = os.path.join(self.temp_dir, "test_models")
        
        # Créer le callback
        self.callback = HybridTrainingCallback(save_dir=self.save_dir)
        
        # Créer un vrai modèle PyTorch simple pour les tests
        self.real_model = self._create_real_model()
        
        # Créer un vrai optimiseur PyTorch
        self.real_optimizer = torch.optim.Adam(self.real_model.parameters(), lr=0.001)
        
        # Données d'entraînement réalistes (simulation d'un vrai entraînement)
        self.real_training_metrics = {
            'epoch_0': {'train_loss': 2.5, 'val_loss': 2.8, 'train_acc': 0.45, 'val_acc': 0.42},
            'epoch_1': {'train_loss': 2.0, 'val_loss': 2.3, 'train_acc': 0.55, 'val_acc': 0.52},
            'epoch_2': {'train_loss': 1.8, 'val_loss': 2.1, 'train_acc': 0.65, 'val_acc': 0.62},
            'epoch_3': {'train_loss': 1.6, 'val_loss': 2.0, 'train_acc': 0.72, 'val_acc': 0.68},
            'epoch_4': {'train_loss': 1.5, 'val_loss': 1.9, 'train_acc': 0.78, 'val_acc': 0.75}
        }
    
    def _create_real_model(self):
        """Créer un vrai modèle PyTorch pour les tests."""
        class SimpleHybridModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes = 3
                self.image_model = 'resnet18'
                self.fusion_method = 'concatenation'
                
                # Couches simples pour simuler un modèle hybride
                self.image_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten()
                )
                
                self.geophysical_encoder = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2)
                )
                
                self.fusion_layer = torch.nn.Sequential(
                    torch.nn.Linear(16 + 32, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64, self.num_classes)
                )
            
            def forward(self, image_input, geophysical_input):
                image_features = self.image_encoder(image_input)
                geo_features = self.geophysical_encoder(geophysical_input)
                combined = torch.cat([image_features, geo_features], dim=1)
                return self.fusion_layer(combined)
        
        return SimpleHybridModel()
    
    def tearDown(self):
        """Nettoyer après les tests."""
        # Supprimer le dossier temporaire
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_default_save_dir(self):
        """Tester l'initialisation avec le dossier de sauvegarde par défaut."""
        callback = HybridTrainingCallback()
        
        # Vérifier que le dossier par défaut est créé
        self.assertEqual(callback.save_dir, "artifacts/models/hybrid")
        self.assertTrue(os.path.exists("artifacts/models/hybrid"))
        
        # Nettoyer le dossier créé
        if os.path.exists("artifacts/models/hybrid"):
            shutil.rmtree("artifacts/models/hybrid")
    
    def test_init_custom_save_dir(self):
        """Tester l'initialisation avec un dossier personnalisé."""
        custom_dir = os.path.join(self.temp_dir, "custom_models")
        callback = HybridTrainingCallback(save_dir=custom_dir)
        
        # Vérifier que le dossier personnalisé est créé
        self.assertEqual(callback.save_dir, custom_dir)
        self.assertTrue(os.path.exists(custom_dir))
    
    def test_init_attributes(self):
        """Tester l'initialisation des attributs."""
        # Vérifier les attributs initiaux
        self.assertEqual(self.callback.best_val_loss, float('inf'))
        self.assertEqual(self.callback.best_val_acc, 0.0)
        self.assertEqual(self.callback.patience_counter, 0)
        
        # Vérifier la structure de l'historique
        expected_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'learning_rate']
        for key in expected_keys:
            self.assertIn(key, self.callback.training_history)
            self.assertEqual(len(self.callback.training_history[key]), 0)
    
    def test_on_epoch_end_first_epoch(self):
        """Tester le callback sur le premier epoch avec un vrai modèle."""
        epoch = 0
        train_loss = 2.5
        val_loss = 2.8
        train_acc = 0.45
        val_acc = 0.42
        
        # Appeler le callback avec le vrai modèle
        epoch_info = self.callback.on_epoch_end(
            epoch, self.real_model, train_loss, val_loss, train_acc, val_acc, self.real_optimizer
        )
        
        # Vérifier que l'historique est mis à jour
        self.assertEqual(len(self.callback.training_history['train_loss']), 1)
        self.assertEqual(len(self.callback.training_history['val_loss']), 1)
        self.assertEqual(len(self.callback.training_history['train_acc']), 1)
        self.assertEqual(len(self.callback.training_history['val_acc']), 1)
        self.assertEqual(len(self.callback.training_history['learning_rate']), 1)
        
        # Vérifier les valeurs
        self.assertEqual(self.callback.training_history['train_loss'][0], train_loss)
        self.assertEqual(self.callback.training_history['val_loss'][0], val_loss)
        self.assertEqual(self.callback.training_history['train_acc'][0], train_acc)
        self.assertEqual(self.callback.training_history['val_acc'][0], val_acc)
        self.assertEqual(self.callback.training_history['learning_rate'][0], 0.001)
        
        # Vérifier que c'est le meilleur modèle (premier epoch)
        self.assertEqual(self.callback.best_val_loss, val_loss)
        self.assertEqual(self.callback.best_val_acc, val_acc)
        self.assertEqual(self.callback.patience_counter, 0)
        
        # Vérifier les informations retournées
        self.assertEqual(epoch_info['epoch'], epoch)
        self.assertEqual(epoch_info['train_loss'], train_loss)
        self.assertEqual(epoch_info['val_loss'], val_loss)
        self.assertEqual(epoch_info['train_acc'], train_acc)
        self.assertEqual(epoch_info['val_acc'], val_acc)
        self.assertTrue(epoch_info['is_best_loss'])
        self.assertTrue(epoch_info['is_best_acc'])
        self.assertEqual(epoch_info['patience_counter'], 0)
    
    def test_on_epoch_end_improving_model(self):
        """Tester le callback avec un modèle qui s'améliore."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Deuxième epoch avec amélioration
        epoch_info = self.callback.on_epoch_end(
            1, self.real_model, 2.0, 2.3, 0.55, 0.52, self.real_optimizer
        )
        
        # Vérifier que les meilleurs scores sont mis à jour
        self.assertEqual(self.callback.best_val_loss, 2.3)
        self.assertEqual(self.callback.best_val_acc, 0.52)
        self.assertEqual(self.callback.patience_counter, 0)
        
        # Vérifier les informations retournées
        self.assertTrue(epoch_info['is_best_loss'])
        self.assertTrue(epoch_info['is_best_acc'])
        self.assertEqual(epoch_info['patience_counter'], 0)
    
    def test_on_epoch_end_worse_model(self):
        """Tester le callback avec un modèle qui se dégrade."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Deuxième epoch avec dégradation
        epoch_info = self.callback.on_epoch_end(
            1, self.real_model, 2.8, 3.0, 0.40, 0.38, self.real_optimizer
        )
        
        # Vérifier que les meilleurs scores ne changent pas
        self.assertEqual(self.callback.best_val_loss, 2.8)
        self.assertEqual(self.callback.best_val_acc, 0.42)
        self.assertEqual(self.callback.patience_counter, 1)
        
        # Vérifier les informations retournées
        self.assertFalse(epoch_info['is_best_loss'])
        self.assertFalse(epoch_info['is_best_acc'])
        self.assertEqual(epoch_info['patience_counter'], 1)
    
    def test_on_epoch_end_mixed_improvement(self):
        """Tester le callback avec amélioration partielle."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Deuxième epoch : meilleur loss mais pire accuracy
        epoch_info = self.callback.on_epoch_end(
            1, self.real_model, 2.0, 2.3, 0.40, 0.38, self.real_optimizer
        )
        
        # Vérifier que seul le loss s'améliore
        self.assertEqual(self.callback.best_val_loss, 2.3)
        self.assertEqual(self.callback.best_val_acc, 0.42)  # Ne change pas
        self.assertEqual(self.callback.patience_counter, 0)  # Reset car meilleur loss
        
        # Vérifier les informations retournées
        self.assertTrue(epoch_info['is_best_loss'])
        self.assertFalse(epoch_info['is_best_acc'])
        self.assertEqual(epoch_info['patience_counter'], 0)
    
    def test_save_model_creation(self):
        """Tester la sauvegarde d'un modèle."""
        epoch = 5
        val_loss = 2.1
        val_acc = 0.75
        
        # Sauvegarder le modèle
        filepath = os.path.join(self.save_dir, "test_model.pth")
        self.callback._save_model(self.real_model, filepath, epoch, val_loss, val_acc)
        
        # Vérifier que le fichier est créé
        self.assertTrue(os.path.exists(filepath))
        
        # Charger et vérifier le contenu
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.assertEqual(checkpoint['epoch'], epoch)
        self.assertEqual(checkpoint['val_loss'], val_loss)
        self.assertEqual(checkpoint['val_acc'], val_acc)
        self.assertEqual(checkpoint['model_config']['num_classes'], 3)
        self.assertEqual(checkpoint['model_config']['image_model'], 'resnet18')
        self.assertEqual(checkpoint['model_config']['fusion_method'], 'concatenation')
        self.assertIn('timestamp', checkpoint)
        self.assertIn('training_history', checkpoint)
    
    def test_save_model_with_training_history(self):
        """Tester la sauvegarde avec l'historique d'entraînement."""
        # Ajouter quelques epochs à l'historique
        self.callback.training_history['train_loss'] = [2.5, 2.0, 1.8]
        self.callback.training_history['val_loss'] = [2.8, 2.3, 2.1]
        self.callback.training_history['train_acc'] = [0.45, 0.55, 0.65]
        self.callback.training_history['val_acc'] = [0.42, 0.52, 0.62]
        self.callback.training_history['learning_rate'] = [0.001, 0.001, 0.0005]
        
        # Sauvegarder le modèle
        filepath = os.path.join(self.save_dir, "test_model_with_history.pth")
        self.callback._save_model(self.real_model, filepath, 3, 2.1, 0.62)
        
        # Charger et vérifier l'historique
        checkpoint = torch.load(filepath, map_location='cpu')
        saved_history = checkpoint['training_history']
        
        self.assertEqual(saved_history['train_loss'], [2.5, 2.0, 1.8])
        self.assertEqual(saved_history['val_loss'], [2.8, 2.3, 2.1])
        self.assertEqual(saved_history['train_acc'], [0.45, 0.55, 0.65])
        self.assertEqual(saved_history['val_acc'], [0.42, 0.52, 0.62])
        self.assertEqual(saved_history['learning_rate'], [0.001, 0.001, 0.0005])
    
    def test_get_training_summary_empty(self):
        """Tester l'obtention du résumé d'entraînement vide."""
        summary = self.callback.get_training_summary()
        
        # Vérifier la structure
        self.assertIn('best_val_loss', summary)
        self.assertIn('best_val_acc', summary)
        self.assertIn('total_epochs', summary)
        self.assertIn('training_history', summary)
        
        # Vérifier les valeurs par défaut
        self.assertEqual(summary['best_val_loss'], float('inf'))
        self.assertEqual(summary['best_val_acc'], 0.0)
        self.assertEqual(summary['total_epochs'], 0)
        self.assertEqual(summary['training_history'], self.callback.training_history)
    
    def test_get_training_summary_with_data(self):
        """Tester l'obtention du résumé d'entraînement avec des données."""
        # Ajouter des epochs à l'historique
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        self.callback.on_epoch_end(
            1, self.real_model, 2.0, 2.3, 0.55, 0.52, self.real_optimizer
        )
        
        # Obtenir le résumé
        summary = self.callback.get_training_summary()
        
        # Vérifier les valeurs
        self.assertEqual(summary['best_val_loss'], 2.3)
        self.assertEqual(summary['best_val_acc'], 0.52)
        self.assertEqual(summary['total_epochs'], 2)
        self.assertEqual(summary['training_history'], self.callback.training_history)
    
    def test_early_stopping_patience(self):
        """Tester le mécanisme d'early stopping avec patience."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Epochs suivants sans amélioration
        for epoch in range(1, 6):
            epoch_info = self.callback.on_epoch_end(
                epoch, self.real_model, 2.8, 3.0, 0.40, 0.38, self.real_optimizer
            )
            
            # Vérifier que la patience augmente
            self.assertEqual(epoch_info['patience_counter'], epoch)
            self.assertEqual(self.callback.patience_counter, epoch)
    
    def test_early_stopping_reset(self):
        """Tester la réinitialisation de la patience lors d'amélioration."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Epochs sans amélioration
        for epoch in range(1, 4):
            self.callback.on_epoch_end(
                epoch, self.real_model, 2.8, 3.0, 0.40, 0.38, self.real_optimizer
            )
        
        # Vérifier que la patience est à 3
        self.assertEqual(self.callback.patience_counter, 3)
        
        # Epoch avec amélioration
        epoch_info = self.callback.on_epoch_end(
            4, self.real_model, 2.0, 2.3, 0.55, 0.52, self.real_optimizer
        )
        
        # Vérifier que la patience est réinitialisée
        self.assertEqual(self.callback.patience_counter, 0)
        self.assertEqual(epoch_info['patience_counter'], 0)
    
    def test_model_saving_on_best_metrics(self):
        """Tester la sauvegarde automatique des meilleurs modèles."""
        # Premier epoch
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Vérifier que les fichiers sont créés
        best_loss_path = os.path.join(self.save_dir, "best_loss_model.pth")
        best_acc_path = os.path.join(self.save_dir, "best_acc_model.pth")
        
        self.assertTrue(os.path.exists(best_loss_path))
        self.assertTrue(os.path.exists(best_acc_path))
        
        # Deuxième epoch avec amélioration
        self.callback.on_epoch_end(
            1, self.real_model, 2.0, 2.3, 0.55, 0.52, self.real_optimizer
        )
        
        # Vérifier que les nouveaux fichiers sont créés
        self.assertTrue(os.path.exists(best_loss_path))
        self.assertTrue(os.path.exists(best_acc_path))
        
        # Vérifier que les fichiers contiennent les bonnes informations
        loss_checkpoint = torch.load(best_loss_path, map_location='cpu')
        acc_checkpoint = torch.load(best_acc_path, map_location='cpu')
        
        self.assertEqual(loss_checkpoint['val_loss'], 2.3)
        self.assertEqual(acc_checkpoint['val_acc'], 0.52)
    
    def test_learning_rate_tracking(self):
        """Tester le suivi du learning rate."""
        # Premier epoch avec le vrai optimiseur
        self.callback.on_epoch_end(
            0, self.real_model, 2.5, 2.8, 0.45, 0.42, self.real_optimizer
        )
        
        # Vérifier que le learning rate est enregistré
        self.assertEqual(self.callback.training_history['learning_rate'][0], 0.001)
        
        # Changer le learning rate de l'optimiseur réel
        for param_group in self.real_optimizer.param_groups:
            param_group['lr'] = 0.0005
        
        # Deuxième epoch
        self.callback.on_epoch_end(
            1, self.real_model, 2.0, 2.3, 0.55, 0.52, self.real_optimizer
        )
        
        # Vérifier que le nouveau learning rate est enregistré
        self.assertEqual(self.callback.training_history['learning_rate'][1], 0.0005)
    
    def test_real_training_simulation(self):
        """Tester le callback avec une simulation d'entraînement réaliste."""
        # Simuler un vrai entraînement sur plusieurs epochs
        for epoch in range(5):
            metrics = self.real_training_metrics[f'epoch_{epoch}']
            
            # Appeler le callback avec les vraies métriques
            epoch_info = self.callback.on_epoch_end(
                epoch, 
                self.real_model, 
                metrics['train_loss'], 
                metrics['val_loss'], 
                metrics['train_acc'], 
                metrics['val_acc'], 
                self.real_optimizer
            )
            
            # Vérifier que les informations sont cohérentes
            self.assertEqual(epoch_info['epoch'], epoch)
            self.assertEqual(epoch_info['train_loss'], metrics['train_loss'])
            self.assertEqual(epoch_info['val_loss'], metrics['val_loss'])
            self.assertEqual(epoch_info['train_acc'], metrics['train_acc'])
            self.assertEqual(epoch_info['val_acc'], metrics['val_acc'])
        
        # Vérifier l'historique complet
        self.assertEqual(len(self.callback.training_history['train_loss']), 5)
        self.assertEqual(len(self.callback.training_history['val_loss']), 5)
        self.assertEqual(len(self.callback.training_history['train_acc']), 5)
        self.assertEqual(len(self.callback.training_history['val_acc']), 5)
        
        # Vérifier que les meilleurs scores sont corrects
        self.assertEqual(self.callback.best_val_loss, 1.9)  # Meilleur loss à l'epoch 4
        self.assertEqual(self.callback.best_val_acc, 0.75)  # Meilleur accuracy à l'epoch 4
        
        # Vérifier que les fichiers de sauvegarde sont créés
        best_loss_path = os.path.join(self.save_dir, "best_loss_model.pth")
        best_acc_path = os.path.join(self.save_dir, "best_acc_model.pth")
        
        self.assertTrue(os.path.exists(best_loss_path))
        self.assertTrue(os.path.exists(best_acc_path))
    
    def test_model_state_dict_real(self):
        """Tester que le vrai modèle PyTorch peut être sauvegardé et rechargé."""
        # Sauvegarder le modèle
        filepath = os.path.join(self.save_dir, "real_model_test.pth")
        self.callback._save_model(self.real_model, filepath, 0, 2.5, 0.45)
        
        # Vérifier que le fichier est créé
        self.assertTrue(os.path.exists(filepath))
        
        # Charger le checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Vérifier que le state_dict est valide
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('image_encoder.0.weight', checkpoint['model_state_dict'])
        self.assertIn('geophysical_encoder.0.weight', checkpoint['model_state_dict'])
        self.assertIn('fusion_layer.0.weight', checkpoint['model_state_dict'])
        
        # Vérifier la configuration du modèle
        self.assertEqual(checkpoint['model_config']['num_classes'], 3)
        self.assertEqual(checkpoint['model_config']['image_model'], 'resnet18')
        self.assertEqual(checkpoint['model_config']['fusion_method'], 'concatenation')


if __name__ == '__main__':
    # Créer la suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_suite.addTest(unittest.makeSuite(TestHybridTrainingCallback))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print(f"RÉSULTATS DES TESTS HybridTrainingCallback")
    print(f"{'='*60}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print(f"\nÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n{'='*60}")
    print(f"TESTS HybridTrainingCallback TERMINÉS")
    print(f"{'='*60}")
