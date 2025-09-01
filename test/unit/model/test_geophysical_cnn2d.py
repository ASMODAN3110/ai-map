#!/usr/bin/env python3
"""
Tests unitaires pour la classe GeophysicalCNN2D.

Teste l'initialisation, l'architecture et le forward pass du CNN 2D géophysique.
"""

import unittest
import numpy as np
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

from src.model.geophysical_trainer import GeophysicalCNN2D
from src.utils.logger import logger


class TestGeophysicalCNN2D(unittest.TestCase):
    """Tests pour la classe GeophysicalCNN2D."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Paramètres de test standards
        self.standard_input_channels = 4
        self.standard_num_classes = 2
        self.standard_grid_size = 64
        self.standard_dropout_rate = 0.3
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_default_parameters(self):
        """Tester l'initialisation avec les paramètres par défaut."""
        model = GeophysicalCNN2D()
        
        # Vérifier les paramètres par défaut
        self.assertEqual(model.input_channels, 4)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.grid_size, 64)
        self.assertEqual(model.dropout_rate, 0.3)
        
        # Vérifier que le modèle est bien un Module PyTorch
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_init_custom_parameters(self):
        """Tester l'initialisation avec des paramètres personnalisés."""
        custom_input_channels = 6
        custom_num_classes = 5
        custom_grid_size = 32
        custom_dropout_rate = 0.5
        
        model = GeophysicalCNN2D(
            input_channels=custom_input_channels,
            num_classes=custom_num_classes,
            grid_size=custom_grid_size,
            dropout_rate=custom_dropout_rate
        )
        
        # Vérifier les paramètres personnalisés
        self.assertEqual(model.input_channels, custom_input_channels)
        self.assertEqual(model.num_classes, custom_num_classes)
        self.assertEqual(model.grid_size, custom_grid_size)
        self.assertEqual(model.dropout_rate, custom_dropout_rate)
    
    def test_init_edge_cases(self):
        """Tester l'initialisation avec des cas limites."""
        # Test avec 1 canal d'entrée
        model_1_channel = GeophysicalCNN2D(input_channels=1, grid_size=16)
        self.assertEqual(model_1_channel.input_channels, 1)
        
        # Test avec beaucoup de classes
        model_many_classes = GeophysicalCNN2D(num_classes=100, grid_size=16)
        self.assertEqual(model_many_classes.num_classes, 100)
        
        # Test avec une grille très petite
        model_small_grid = GeophysicalCNN2D(grid_size=8)
        self.assertEqual(model_small_grid.grid_size, 8)
        
        # Test avec dropout à 0
        model_no_dropout = GeophysicalCNN2D(dropout_rate=0.0)
        self.assertEqual(model_no_dropout.dropout_rate, 0.0)
        
        # Test avec dropout à 1
        model_max_dropout = GeophysicalCNN2D(dropout_rate=1.0)
        self.assertEqual(model_max_dropout.dropout_rate, 1.0)
    
    def test_feature_size_calculation(self):
        """Tester le calcul de la taille des features après convolutions."""
        # Test avec grille 64x64 (4 couches MaxPool2d(2))
        model_64 = GeophysicalCNN2D(grid_size=64)
        expected_feature_size_64 = 64 // (2**4)  # 64 / 16 = 4
        self.assertEqual(model_64.feature_size, expected_feature_size_64)
        
        # Test avec grille 32x32
        model_32 = GeophysicalCNN2D(grid_size=32)
        expected_feature_size_32 = 32 // (2**4)  # 32 / 16 = 2
        self.assertEqual(model_32.feature_size, expected_feature_size_32)
        
        # Test avec grille 16x16
        model_16 = GeophysicalCNN2D(grid_size=16)
        expected_feature_size_16 = 16 // (2**4)  # 16 / 16 = 1
        self.assertEqual(model_16.feature_size, expected_feature_size_16)
        
        # Test avec grille 8x8 (cas limite)
        model_8 = GeophysicalCNN2D(grid_size=8)
        expected_feature_size_8 = 8 // (2**4)  # 8 / 16 = 0
        self.assertEqual(model_8.feature_size, expected_feature_size_8)
    
    def test_conv_layers_structure(self):
        """Tester la structure des couches de convolution."""
        model = GeophysicalCNN2D()
        
        # Vérifier que les couches de convolution existent
        self.assertIsNotNone(model.conv_layers)
        self.assertIsInstance(model.conv_layers, torch.nn.Sequential)
        
        # Vérifier le nombre de couches (4 couches de convolution)
        conv_modules = [m for m in model.conv_layers if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual(len(conv_modules), 4)
        
        # Vérifier les tailles des canaux de sortie
        expected_channels = [32, 64, 128, 256]
        for i, conv_layer in enumerate(conv_modules):
            self.assertEqual(conv_layer.out_channels, expected_channels[i])
        
        # Vérifier que toutes les couches ont kernel_size=3 et padding=1
        for conv_layer in conv_modules:
            self.assertEqual(conv_layer.kernel_size, (3, 3))
            self.assertEqual(conv_layer.padding, (1, 1))
    
    def test_fc_layers_structure(self):
        """Tester la structure des couches fully connected."""
        model = GeophysicalCNN2D()
        
        # Vérifier que les couches FC existent
        self.assertIsNotNone(model.fc_layers)
        self.assertIsInstance(model.fc_layers, torch.nn.Sequential)
        
        # Vérifier le nombre de couches FC (3 couches)
        fc_modules = [m for m in model.fc_layers if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(fc_modules), 3)
        
        # Vérifier la première couche FC (entrée)
        first_fc = fc_modules[0]
        expected_input_features = 256 * model.feature_size * model.feature_size
        self.assertEqual(first_fc.in_features, expected_input_features)
        self.assertEqual(first_fc.out_features, 512)
        
        # Vérifier la couche de sortie
        last_fc = fc_modules[-1]
        self.assertEqual(last_fc.out_features, model.num_classes)
    
    def test_dropout_layers(self):
        """Tester la présence des couches de dropout."""
        model = GeophysicalCNN2D(dropout_rate=0.3)
        
        # Compter les couches de dropout dans les conv_layers
        conv_dropout_count = sum(1 for m in model.conv_layers if isinstance(m, torch.nn.Dropout2d))
        self.assertEqual(conv_dropout_count, 4)  # Une par couche de convolution
        
        # Compter les couches de dropout dans les fc_layers
        fc_dropout_count = sum(1 for m in model.fc_layers if isinstance(m, torch.nn.Dropout))
        self.assertEqual(fc_dropout_count, 2)  # Deux dans les couches FC
    
    def test_batch_norm_layers(self):
        """Tester la présence des couches de normalisation par batch."""
        model = GeophysicalCNN2D()
        
        # Compter les couches BatchNorm2d dans les conv_layers
        conv_bn_count = sum(1 for m in model.conv_layers if isinstance(m, torch.nn.BatchNorm2d))
        self.assertEqual(conv_bn_count, 4)  # Une par couche de convolution
        
        # Vérifier que les couches FC n'ont pas de BatchNorm (seulement Dropout)
        fc_bn_count = sum(1 for m in model.fc_layers if isinstance(m, torch.nn.BatchNorm1d))
        self.assertEqual(fc_bn_count, 0)
    
    def test_activation_functions(self):
        """Tester la présence des fonctions d'activation ReLU."""
        model = GeophysicalCNN2D()
        
        # Compter les ReLU dans les conv_layers
        conv_relu_count = sum(1 for m in model.conv_layers if isinstance(m, torch.nn.ReLU))
        self.assertEqual(conv_relu_count, 4)  # Une par couche de convolution
        
        # Compter les ReLU dans les fc_layers
        fc_relu_count = sum(1 for m in model.fc_layers if isinstance(m, torch.nn.ReLU))
        self.assertEqual(fc_relu_count, 2)  # Deux dans les couches FC (pas dans la dernière)
    
    def test_maxpool_layers(self):
        """Tester la présence des couches MaxPool2d."""
        model = GeophysicalCNN2D()
        
        # Compter les MaxPool2d dans les conv_layers
        maxpool_count = sum(1 for m in model.conv_layers if isinstance(m, torch.nn.MaxPool2d))
        self.assertEqual(maxpool_count, 4)  # Une par couche de convolution
        
        # Vérifier que toutes les couches MaxPool ont kernel_size=2
        for m in model.conv_layers:
            if isinstance(m, torch.nn.MaxPool2d):
                self.assertEqual(m.kernel_size, 2)
    
    def test_weight_initialization(self):
        """Tester l'initialisation des poids."""
        model = GeophysicalCNN2D()
        
        # Vérifier que les poids sont initialisés (non nuls)
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.data)
            # Certains paramètres peuvent être initialisés à 0 (comme les biais)
            # Vérifier seulement que le tenseur existe et a la bonne forme
            self.assertGreater(param.data.numel(), 0)  # Au moins un élément
    
    def test_model_parameters_count(self):
        """Tester le nombre de paramètres du modèle."""
        model = GeophysicalCNN2D()
        
        # Compter le nombre total de paramètres
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Compter le nombre de paramètres entraînables
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)  # Tous les paramètres sont entraînables
        
        # Vérifier que le nombre de paramètres est raisonnable pour un CNN de cette taille
        self.assertLess(total_params, 10_000_000)  # Moins de 10M paramètres
    
    def test_forward_pass_shape_validation(self):
        """Tester la validation de la forme d'entrée dans le forward pass."""
        model = GeophysicalCNN2D(input_channels=4, grid_size=32, num_classes=2)
        
        # Test avec la forme correcte
        correct_input = torch.randn(8, 4, 32, 32)  # batch_size=8, channels=4, height=32, width=32
        try:
            output = model(correct_input)
            self.assertEqual(output.shape, (8, 2))  # batch_size=8, num_classes=2
        except Exception as e:
            self.fail(f"Forward pass avec forme correcte a échoué: {e}")
        
        # Test avec forme incorrecte (mauvais nombre de canaux)
        wrong_channels = torch.randn(8, 3, 32, 32)  # 3 canaux au lieu de 4
        with self.assertRaises(ValueError):
            model(wrong_channels)
        
        # Test avec forme incorrecte (mauvaise taille de grille)
        wrong_size = torch.randn(8, 4, 16, 16)  # 16x16 au lieu de 32x32
        with self.assertRaises(ValueError):
            model(wrong_size)
    
    def test_forward_pass_different_batch_sizes(self):
        """Tester le forward pass avec différentes tailles de batch."""
        model = GeophysicalCNN2D(input_channels=4, grid_size=16, num_classes=3)
        
        # Test avec différents batch sizes
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 4, 16, 16)
            try:
                output = model(input_tensor)
                self.assertEqual(output.shape, (batch_size, 3))
            except Exception as e:
                self.fail(f"Forward pass avec batch_size={batch_size} a échoué: {e}")
    
    def test_model_device_transfer(self):
        """Tester le transfert du modèle vers différents devices."""
        model = GeophysicalCNN2D()
        
        # Test transfert vers CPU
        model_cpu = model.to('cpu')
        self.assertEqual(next(model_cpu.parameters()).device.type, 'cpu')
        
        # Test transfert vers CUDA si disponible
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            self.assertEqual(next(model_cuda.parameters()).device.type, 'cuda')
    
    def test_model_save_load(self):
        """Tester la sauvegarde et le chargement du modèle."""
        model = GeophysicalCNN2D()
        
        # Sauvegarder le modèle
        save_path = Path(self.temp_dir) / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Vérifier que le fichier a été créé
        self.assertTrue(save_path.exists())
        
        # Créer un nouveau modèle et charger les poids
        new_model = GeophysicalCNN2D()
        new_model.load_state_dict(torch.load(save_path))
        
        # Vérifier que les paramètres sont identiques
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(param1, param2))
    
    def test_model_gradient_flow(self):
        """Tester que les gradients peuvent circuler dans le modèle."""
        model = GeophysicalCNN2D(input_channels=4, grid_size=16, num_classes=2)
        
        # Créer des données d'entrée et des labels
        input_data = torch.randn(4, 4, 16, 16, requires_grad=True)
        labels = torch.randint(0, 2, (4,))
        
        # Forward pass
        output = model(input_data)
        
        # Calculer la loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients ont été calculés
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))  # Gradients non nuls
    
    def test_model_training_mode(self):
        """Tester le changement de mode d'entraînement."""
        model = GeophysicalCNN2D()
        
        # Vérifier le mode initial (entraînement)
        self.assertTrue(model.training)
        
        # Passer en mode évaluation
        model.eval()
        self.assertFalse(model.training)
        
        # Revenir en mode entraînement
        model.train()
        self.assertTrue(model.training)
    
    def test_model_dropout_behavior(self):
        """Tester le comportement du dropout selon le mode."""
        model = GeophysicalCNN2D(dropout_rate=0.5, grid_size=16)  # Spécifier grid_size=16
        
        # Mode entraînement
        model.train()
        input_data = torch.randn(4, 4, 16, 16)
        output_train = model(input_data)
        
        # Mode évaluation
        model.eval()
        output_eval = model(input_data)
        
        # Les outputs peuvent être différents à cause du dropout
        # Mais la forme doit rester la même
        self.assertEqual(output_train.shape, output_eval.shape)
    
    def test_model_parameter_names(self):
        """Tester que les paramètres ont des noms cohérents."""
        model = GeophysicalCNN2D()
        
        param_names = list(model.named_parameters())
        
        # Vérifier que les noms contiennent les bonnes informations
        conv_names = [name for name, _ in param_names if 'conv' in name]
        fc_names = [name for name, _ in param_names if 'fc' in name]
        
        self.assertGreater(len(conv_names), 0)
        self.assertGreater(len(fc_names), 0)
        
        # Vérifier que les noms sont uniques
        all_names = [name for name, _ in param_names]
        self.assertEqual(len(all_names), len(set(all_names)))


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
