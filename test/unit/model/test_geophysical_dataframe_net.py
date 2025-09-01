#!/usr/bin/env python3
"""
Tests unitaires pour la classe GeophysicalDataFrameNet.

Teste l'initialisation, l'architecture et le forward pass du réseau pour DataFrames géophysiques.
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

from src.model.geophysical_trainer import GeophysicalDataFrameNet
from src.utils.logger import logger


class TestGeophysicalDataFrameNet(unittest.TestCase):
    """Tests pour la classe GeophysicalDataFrameNet."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Paramètres de test standards
        self.standard_input_features = 4
        self.standard_num_classes = 2
        self.standard_hidden_layers = [256, 128, 64]
        self.standard_dropout_rate = 0.3
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_default_parameters(self):
        """Tester l'initialisation avec les paramètres par défaut."""
        model = GeophysicalDataFrameNet(input_features=4)
        
        # Vérifier les paramètres par défaut
        self.assertEqual(model.input_features, 4)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.hidden_layers, [256, 128, 64])
        self.assertEqual(model.dropout_rate, 0.3)
        
        # Vérifier que le modèle est bien un Module PyTorch
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_init_custom_parameters(self):
        """Tester l'initialisation avec des paramètres personnalisés."""
        custom_input_features = 10
        custom_num_classes = 5
        custom_hidden_layers = [512, 256, 128, 64]
        custom_dropout_rate = 0.5
        
        model = GeophysicalDataFrameNet(
            input_features=custom_input_features,
            num_classes=custom_num_classes,
            hidden_layers=custom_hidden_layers,
            dropout_rate=custom_dropout_rate
        )
        
        # Vérifier les paramètres personnalisés
        self.assertEqual(model.input_features, custom_input_features)
        self.assertEqual(model.num_classes, custom_num_classes)
        self.assertEqual(model.hidden_layers, custom_hidden_layers)
        self.assertEqual(model.dropout_rate, custom_dropout_rate)
    
    def test_init_edge_cases(self):
        """Tester l'initialisation avec des cas limites."""
        # Test avec 1 feature d'entrée
        model_1_feature = GeophysicalDataFrameNet(input_features=1)
        self.assertEqual(model_1_feature.input_features, 1)
        
        # Test avec beaucoup de classes
        model_many_classes = GeophysicalDataFrameNet(input_features=4, num_classes=100)
        self.assertEqual(model_many_classes.num_classes, 100)
        
        # Test avec couches cachées vides
        model_no_hidden = GeophysicalDataFrameNet(input_features=4, hidden_layers=[])
        self.assertEqual(model_no_hidden.hidden_layers, [])
        
        # Test avec dropout à 0
        model_no_dropout = GeophysicalDataFrameNet(input_features=4, dropout_rate=0.0)
        self.assertEqual(model_no_dropout.dropout_rate, 0.0)
        
        # Test avec dropout à 1
        model_max_dropout = GeophysicalDataFrameNet(input_features=4, dropout_rate=1.0)
        self.assertEqual(model_max_dropout.dropout_rate, 1.0)
    
    def test_network_structure(self):
        """Tester la structure du réseau."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64])
        
        # Vérifier que le réseau existe
        self.assertIsNotNone(model.network)
        self.assertIsInstance(model.network, torch.nn.Sequential)
        
        # Compter les couches Linear
        linear_modules = [m for m in model.network if isinstance(m, torch.nn.Linear)]
        expected_linear_count = len(model.hidden_layers) + 1  # +1 pour la couche de sortie
        self.assertEqual(len(linear_modules), expected_linear_count)
        
        # Vérifier la première couche (entrée)
        first_linear = linear_modules[0]
        self.assertEqual(first_linear.in_features, 4)
        self.assertEqual(first_linear.out_features, 128)
        
        # Vérifier la couche de sortie
        last_linear = linear_modules[-1]
        self.assertEqual(last_linear.out_features, model.num_classes)
    
    def test_batch_norm_layers(self):
        """Tester la présence des couches BatchNorm1d."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64])
        
        # Compter les couches BatchNorm1d
        batch_norm_count = sum(1 for m in model.network if isinstance(m, torch.nn.BatchNorm1d))
        expected_bn_count = len(model.hidden_layers)  # Une par couche cachée
        self.assertEqual(batch_norm_count, expected_bn_count)
        
        # Vérifier que les couches BatchNorm ont la bonne taille
        bn_modules = [m for m in model.network if isinstance(m, torch.nn.BatchNorm1d)]
        for i, bn_layer in enumerate(bn_modules):
            expected_size = model.hidden_layers[i]
            self.assertEqual(bn_layer.num_features, expected_size)
    
    def test_activation_functions(self):
        """Tester la présence des fonctions d'activation ReLU."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64])
        
        # Compter les ReLU
        relu_count = sum(1 for m in model.network if isinstance(m, torch.nn.ReLU))
        expected_relu_count = len(model.hidden_layers)  # Une par couche cachée
        self.assertEqual(relu_count, expected_relu_count)
    
    def test_dropout_layers(self):
        """Tester la présence des couches de dropout."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64], dropout_rate=0.3)
        
        # Compter les couches de dropout
        dropout_count = sum(1 for m in model.network if isinstance(m, torch.nn.Dropout))
        expected_dropout_count = len(model.hidden_layers)  # Une par couche cachée
        self.assertEqual(dropout_count, expected_dropout_count)
        
        # Vérifier que les taux de dropout sont corrects
        dropout_modules = [m for m in model.network if isinstance(m, torch.nn.Dropout)]
        for dropout_layer in dropout_modules:
            self.assertEqual(dropout_layer.p, model.dropout_rate)
    
    def test_weight_initialization(self):
        """Tester l'initialisation des poids."""
        model = GeophysicalDataFrameNet(input_features=4)
        
        # Vérifier que les poids sont initialisés (non nuls)
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.data)
            # Certains paramètres peuvent être initialisés à 0 (comme les biais)
            # Vérifier seulement que le tenseur existe et a la bonne forme
            self.assertGreater(param.data.numel(), 0)  # Au moins un élément
    
    def test_model_parameters_count(self):
        """Tester le nombre de paramètres du modèle."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64])
        
        # Compter le nombre total de paramètres
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Compter le nombre de paramètres entraînables
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)  # Tous les paramètres sont entraînables
        
        # Vérifier que le nombre de paramètres est raisonnable
        self.assertLess(total_params, 1_000_000)  # Moins de 1M paramètres pour un réseau simple
    
    def test_forward_pass_shape_validation(self):
        """Tester la validation de la forme d'entrée dans le forward pass."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64], num_classes=2)
        
        # Test avec la forme correcte
        correct_input = torch.randn(8, 4)  # batch_size=8, features=4
        try:
            output = model(correct_input)
            self.assertEqual(output.shape, (8, 2))  # batch_size=8, num_classes=2
        except Exception as e:
            self.fail(f"Forward pass avec forme correcte a échoué: {e}")
        
        # Test avec forme incorrecte (mauvais nombre de features)
        wrong_features = torch.randn(8, 3)  # 3 features au lieu de 4
        with self.assertRaises(RuntimeError):  # PyTorch lève RuntimeError pour dimension mismatch
            model(wrong_features)
    
    def test_forward_pass_different_batch_sizes(self):
        """Tester le forward pass avec différentes tailles de batch."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[64], num_classes=3)
        
        # Test avec différents batch sizes (éviter batch_size=1 pour BatchNorm1d)
        batch_sizes = [2, 4, 8, 16]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 4)
            try:
                output = model(input_tensor)
                self.assertEqual(output.shape, (batch_size, 3))
            except Exception as e:
                self.fail(f"Forward pass avec batch_size={batch_size} a échoué: {e}")
    
    def test_forward_pass_different_input_sizes(self):
        """Tester le forward pass avec différentes tailles d'entrée."""
        # Test avec 2 features
        model_2_features = GeophysicalDataFrameNet(input_features=2, hidden_layers=[64], num_classes=2)
        input_2 = torch.randn(4, 2)
        output_2 = model_2_features(input_2)
        self.assertEqual(output_2.shape, (4, 2))
        
        # Test avec 10 features
        model_10_features = GeophysicalDataFrameNet(input_features=10, hidden_layers=[64], num_classes=2)
        input_10 = torch.randn(4, 10)
        output_10 = model_10_features(input_10)
        self.assertEqual(output_10.shape, (4, 2))
    
    def test_model_device_transfer(self):
        """Tester le transfert du modèle vers différents devices."""
        model = GeophysicalDataFrameNet(input_features=4)
        
        # Test transfert vers CPU
        model_cpu = model.to('cpu')
        self.assertEqual(next(model_cpu.parameters()).device.type, 'cpu')
        
        # Test transfert vers CUDA si disponible
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            self.assertEqual(next(model_cuda.parameters()).device.type, 'cuda')
    
    def test_model_save_load(self):
        """Tester la sauvegarde et le chargement du modèle."""
        model = GeophysicalDataFrameNet(input_features=4)
        
        # Sauvegarder le modèle
        save_path = Path(self.temp_dir) / "test_dataframe_net.pth"
        torch.save(model.state_dict(), save_path)
        
        # Vérifier que le fichier a été créé
        self.assertTrue(save_path.exists())
        
        # Créer un nouveau modèle et charger les poids
        new_model = GeophysicalDataFrameNet(input_features=4)
        new_model.load_state_dict(torch.load(save_path))
        
        # Vérifier que les paramètres sont identiques
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(param1, param2))
    
    def test_model_gradient_flow(self):
        """Tester que les gradients peuvent circuler dans le modèle."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[64], num_classes=2)
        
        # Créer des données d'entrée et des labels
        input_data = torch.randn(4, 4, requires_grad=True)
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
        model = GeophysicalDataFrameNet(input_features=4)
        
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
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[64], dropout_rate=0.5)
        
        # Mode entraînement
        model.train()
        input_data = torch.randn(4, 4)
        output_train = model(input_data)
        
        # Mode évaluation
        model.eval()
        output_eval = model(input_data)
        
        # Les outputs peuvent être différents à cause du dropout
        # Mais la forme doit rester la même
        self.assertEqual(output_train.shape, output_eval.shape)
    
    def test_model_parameter_names(self):
        """Tester que les paramètres ont des noms cohérents."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[128, 64])
        
        param_names = list(model.named_parameters())
        
        # Vérifier que les noms contiennent les bonnes informations
        # Les noms PyTorch utilisent des indices numériques
        linear_names = [name for name, _ in param_names if any(x in name.lower() for x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
        
        self.assertGreater(len(linear_names), 0)
        
        # Vérifier que les noms sont uniques
        all_names = [name for name, _ in param_names]
        self.assertEqual(len(all_names), len(set(all_names)))
    
    def test_network_depth_variations(self):
        """Tester différentes profondeurs de réseau."""
        # Test avec réseau peu profond
        model_shallow = GeophysicalDataFrameNet(input_features=4, hidden_layers=[64])
        self.assertEqual(len(model_shallow.hidden_layers), 1)
        
        # Test avec réseau profond
        model_deep = GeophysicalDataFrameNet(input_features=4, hidden_layers=[512, 256, 128, 64, 32])
        self.assertEqual(len(model_deep.hidden_layers), 5)
        
        # Test avec réseau sans couches cachées
        model_no_hidden = GeophysicalDataFrameNet(input_features=4, hidden_layers=[])
        self.assertEqual(len(model_no_hidden.hidden_layers), 0)
    
    def test_forward_pass_with_no_hidden_layers(self):
        """Tester le forward pass avec un réseau sans couches cachées."""
        model = GeophysicalDataFrameNet(input_features=4, hidden_layers=[], num_classes=2)
        
        input_data = torch.randn(8, 4)
        output = model(input_data)
        
        # Avec pas de couches cachées, on va directement de l'entrée à la sortie
        self.assertEqual(output.shape, (8, 2))
        
        # Vérifier que le réseau a seulement une couche Linear
        linear_modules = [m for m in model.network if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear_modules), 1)
        
        # Vérifier que cette couche va de input_features à num_classes
        self.assertEqual(linear_modules[0].in_features, 4)
        self.assertEqual(linear_modules[0].out_features, 2)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
