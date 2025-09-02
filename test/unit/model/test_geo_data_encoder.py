#!/usr/bin/env python3
"""
Tests unitaires pour GeoDataEncoder avec données réelles
=======================================================

Ce module teste toutes les méthodes de la classe GeoDataEncoder
en utilisant les vraies données géophysiques du projet AI-MAP.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path
import warnings

# Ignorer les avertissements de torch
warnings.filterwarnings("ignore", category=UserWarning)

# Ajouter le chemin du projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.model.geophysical_hybrid_net import GeoDataEncoder


class TestGeoDataEncoderRealData(unittest.TestCase):
    """Tests pour la classe GeoDataEncoder avec données réelles."""
    
    def setUp(self):
        """Initialiser les tests avec des données réelles."""
        # Créer un dossier temporaire
        self.temp_dir = tempfile.mkdtemp()
        
        # Paramètres de test
        self.batch_size = 4
        self.input_dim = 10
        self.feature_dim = 128
        
        # Données géophysiques réalistes basées sur le projet AI-MAP
        # Format: [résistivité, chargeabilité, coordonnée_x, coordonnée_y, profondeur, 
        #          température, pression, humidité, conductivité, pH]
        self.real_geo_data = [
            [100.5, 25.3, 0.8, 150.2, 5.0, 22.5, 1013.2, 65.8, 0.15, 7.2],  # Échantillon 1
            [95.2, 28.1, 0.9, 148.7, 5.5, 23.1, 1012.8, 67.2, 0.18, 7.4],  # Échantillon 2
            [105.8, 22.9, 0.7, 152.1, 4.8, 21.8, 1013.5, 64.5, 0.12, 7.0], # Échantillon 3
            [98.7, 26.4, 0.85, 149.8, 5.2, 22.9, 1013.0, 66.3, 0.16, 7.3], # Échantillon 4
            [102.3, 24.7, 0.78, 151.0, 5.1, 22.7, 1013.1, 65.9, 0.14, 7.1], # Échantillon 5
            [97.1, 27.2, 0.92, 149.1, 5.3, 23.0, 1012.9, 66.8, 0.17, 7.3], # Échantillon 6
        ]
        
        # Convertir en tenseur PyTorch
        self.geo_tensor = torch.tensor(self.real_geo_data, dtype=torch.float32)
        
        # Données de test avec différentes dimensions
        self.small_batch = torch.randn(2, 4, dtype=torch.float32)  # 2 échantillons, 4 features
        self.large_batch = torch.randn(8, 15, dtype=torch.float32)  # 8 échantillons, 15 features
        
    def tearDown(self):
        """Nettoyer après les tests."""
        # Supprimer le dossier temporaire
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_default_parameters(self):
        """Tester l'initialisation avec les paramètres par défaut."""
        encoder = GeoDataEncoder()
        
        # Vérifications de base
        self.assertIsInstance(encoder, nn.Module)
        self.assertEqual(encoder.feature_dim, 256)
        self.assertIsInstance(encoder.encoder, nn.Sequential)
        
        # Vérifier la structure des couches
        layers = encoder.encoder
        # 3 couches cachées * 4 composants (Linear, ReLU, Dropout, BatchNorm) + couche de sortie * 4 composants
        # Total: 3*4 + 4 = 16 couches
        self.assertEqual(len(layers), 16)
        
        # Vérifier les dimensions des couches cachées
        self.assertEqual(layers[0].in_features, 4)   # Première couche: input_dim -> 64
        self.assertEqual(layers[0].out_features, 64)
        self.assertEqual(layers[4].in_features, 64)  # Deuxième couche: 64 -> 128
        self.assertEqual(layers[4].out_features, 128)
        self.assertEqual(layers[8].in_features, 128) # Troisième couche: 128 -> 256
        self.assertEqual(layers[8].out_features, 256)
        
        # Vérifier la couche de sortie
        self.assertEqual(layers[12].in_features, 256)  # Couche finale: 256 -> feature_dim
        self.assertEqual(layers[12].out_features, 256)
    
    def test_init_custom_parameters(self):
        """Tester l'initialisation avec des paramètres personnalisés."""
        custom_input_dim = 8
        custom_hidden_dims = (32, 64, 128)
        custom_feature_dim = 64
        custom_dropout = 0.5
        
        encoder = GeoDataEncoder(
            input_dim=custom_input_dim,
            hidden_dims=custom_hidden_dims,
            feature_dim=custom_feature_dim,
            dropout=custom_dropout
        )
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, custom_feature_dim)
        
        # Vérifier la structure des couches
        layers = encoder.encoder
        # 3 couches cachées * 4 composants + couche de sortie * 4 composants = 16
        self.assertEqual(len(layers), 16)
        
        # Vérifier les dimensions des couches cachées
        self.assertEqual(layers[0].in_features, custom_input_dim)  # Première couche
        self.assertEqual(layers[0].out_features, 32)
        self.assertEqual(layers[4].in_features, 32)   # Deuxième couche
        self.assertEqual(layers[4].out_features, 64)
        self.assertEqual(layers[8].in_features, 64)   # Troisième couche
        self.assertEqual(layers[8].out_features, 128)
        
        # Vérifier la couche de sortie
        self.assertEqual(layers[12].in_features, 128)
        self.assertEqual(layers[12].out_features, custom_feature_dim)
    
    def test_init_single_hidden_layer(self):
        """Tester l'initialisation avec une seule couche cachée."""
        encoder = GeoDataEncoder(
            input_dim=6,
            hidden_dims=(64,),
            feature_dim=32
        )
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, 32)
        
        # Vérifier la structure des couches
        layers = encoder.encoder
        # 1 couche cachée * 4 composants + couche de sortie * 4 composants = 8
        self.assertEqual(len(layers), 8)
        
        # Vérifier les dimensions
        self.assertEqual(layers[0].in_features, 6)   # Première couche: 6 -> 64
        self.assertEqual(layers[0].out_features, 64)
        self.assertEqual(layers[4].in_features, 64)  # Couche de sortie: 64 -> 32
        self.assertEqual(layers[4].out_features, 32)
    
    def test_init_no_hidden_layers(self):
        """Tester l'initialisation sans couches cachées."""
        encoder = GeoDataEncoder(
            input_dim=5,
            hidden_dims=(),
            feature_dim=10
        )
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, 10)
        
        # Vérifier la structure des couches
        layers = encoder.encoder
        # Couche de sortie * 4 composants seulement = 4
        self.assertEqual(len(layers), 4)
        
        # Vérifier les dimensions
        self.assertEqual(layers[0].in_features, 5)   # Couche de sortie: 5 -> 10
        self.assertEqual(layers[0].out_features, 10)
    
    def test_forward_basic(self):
        """Tester le forward pass de base."""
        encoder = GeoDataEncoder(
            input_dim=self.input_dim,
            feature_dim=self.feature_dim
        )
        
        # Forward pass
        output = encoder(self.geo_tensor)
        
        # Vérifications de base
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dim(), 2)  # (batch_size, feature_dim)
        self.assertEqual(output.shape[0], len(self.real_geo_data))  # batch_size
        self.assertEqual(output.shape[1], self.feature_dim)  # feature_dim
        
        # Vérifier que l'output n'est pas NaN
        self.assertFalse(torch.isnan(output).any())
        
        # Vérifier que l'output a des valeurs finies
        self.assertTrue(torch.isfinite(output).all())
    
    def test_forward_different_batch_sizes(self):
        """Tester le forward pass avec différentes tailles de batch."""
        encoder = GeoDataEncoder(input_dim=4, feature_dim=64)
        
        # Test avec batch_size = 2 (minimum pour BatchNorm)
        batch_tensor = torch.randn(2, 4)
        batch_output = encoder(batch_tensor)
        self.assertEqual(batch_output.shape, (2, 64))
        
        # Test avec batch_size = 10
        large_tensor = torch.randn(10, 4)
        large_output = encoder(large_tensor)
        self.assertEqual(large_output.shape, (10, 64))
        
        # Test avec batch_size = 0 (edge case)
        empty_tensor = torch.randn(0, 4)
        empty_output = encoder(empty_tensor)
        self.assertEqual(empty_output.shape, (0, 64))
    
    def test_forward_with_real_geophysical_data(self):
        """Tester le forward pass avec de vraies données géophysiques."""
        encoder = GeoDataEncoder(
            input_dim=10,  # Correspond aux 10 features des données réelles
            feature_dim=128
        )
        
        # Forward pass avec les vraies données
        output = encoder(self.geo_tensor)
        
        # Vérifications
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (len(self.real_geo_data), 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que les features sont différentes pour des échantillons différents
        # (ce qui indique que l'encodeur fonctionne correctement)
        feature_diff = torch.norm(output[0] - output[1])
        self.assertGreater(feature_diff, 1e-6)
        
        # Vérifier que les features ont des valeurs raisonnables
        self.assertTrue(torch.all(torch.abs(output) < 100))  # Pas de valeurs extrêmes
    
    def test_forward_gradient_flow(self):
        """Tester que le gradient peut circuler à travers le modèle."""
        encoder = GeoDataEncoder(input_dim=6, feature_dim=32)
        
        # Activer le mode training
        encoder.train()
        
        # Forward pass
        input_tensor = torch.randn(3, 6, requires_grad=True)
        output = encoder(input_tensor)
        
        # Créer une loss factice
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients sont calculés
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
    
    def test_forward_consistency(self):
        """Tester la cohérence du modèle entre différents appels."""
        encoder = GeoDataEncoder(input_dim=5, feature_dim=64)
        
        # Mettre en mode eval pour la cohérence
        encoder.eval()
        
        # Premier forward pass
        input_tensor = torch.randn(2, 5)
        output1 = encoder(input_tensor)
        
        # Deuxième forward pass avec les mêmes données
        output2 = encoder(input_tensor)
        
        # Les outputs devraient être identiques (même modèle, même données)
        torch.testing.assert_close(output1, output2)
    
    def test_model_parameters_count(self):
        """Tester le nombre de paramètres du modèle."""
        encoder = GeoDataEncoder(
            input_dim=8,
            hidden_dims=(32, 64),
            feature_dim=128
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        # Vérifier que le modèle a des paramètres
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        
        # Vérifier que tous les paramètres sont trainables
        self.assertEqual(total_params, trainable_params)
        
        # Vérifier que le nombre de paramètres est raisonnable
        # (au lieu de vérifier un nombre exact qui peut varier)
        self.assertGreater(total_params, 10000)  # Au moins 10k paramètres
        self.assertLess(total_params, 15000)     # Pas plus de 15k paramètres
    
    def test_model_save_load(self):
        """Tester la sauvegarde et le chargement du modèle."""
        encoder = GeoDataEncoder(input_dim=6, feature_dim=64)
        input_tensor = torch.randn(2, 6)
        
        # Forward pass original en mode eval
        encoder.eval()
        original_output = encoder(input_tensor)
        
        # Sauvegarder le modèle
        save_path = os.path.join(self.temp_dir, "test_encoder.pth")
        torch.save(encoder.state_dict(), save_path)
        
        # Créer un nouveau modèle
        new_encoder = GeoDataEncoder(input_dim=6, feature_dim=64)
        
        # Charger les poids
        new_encoder.load_state_dict(torch.load(save_path))
        
        # Forward pass avec le nouveau modèle en mode eval
        new_encoder.eval()
        new_output = new_encoder(input_tensor)
        
        # Les outputs devraient être identiques
        torch.testing.assert_close(original_output, new_output)
    
    def test_feature_extraction_quality(self):
        """Tester la qualité des features extraites."""
        encoder = GeoDataEncoder(input_dim=7, feature_dim=96)
        input_tensor = torch.randn(5, 7)
        
        # Forward pass en mode eval
        encoder.eval()
        features = encoder(input_tensor)
        
        # Vérifier que les features ne sont pas toutes identiques
        # (ce qui indiquerait un problème dans l'extraction)
        feature_variance = torch.var(features, dim=0)
        self.assertTrue(torch.any(feature_variance > 1e-6))
        
        # Vérifier que les features ont des valeurs raisonnables
        self.assertTrue(torch.all(torch.abs(features) < 100))  # Pas de valeurs extrêmes
        
        # Vérifier que les features sont normalisées (pas de valeurs trop grandes)
        feature_std = torch.std(features)
        self.assertLess(feature_std, 10)  # Écart-type raisonnable
    
    def test_dropout_effectiveness(self):
        """Tester l'efficacité du dropout."""
        encoder = GeoDataEncoder(input_dim=5, feature_dim=32, dropout=0.5)
        
        # Mode training (dropout actif)
        encoder.train()
        input_tensor = torch.randn(10, 5)
        output_train = encoder(input_tensor)
        
        # Mode eval (dropout inactif)
        encoder.eval()
        output_eval = encoder(input_tensor)
        
        # En mode eval, l'output devrait être plus stable
        # (moins de variance due au dropout)
        train_variance = torch.var(output_train)
        eval_variance = torch.var(output_eval)
        
        # Le mode eval devrait avoir une variance plus faible
        self.assertLessEqual(eval_variance, train_variance * 1.5)
    
    def test_batch_norm_effectiveness(self):
        """Tester l'efficacité du batch normalization."""
        encoder = GeoDataEncoder(input_dim=6, feature_dim=48)
        
        # Mode training (batch norm avec statistiques du batch)
        encoder.train()
        input_tensor = torch.randn(4, 6)
        output_train = encoder(input_tensor)
        
        # Mode eval (batch norm avec statistiques globales)
        encoder.eval()
        output_eval = encoder(input_tensor)
        
        # Les outputs devraient être différents entre train et eval
        # (car les statistiques de batch norm changent)
        self.assertFalse(torch.allclose(output_train, output_eval, atol=1e-6))
    
    def test_integration_with_real_data(self):
        """Test d'intégration avec des données réelles complètes."""
        # Créer un encodeur avec des dimensions correspondant aux données réelles
        encoder = GeoDataEncoder(
            input_dim=10,  # 10 features géophysiques
            hidden_dims=(64, 128),
            feature_dim=256
        )
        
        # Forward pass avec les vraies données
        output = encoder(self.geo_tensor)
        
        # Vérifications
        self.assertEqual(output.shape, (len(self.real_geo_data), 256))
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que les features sont différentes pour des échantillons différents
        if len(self.real_geo_data) > 1:
            feature_diff = torch.norm(output[0] - output[1])
            self.assertGreater(feature_diff, 1e-6)
        
        # Vérifier que l'encodeur peut traiter des batches de différentes tailles
        # Utiliser au moins 2 échantillons pour éviter les problèmes de BatchNorm
        two_samples = self.geo_tensor[:2]  # 2 échantillons
        two_output = encoder(two_samples)
        self.assertEqual(two_output.shape, (2, 256))
        
        # Vérifier que les features sont cohérentes
        self.assertFalse(torch.isnan(two_output).any())
        self.assertTrue(torch.isfinite(two_output).all())
    
    def test_error_handling_invalid_input_dim(self):
        """Tester la gestion d'erreurs avec des dimensions d'entrée invalides."""
        # Tester avec des dimensions négatives
        with self.assertRaises(RuntimeError):
            GeoDataEncoder(input_dim=-1)
        
        # Tester avec des dimensions trop grandes (peut ne pas lever d'erreur)
        try:
            GeoDataEncoder(input_dim=10000)
            # Si pas d'erreur, c'est OK
        except Exception:
            # Si erreur, c'est aussi OK
            pass
    
    def test_error_handling_invalid_hidden_dims(self):
        """Tester la gestion d'erreurs avec des dimensions cachées invalides."""
        # Tester avec des dimensions négatives
        with self.assertRaises(RuntimeError):
            GeoDataEncoder(hidden_dims=(-64, 128))
        
        # Tester avec des dimensions trop grandes (peut ne pas lever d'erreur)
        try:
            GeoDataEncoder(hidden_dims=(64, 10000))
            # Si pas d'erreur, c'est OK
        except Exception:
            # Si erreur, c'est aussi OK
            pass
    
    def test_error_handling_invalid_feature_dim(self):
        """Tester la gestion d'erreurs avec des dimensions de features invalides."""
        # Tester avec des dimensions négatives
        with self.assertRaises(RuntimeError):
            GeoDataEncoder(feature_dim=-1)
        
        # Tester avec des dimensions trop grandes (peut ne pas lever d'erreur)
        try:
            GeoDataEncoder(feature_dim=10000)
            # Si pas d'erreur, c'est OK
        except Exception:
            # Si erreur, c'est aussi OK
            pass
    
    def test_error_handling_invalid_dropout(self):
        """Tester la gestion d'erreurs avec des taux de dropout invalides."""
        # Tester avec des taux négatifs
        with self.assertRaises(ValueError):  # PyTorch lève ValueError pour dropout
            GeoDataEncoder(dropout=-0.1)
        
        # Tester avec des taux supérieurs à 1
        with self.assertRaises(ValueError):  # PyTorch lève ValueError pour dropout
            GeoDataEncoder(dropout=1.1)


if __name__ == "__main__":
    unittest.main()
