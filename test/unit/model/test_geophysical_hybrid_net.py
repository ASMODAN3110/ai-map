"""
Tests Unitaires pour le Modèle Hybride Images + Données Géophysiques
===================================================================

Ce module teste toutes les fonctionnalités du modèle hybride qui combine
images et données géophysiques.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

# Import du module à tester
from src.model.geophysical_hybrid_net import (
    ImageEncoder,
    GeoDataEncoder, 
    FusionModule,
    GeophysicalHybridNet,
    create_hybrid_model,
    get_model_summary
)


class TestImageEncoder(unittest.TestCase):
    """Tests pour la classe ImageEncoder."""
    
    def setUp(self):
        """Initialiser les tests."""
        self.encoder = ImageEncoder(model_name="resnet18", feature_dim=256)
    
    def test_init(self):
        """Tester l'initialisation de l'encodeur d'images."""
        self.assertEqual(self.encoder.feature_dim, 256)
        self.assertIsNotNone(self.encoder.backbone)
    
    def test_init_resnet18(self):
        """Tester l'initialisation avec ResNet18."""
        encoder = ImageEncoder(model_name="resnet18", feature_dim=512)
        self.assertEqual(encoder.feature_dim, 512)
    
    def test_init_resnet34(self):
        """Tester l'initialisation avec ResNet34."""
        encoder = ImageEncoder(model_name="resnet34", feature_dim=512)
        self.assertEqual(encoder.feature_dim, 512)
    
    def test_init_resnet50(self):
        """Tester l'initialisation avec ResNet50."""
        encoder = ImageEncoder(model_name="resnet50", feature_dim=1024)
        self.assertEqual(encoder.feature_dim, 1024)
    
    def test_init_unsupported_model(self):
        """Tester l'initialisation avec un modèle non supporté."""
        with self.assertRaises(ValueError):
            ImageEncoder(model_name="unsupported_model")
    
    def test_freeze_backbone(self):
        """Tester le gel du backbone."""
        encoder = ImageEncoder(model_name="resnet18", freeze_backbone=True)
        
        # Vérifier que les paramètres du backbone sont gelés
        for param in encoder.backbone.parameters():
            self.assertFalse(param.requires_grad)
    
    def test_forward(self):
        """Tester le forward pass."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 64, 64)
        
        output = self.encoder(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 256))
        self.assertIsInstance(output, torch.Tensor)


class TestGeoDataEncoder(unittest.TestCase):
    """Tests pour la classe GeoDataEncoder."""
    
    def setUp(self):
        """Initialiser les tests."""
        self.encoder = GeoDataEncoder(input_dim=5, feature_dim=128)
    
    def test_init(self):
        """Tester l'initialisation de l'encodeur de données géophysiques."""
        self.assertEqual(self.encoder.feature_dim, 128)
        self.assertIsNotNone(self.encoder.encoder)
    
    def test_init_custom_hidden_dims(self):
        """Tester l'initialisation avec des dimensions cachées personnalisées."""
        encoder = GeoDataEncoder(
            input_dim=4, 
            hidden_dims=(32, 64), 
            feature_dim=128
        )
        self.assertEqual(encoder.feature_dim, 128)
    
    def test_init_custom_dropout(self):
        """Tester l'initialisation avec un dropout personnalisé."""
        encoder = GeoDataEncoder(input_dim=4, dropout=0.5)
        self.assertIsNotNone(encoder.encoder)
    
    def test_forward(self):
        """Tester le forward pass."""
        batch_size = 8
        input_tensor = torch.randn(batch_size, 5)
        
        output = self.encoder(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 128))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_forward_different_input_dim(self):
        """Tester le forward pass avec une dimension d'entrée différente."""
        encoder = GeoDataEncoder(input_dim=3, feature_dim=64)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3)
        
        output = encoder(input_tensor)
        self.assertEqual(output.shape, (batch_size, 64))


class TestFusionModule(unittest.TestCase):
    """Tests pour la classe FusionModule."""
    
    def setUp(self):
        """Initialiser les tests."""
        self.fusion = FusionModule(
            image_features=512,
            geo_features=256,
            num_classes=3
        )
    
    def test_init_concatenation(self):
        """Tester l'initialisation avec fusion par concaténation."""
        fusion = FusionModule(fusion_method="concatenation")
        self.assertEqual(fusion.fusion_method, "concatenation")
    
    def test_init_attention(self):
        """Tester l'initialisation avec fusion par attention."""
        fusion = FusionModule(fusion_method="attention")
        self.assertEqual(fusion.fusion_method, "attention")
        self.assertIsNotNone(fusion.attention)
    
    def test_init_weighted(self):
        """Tester l'initialisation avec fusion pondérée."""
        fusion = FusionModule(fusion_method="weighted")
        self.assertEqual(fusion.fusion_method, "weighted")
        self.assertIsNotNone(fusion.image_weight)
        self.assertIsNotNone(fusion.geo_weight)
    
    def test_init_unsupported_method(self):
        """Tester l'initialisation avec une méthode non supportée."""
        with self.assertRaises(ValueError):
            FusionModule(fusion_method="unsupported")
    
    def test_forward_concatenation(self):
        """Tester le forward pass avec fusion par concaténation."""
        batch_size = 4
        image_features = torch.randn(batch_size, 512)
        geo_features = torch.randn(batch_size, 256)
        
        output = self.fusion(image_features, geo_features)
        
        self.assertEqual(output.shape, (batch_size, 3))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_forward_attention(self):
        """Tester le forward pass avec fusion par attention."""
        fusion = FusionModule(fusion_method="attention")
        batch_size = 4
        image_features = torch.randn(batch_size, 512)
        geo_features = torch.randn(batch_size, 256)
        
        output = fusion(image_features, geo_features)
        
        self.assertEqual(output.shape, (batch_size, 3))
    
    def test_forward_weighted(self):
        """Tester le forward pass avec fusion pondérée."""
        fusion = FusionModule(fusion_method="weighted")
        batch_size = 4
        image_features = torch.randn(batch_size, 512)
        geo_features = torch.randn(batch_size, 256)
        
        output = fusion(image_features, geo_features)
        
        self.assertEqual(output.shape, (batch_size, 3))
    
    def test_forward_weighted_different_dims(self):
        """Tester le forward pass avec fusion pondérée et dimensions différentes."""
        fusion = FusionModule(fusion_method="weighted")
        batch_size = 4
        image_features = torch.randn(batch_size, 256)
        geo_features = torch.randn(batch_size, 512)
        
        output = fusion(image_features, geo_features)
        
        self.assertEqual(output.shape, (batch_size, 3))


class TestGeophysicalHybridNet(unittest.TestCase):
    """Tests pour la classe GeophysicalHybridNet."""
    
    def setUp(self):
        """Initialiser les tests."""
        self.model = GeophysicalHybridNet(
            num_classes=2,
            image_model="resnet18",
            geo_input_dim=4
        )
    
    def test_init(self):
        """Tester l'initialisation du modèle hybride."""
        self.assertEqual(self.model.num_classes, 2)
        self.assertEqual(self.model.image_model, "resnet18")
        self.assertEqual(self.model.fusion_method, "concatenation")
        
        # Vérifier les composants
        self.assertIsInstance(self.model.image_encoder, ImageEncoder)
        self.assertIsInstance(self.model.geo_encoder, GeoDataEncoder)
        self.assertIsInstance(self.model.fusion, FusionModule)
    
    def test_init_custom_fusion(self):
        """Tester l'initialisation avec une méthode de fusion personnalisée."""
        model = GeophysicalHybridNet(fusion_method="attention")
        self.assertEqual(model.fusion_method, "attention")
    
    def test_init_freeze_backbone(self):
        """Tester l'initialisation avec backbone gelé."""
        model = GeophysicalHybridNet(freeze_backbone=True)
        
        # Vérifier que certains paramètres du backbone sont gelés
        backbone_params = list(model.image_encoder.backbone.parameters())
        self.assertFalse(backbone_params[0].requires_grad)
    
    def test_forward(self):
        """Tester le forward pass du modèle hybride."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        geo_data = torch.randn(batch_size, 4)
        
        output = self.model(images, geo_data)
        
        self.assertEqual(output.shape, (batch_size, 2))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_forward_different_batch_size(self):
        """Tester le forward pass avec différentes tailles de batch."""
        batch_sizes = [1, 8, 16]
        
        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 64, 64)
            geo_data = torch.randn(batch_size, 4)
            
            output = self.model(images, geo_data)
            self.assertEqual(output.shape, (batch_size, 2))
    
    def test_get_feature_maps(self):
        """Tester l'obtention des features intermédiaires."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        geo_data = torch.randn(batch_size, 4)
        
        features = self.model.get_feature_maps(images, geo_data)
        
        expected_keys = ['image_features', 'geo_features', 'output']
        for key in expected_keys:
            self.assertIn(key, features)
        
        self.assertEqual(features['image_features'].shape, (batch_size, 512))
        self.assertEqual(features['geo_features'].shape, (batch_size, 256))
        self.assertEqual(features['output'].shape, (batch_size, 2))
    
    def test_count_parameters(self):
        """Tester le comptage des paramètres."""
        param_counts = self.model.count_parameters()
        
        expected_keys = ['total_parameters', 'trainable_parameters', 'frozen_parameters']
        for key in expected_keys:
            self.assertIn(key, param_counts)
        
        # Vérifier que les comptes sont cohérents
        total = param_counts['total_parameters']
        trainable = param_counts['trainable_parameters']
        frozen = param_counts['frozen_parameters']
        
        self.assertEqual(total, trainable + frozen)
        self.assertGreater(total, 0)
    
    def test_model_configuration(self):
        """Tester différentes configurations du modèle."""
        configs = [
            {'num_classes': 3, 'image_model': 'resnet34'},
            {'num_classes': 5, 'image_model': 'resnet50', 'fusion_method': 'attention'},
            {'num_classes': 2, 'geo_input_dim': 6, 'fusion_method': 'weighted'}
        ]
        
        for config in configs:
            model = GeophysicalHybridNet(**config)
            
            # Vérifier que la configuration est appliquée
            for key, value in config.items():
                if hasattr(model, key):
                    self.assertEqual(getattr(model, key), value)
            
            # Tester le forward pass
            batch_size = 2
            images = torch.randn(batch_size, 3, 64, 64)
            geo_data = torch.randn(batch_size, config.get('geo_input_dim', 4))
            
            output = model(images, geo_data)
            self.assertEqual(output.shape, (batch_size, config['num_classes']))


class TestUtilityFunctions(unittest.TestCase):
    """Tests pour les fonctions utilitaires."""
    
    def test_create_hybrid_model(self):
        """Tester la fonction de création de modèle hybride."""
        model = create_hybrid_model(
            num_classes=4,
            image_model="resnet18",
            geo_input_dim=6
        )
        
        self.assertIsInstance(model, GeophysicalHybridNet)
        self.assertEqual(model.num_classes, 4)
        self.assertEqual(model.image_model, "resnet18")
    
    def test_get_model_summary(self):
        """Tester la fonction d'obtention du résumé du modèle."""
        model = GeophysicalHybridNet(num_classes=2)
        summary = get_model_summary(model)
        
        expected_keys = ['model_type', 'image_model', 'fusion_method', 
                        'num_classes', 'parameters', 'architecture']
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
        self.assertEqual(summary['num_classes'], 2)


class TestModelIntegration(unittest.TestCase):
    """Tests d'intégration pour le modèle complet."""
    
    def setUp(self):
        """Initialiser les tests."""
        self.model = GeophysicalHybridNet(
            num_classes=3,
            image_model="resnet18",
            geo_input_dim=5
        )
    
    def test_end_to_end_pipeline(self):
        """Tester le pipeline complet du modèle."""
        batch_size = 8
        
        # Données d'entrée
        images = torch.randn(batch_size, 3, 64, 64)
        geo_data = torch.randn(batch_size, 5)
        
        # Forward pass
        output = self.model(images, geo_data)
        
        # Vérifications
        self.assertEqual(output.shape, (batch_size, 3))
        self.assertIsInstance(output, torch.Tensor)
        
        # Vérifier que les prédictions sont des logits valides
        self.assertTrue(torch.isfinite(output).all())
    
    def test_feature_extraction_pipeline(self):
        """Tester le pipeline d'extraction de features."""
        batch_size = 4
        
        # Données d'entrée
        images = torch.randn(batch_size, 3, 64, 64)
        geo_data = torch.randn(batch_size, 5)
        
        # Extraire les features
        features = self.model.get_feature_maps(images, geo_data)
        
        # Vérifier la cohérence des features
        self.assertEqual(features['image_features'].shape[0], batch_size)
        self.assertEqual(features['geo_features'].shape[0], batch_size)
        self.assertEqual(features['output'].shape[0], batch_size)
    
    def test_model_parameters_consistency(self):
        """Tester la cohérence des paramètres du modèle."""
        param_counts = self.model.count_parameters()
        
        # Vérifier que tous les paramètres sont comptés
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(param_counts['total_parameters'], total_params)
        
        # Vérifier que les paramètres entraînables sont comptés
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(param_counts['trainable_parameters'], trainable_params)


if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2)
