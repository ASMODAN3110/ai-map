#!/usr/bin/env python3
"""
Tests unitaires pour ImageEncoder avec données réelles
=====================================================

Ce module teste toutes les méthodes de la classe ImageEncoder
en utilisant les vraies images géophysiques du projet AI-MAP.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import warnings

# Ignorer les avertissements de torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# Ajouter le chemin du projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.model.geophysical_hybrid_net import ImageEncoder


class TestImageEncoderRealData(unittest.TestCase):
    """Tests pour la classe ImageEncoder avec données réelles."""
    
    def setUp(self):
        """Initialiser les tests avec des données réelles."""
        # Chemins vers les données réelles du projet
        self.data_root = Path("data")
        self.training_images = self.data_root / "training" / "images"
        
        # Images réelles de résistivité et chargeabilité
        self.resistivity_images = list((self.training_images / "resistivity").glob("*.JPG"))
        self.chargeability_images = list((self.training_images / "chargeability").glob("*.JPG"))
        self.chargeability_images.extend(list((self.training_images / "chargeability").glob("*.PNG")))
        
        # Vérifier que nous avons des données réelles
        self.assertGreater(len(self.resistivity_images), 0, "Aucune image de résistivité trouvée")
        self.assertGreater(len(self.chargeability_images), 0, "Aucune image de chargeabilité trouvée")
        
        # Utiliser les 3 premières images pour les tests
        self.test_image_paths = [str(img) for img in self.resistivity_images[:3]]
        
        # Créer un dossier temporaire
        self.temp_dir = tempfile.mkdtemp()
        
        # Paramètres de test
        self.batch_size = 2
        self.image_size = (64, 64)
        self.channels = 3
        
    def tearDown(self):
        """Nettoyer après les tests."""
        # Supprimer le dossier temporaire
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Charger et prétraiter une image pour les tests."""
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        # Redimensionner
        image = image.resize(self.image_size)
        
        # Convertir en tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # Normaliser et réorganiser les dimensions (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        return image_tensor
    
    def _create_batch_tensor(self) -> torch.Tensor:
        """Créer un batch de tenseurs d'images pour les tests."""
        batch_tensors = []
        
        for i in range(self.batch_size):
            # Utiliser une image réelle ou créer un tenseur simulé
            if i < len(self.test_image_paths):
                try:
                    img_tensor = self._load_and_preprocess_image(self.test_image_paths[i])
                    batch_tensors.append(img_tensor)
                except Exception:
                    # Fallback: créer un tenseur simulé
                    img_tensor = torch.randn(self.channels, *self.image_size)
                    batch_tensors.append(img_tensor)
            else:
                # Créer un tenseur simulé pour compléter le batch
                img_tensor = torch.randn(self.channels, *self.image_size)
                batch_tensors.append(img_tensor)
        
        # Concaténer en batch
        batch_tensor = torch.stack(batch_tensors, dim=0)
        return batch_tensor
    
    def test_init_resnet18_default(self):
        """Tester l'initialisation avec ResNet18 par défaut."""
        encoder = ImageEncoder()
        
        # Vérifications de base
        self.assertIsInstance(encoder, nn.Module)
        self.assertEqual(encoder.feature_dim, 512)
        self.assertIsInstance(encoder.backbone, nn.Module)
        
        # Vérifier que le backbone est un modèle ResNet (plus générique)
        self.assertIn("resnet", str(type(encoder.backbone)).lower())
        
        # Vérifier que la couche fc a été remplacée
        self.assertIsInstance(encoder.backbone.fc, nn.Sequential)
        
        # Vérifier la structure de la couche fc
        fc_layers = encoder.backbone.fc
        self.assertEqual(len(fc_layers), 4)  # Linear, ReLU, Dropout, BatchNorm
        self.assertIsInstance(fc_layers[0], nn.Linear)
        self.assertIsInstance(fc_layers[1], nn.ReLU)
        self.assertIsInstance(fc_layers[2], nn.Dropout)
        self.assertIsInstance(fc_layers[3], nn.BatchNorm1d)
        
        # Vérifier les dimensions
        self.assertEqual(fc_layers[0].in_features, 512)  # ResNet18 features
        self.assertEqual(fc_layers[0].out_features, 512)  # feature_dim par défaut
    
    def test_init_resnet34(self):
        """Tester l'initialisation avec ResNet34."""
        encoder = ImageEncoder(model_name="resnet34", feature_dim=256)
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, 256)
        self.assertIn("resnet", str(type(encoder.backbone)).lower())
        
        # Vérifier les dimensions de la couche fc
        fc_layer = encoder.backbone.fc[0]
        self.assertEqual(fc_layer.in_features, 512)  # ResNet34 features
        self.assertEqual(fc_layer.out_features, 256)  # feature_dim personnalisé
    
    def test_init_resnet50(self):
        """Tester l'initialisation avec ResNet50."""
        encoder = ImageEncoder(model_name="resnet50", feature_dim=1024)
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, 1024)
        self.assertIn("resnet", str(type(encoder.backbone)).lower())
        
        # Vérifier les dimensions de la couche fc
        fc_layer = encoder.backbone.fc[0]
        self.assertEqual(fc_layer.in_features, 2048)  # ResNet50 features
        self.assertEqual(fc_layer.out_features, 1024)  # feature_dim personnalisé
    
    def test_init_custom_feature_dim(self):
        """Tester l'initialisation avec une dimension de features personnalisée."""
        custom_dim = 128
        encoder = ImageEncoder(feature_dim=custom_dim)
        
        # Vérifications
        self.assertEqual(encoder.feature_dim, custom_dim)
        
        # Vérifier la couche de sortie
        fc_layer = encoder.backbone.fc[0]
        self.assertEqual(fc_layer.out_features, custom_dim)
    
    def test_init_freeze_backbone(self):
        """Tester l'initialisation avec le backbone gelé."""
        encoder = ImageEncoder(freeze_backbone=True)
        
        # Vérifier que les paramètres du backbone sont gelés (sauf fc)
        for name, param in encoder.backbone.named_parameters():
            if "fc" not in name:
                self.assertFalse(param.requires_grad)
        
        # Vérifier que les paramètres de la couche fc ne sont pas gelés
        for param in encoder.backbone.fc.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_init_invalid_model_name(self):
        """Tester l'initialisation avec un nom de modèle invalide."""
        with self.assertRaises(ValueError) as context:
            ImageEncoder(model_name="invalid_model")
        
        self.assertIn("Modèle non supporté", str(context.exception))
    
    def test_forward_basic(self):
        """Tester le forward pass de base."""
        encoder = ImageEncoder(feature_dim=256)
        batch_tensor = self._create_batch_tensor()
        
        # Mettre le modèle en mode eval pour éviter les problèmes de BatchNorm
        encoder.eval()
        
        # Forward pass
        with torch.no_grad():
            output = encoder(batch_tensor)
        
        # Vérifications de base
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dim(), 2)  # (batch_size, feature_dim)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 256)
        
        # Vérifier que l'output n'est pas NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_forward_different_batch_sizes(self):
        """Tester le forward pass avec différentes tailles de batch."""
        encoder = ImageEncoder(feature_dim=128)
        encoder.eval()  # Mode eval pour BatchNorm
        
        # Test avec batch_size = 2 (minimum pour BatchNorm)
        batch_tensor = torch.randn(2, self.channels, *self.image_size)
        with torch.no_grad():
            batch_output = encoder(batch_tensor)
        self.assertEqual(batch_output.shape, (2, 128))
        
        # Test avec batch_size = 4
        large_tensor = torch.randn(4, self.channels, *self.image_size)
        with torch.no_grad():
            large_output = encoder(large_tensor)
        self.assertEqual(large_output.shape, (4, 128))
    
    def test_forward_with_real_images(self):
        """Tester le forward pass avec de vraies images géophysiques."""
        encoder = ImageEncoder(feature_dim=256)
        encoder.eval()  # Mode eval pour BatchNorm
        
        # Charger une vraie image
        if self.test_image_paths:
            real_image = self._load_and_preprocess_image(self.test_image_paths[0])
            # Créer un batch de 2 images (minimum pour BatchNorm)
            real_batch = torch.stack([real_image, real_image], dim=0)
            
            # Forward pass
            with torch.no_grad():
                output = encoder(real_batch)
            
            # Vérifications
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (2, 256))
            self.assertFalse(torch.isnan(output).any())
            
            # Vérifier que l'output a des valeurs raisonnables
            self.assertTrue(torch.isfinite(output).all())
    
    def test_forward_gradient_flow(self):
        """Tester que le gradient peut circuler à travers le modèle."""
        encoder = ImageEncoder(feature_dim=128)
        batch_tensor = self._create_batch_tensor()
        
        # Activer le mode training
        encoder.train()
        
        # Forward pass
        output = encoder(batch_tensor)
        
        # Créer une loss factice
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients sont calculés
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
    
    def test_forward_with_frozen_backbone(self):
        """Tester le forward pass avec un backbone gelé."""
        encoder = ImageEncoder(freeze_backbone=True, feature_dim=64)
        batch_tensor = self._create_batch_tensor()
        
        # Forward pass
        encoder.eval()
        with torch.no_grad():
            output = encoder(batch_tensor)
        
        # Vérifications
        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertFalse(torch.isnan(output).any())
        
        # Vérifier que seules les couches fc ont des gradients
        encoder.train()
        
        # Créer un nouveau forward pass en mode train pour avoir des gradients
        train_output = encoder(batch_tensor)
        train_output.sum().backward()
        
        # Backbone devrait être gelé (sauf fc)
        for name, param in encoder.backbone.named_parameters():
            if "fc" not in name:
                self.assertFalse(param.requires_grad)
        
        # Couches fc devraient avoir des gradients
        for param in encoder.backbone.fc.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_model_consistency(self):
        """Tester la cohérence du modèle entre différents appels."""
        encoder = ImageEncoder(feature_dim=256)
        batch_tensor = self._create_batch_tensor()
        
        # Mettre en mode eval pour la cohérence
        encoder.eval()
        
        # Premier forward pass
        with torch.no_grad():
            output1 = encoder(batch_tensor)
        
        # Deuxième forward pass avec les mêmes données
        with torch.no_grad():
            output2 = encoder(batch_tensor)
        
        # Les outputs devraient être identiques (même modèle, même données)
        torch.testing.assert_close(output1, output2)
    
    def test_feature_extraction_quality(self):
        """Tester la qualité des features extraites."""
        encoder = ImageEncoder(feature_dim=128)
        batch_tensor = self._create_batch_tensor()
        
        # Forward pass en mode eval
        encoder.eval()
        with torch.no_grad():
            features = encoder(batch_tensor)
        
        # Vérifier que les features ne sont pas toutes identiques
        # (ce qui indiquerait un problème dans l'extraction)
        feature_variance = torch.var(features, dim=0)
        self.assertTrue(torch.any(feature_variance > 1e-6))
        
        # Vérifier que les features ont des valeurs raisonnables
        self.assertTrue(torch.all(torch.abs(features) < 100))  # Pas de valeurs extrêmes
    
    def test_model_parameters_count(self):
        """Tester le nombre de paramètres du modèle."""
        encoder = ImageEncoder(feature_dim=256)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        # Vérifier que le modèle a des paramètres
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        
        # Vérifier que tous les paramètres sont trainables par défaut
        self.assertEqual(total_params, trainable_params)
        
        # Avec backbone gelé
        encoder_frozen = ImageEncoder(freeze_backbone=True, feature_dim=256)
        total_params_frozen = sum(p.numel() for p in encoder_frozen.parameters())
        trainable_params_frozen = sum(p.numel() for p in encoder_frozen.parameters() if p.requires_grad)
        
        # Il devrait y avoir moins de paramètres trainables
        self.assertLess(trainable_params_frozen, total_params_frozen)
    
    def test_model_save_load(self):
        """Tester la sauvegarde et le chargement du modèle."""
        encoder = ImageEncoder(feature_dim=128)
        batch_tensor = self._create_batch_tensor()
        
        # Forward pass original en mode eval
        encoder.eval()
        with torch.no_grad():
            original_output = encoder(batch_tensor)
        
        # Sauvegarder le modèle
        save_path = os.path.join(self.temp_dir, "test_encoder.pth")
        torch.save(encoder.state_dict(), save_path)
        
        # Créer un nouveau modèle
        new_encoder = ImageEncoder(feature_dim=128)
        
        # Charger les poids
        new_encoder.load_state_dict(torch.load(save_path))
        
        # Forward pass avec le nouveau modèle en mode eval
        new_encoder.eval()
        with torch.no_grad():
            new_output = new_encoder(batch_tensor)
        
        # Les outputs devraient être identiques
        torch.testing.assert_close(original_output, new_output)
    
    def test_integration_with_real_data(self):
        """Test d'intégration avec des données réelles complètes."""
        encoder = ImageEncoder(feature_dim=256)
        encoder.eval()  # Mode eval pour BatchNorm
        
        # Charger plusieurs vraies images
        real_images = []
        for i, image_path in enumerate(self.test_image_paths[:2]):  # Utiliser 2 images
            try:
                img_tensor = self._load_and_preprocess_image(image_path)
                real_images.append(img_tensor)
            except Exception as e:
                # Fallback pour les images qui ne peuvent pas être chargées
                img_tensor = torch.randn(self.channels, *self.image_size)
                real_images.append(img_tensor)
        
        if real_images:
            # Créer un batch
            real_batch = torch.stack(real_images, dim=0)
            
            # Forward pass
            with torch.no_grad():
                features = encoder(real_batch)
            
            # Vérifications
            self.assertEqual(features.shape, (len(real_images), 256))
            self.assertFalse(torch.isnan(features).any())
            self.assertTrue(torch.isfinite(features).all())
            
            # Vérifier que les features sont différentes pour des images différentes
            if len(real_images) > 1:
                feature_diff = torch.norm(features[0] - features[1])
                self.assertGreater(feature_diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
