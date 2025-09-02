"""
Modèle CNN Hybride pour Images + Données Géophysiques
======================================================

Ce module implémente un réseau de neurones hybride qui combine :
- Images géophysiques (résistivité, chargeabilité, etc.)
- Données géophysiques tabulaires (mesures, coordonnées, etc.)

Classes:
- GeophysicalHybridNet: Modèle principal hybride
- ImageEncoder: Encodeur d'images (basé sur ResNet)
- GeoDataEncoder: Encodeur de données géophysiques
- FusionModule: Module de fusion des features
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Encodeur d'images géophysiques basé sur ResNet.
    
    Cet encodeur extrait des features visuelles des images géophysiques
    en utilisant un réseau ResNet pré-entraîné.
    """
    
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, 
                 feature_dim: int = 512, freeze_backbone: bool = False):
        """
        Initialiser l'encodeur d'images.
        
        Args:
            model_name (str): Nom du modèle ResNet (resnet18, resnet34, resnet50)
            pretrained (bool): Utiliser des poids pré-entraînés
            feature_dim (int): Dimension des features de sortie
            freeze_backbone (bool): Geler les couches du backbone
        """
        super().__init__()
        
        # Sélectionner le modèle ResNet
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_features = 512
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_features = 512
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_features = 2048
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        # Geler le backbone si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Backbone {model_name} gelé")
        
        # Remplacer la dernière couche fully connected
        self.backbone.fc = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(feature_dim)
        )
        
        self.feature_dim = feature_dim
        logger.info(f"Encodeur d'images initialisé: {model_name}, features: {feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de l'encodeur d'images.
        
        Args:
            x (torch.Tensor): Images d'entrée (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Features d'images (batch_size, feature_dim)
        """
        return self.backbone(x)


class GeoDataEncoder(nn.Module):
    """
    Encodeur de données géophysiques tabulaires.
    
    Cet encodeur traite les données géophysiques numériques (résistivité,
    chargeabilité, coordonnées, etc.) et les convertit en features.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dims: Tuple[int, ...] = (64, 128, 256),
                 feature_dim: int = 256, dropout: float = 0.3):
        """
        Initialiser l'encodeur de données géophysiques.
        
        Args:
            input_dim (int): Dimension des données d'entrée
            hidden_dims (tuple): Dimensions des couches cachées
            feature_dim (int): Dimension des features de sortie
            dropout (float): Taux de dropout
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Construire les couches cachées
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.extend([
            nn.Linear(prev_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(feature_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
        self.feature_dim = feature_dim
        
        logger.info(f"Encodeur de données géophysiques initialisé: {input_dim} -> {feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de l'encodeur de données géophysiques.
        
        Args:
            x (torch.Tensor): Données géophysiques (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Features géophysiques (batch_size, feature_dim)
        """
        return self.encoder(x)


class FusionModule(nn.Module):
    """
    Module de fusion des features images et géophysiques.
    
    Ce module combine les features extraites des images et des données
    géophysiques pour la classification finale.
    """
    
    def __init__(self, image_features: int = 512, geo_features: int = 256,
                 hidden_dims: Tuple[int, ...] = (512, 256), num_classes: int = 2,
                 dropout: float = 0.5, fusion_method: str = "concatenation"):
        """
        Initialiser le module de fusion.
        
        Args:
            image_features (int): Dimension des features d'images
            geo_features (int): Dimension des features géophysiques
            hidden_dims (tuple): Dimensions des couches cachées
            num_classes (int): Nombre de classes de sortie
            dropout (float): Taux de dropout
            fusion_method (str): Méthode de fusion ("concatenation", "attention", "weighted")
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        self.image_features = image_features
        self.geo_features = geo_features
        
        if fusion_method == "concatenation":
            # Fusion par concaténation simple
            input_dim = image_features + geo_features
            self.fusion_layer = nn.Identity()
            
        elif fusion_method == "attention":
            # Fusion par attention
            self.attention = nn.MultiheadAttention(
                embed_dim=image_features, num_heads=8, batch_first=True
            )
            input_dim = image_features + geo_features
            
        elif fusion_method == "weighted":
            # Fusion pondérée
            self.image_weight = nn.Parameter(torch.tensor(0.5))
            self.geo_weight = nn.Parameter(torch.tensor(0.5))
            input_dim = max(image_features, geo_features)
            
        else:
            raise ValueError(f"Méthode de fusion non supportée: {fusion_method}")
        
        # Construire les couches de classification
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        logger.info(f"Module de fusion initialisé: {fusion_method}, classes: {num_classes}")
    
    def forward(self, image_features: torch.Tensor, geo_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du module de fusion.
        
        Args:
            image_features (torch.Tensor): Features d'images (batch_size, image_features)
            geo_features (torch.Tensor): Features géophysiques (batch_size, geo_features)
            
        Returns:
            torch.Tensor: Prédictions (batch_size, num_classes)
        """
        if self.fusion_method == "concatenation":
            # Concaténation simple
            combined_features = torch.cat([image_features, geo_features], dim=1)
            
        elif self.fusion_method == "attention":
            # Fusion par attention
            # Utiliser les features géophysiques comme query pour les features d'images
            geo_features_expanded = geo_features.unsqueeze(1)  # (batch_size, 1, geo_features)
            image_features_expanded = image_features.unsqueeze(1)  # (batch_size, 1, image_features)
            
            # Attention entre features géophysiques et images
            attended_features, _ = self.attention(
                geo_features_expanded, image_features_expanded, image_features_expanded
            )
            attended_features = attended_features.squeeze(1)  # (batch_size, image_features)
            
            combined_features = torch.cat([attended_features, geo_features], dim=1)
            
        elif self.fusion_method == "weighted":
            # Fusion pondérée
            # Redimensionner si nécessaire
            if image_features.shape[1] != geo_features.shape[1]:
                if image_features.shape[1] > geo_features.shape[1]:
                    # Étendre les features géophysiques
                    padding = torch.zeros(geo_features.shape[0], 
                                        image_features.shape[1] - geo_features.shape[1],
                                        device=geo_features.device)
                    geo_features = torch.cat([geo_features, padding], dim=1)
                else:
                    # Étendre les features d'images
                    padding = torch.zeros(image_features.shape[0], 
                                        geo_features.shape[1] - image_features.shape[1],
                                        device=image_features.device)
                    image_features = torch.cat([image_features, padding], dim=1)
            
            # Fusion pondérée
            combined_features = (self.image_weight * image_features + 
                               self.geo_weight * geo_features)
        
        # Classification finale
        output = self.classifier(combined_features)
        
        return output


class GeophysicalHybridNet(nn.Module):
    """
    Réseau hybride pour images + données géophysiques.
    
    Ce modèle combine un encodeur d'images (ResNet) et un encodeur de
    données géophysiques pour la classification géologique.
    """
    
    def __init__(self, num_classes: int = 2, image_model: str = "resnet18",
                 pretrained: bool = True, geo_input_dim: int = 4,
                 image_feature_dim: int = 512, geo_feature_dim: int = 256,
                 fusion_hidden_dims: Tuple[int, ...] = (512, 256),
                 dropout: float = 0.5, fusion_method: str = "concatenation",
                 freeze_backbone: bool = False):
        """
        Initialiser le modèle hybride.
        
        Args:
            num_classes (int): Nombre de classes de sortie
            image_model (str): Modèle ResNet pour les images
            pretrained (bool): Utiliser des poids pré-entraînés
            geo_input_dim (int): Dimension des données géophysiques
            image_feature_dim (int): Dimension des features d'images
            geo_feature_dim (int): Dimension des features géophysiques
            fusion_hidden_dims (tuple): Dimensions des couches de fusion
            dropout (float): Taux de dropout
            fusion_method (str): Méthode de fusion
            freeze_backbone (bool): Geler le backbone d'images
        """
        super().__init__()
        
        # Encodeur d'images
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            pretrained=pretrained,
            feature_dim=image_feature_dim,
            freeze_backbone=freeze_backbone
        )
        
        # Encodeur de données géophysiques
        self.geo_encoder = GeoDataEncoder(
            input_dim=geo_input_dim,
            feature_dim=geo_feature_dim,
            dropout=dropout
        )
        
        # Module de fusion
        self.fusion = FusionModule(
            image_features=image_feature_dim,
            geo_features=geo_feature_dim,
            hidden_dims=fusion_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
        # Paramètres du modèle
        self.num_classes = num_classes
        self.image_model = image_model
        self.fusion_method = fusion_method
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Modèle hybride initialisé:")
        logger.info(f"  - Classes: {num_classes}")
        logger.info(f"  - Modèle d'images: {image_model}")
        logger.info(f"  - Méthode de fusion: {fusion_method}")
        logger.info(f"  - Paramètres totaux: {total_params:,}")
        logger.info(f"  - Paramètres entraînables: {trainable_params:,}")
    
    def forward(self, images: torch.Tensor, geo_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle hybride.
        
        Args:
            images (torch.Tensor): Images d'entrée (batch_size, 3, height, width)
            geo_data (torch.Tensor): Données géophysiques (batch_size, geo_input_dim)
            
        Returns:
            torch.Tensor: Prédictions (batch_size, num_classes)
        """
        # Encoder les images
        image_features = self.image_encoder(images)
        
        # Encoder les données géophysiques
        geo_features = self.geo_encoder(geo_data)
        
        # Fusionner les features
        output = self.fusion(image_features, geo_features)
        
        return output
    
    def get_feature_maps(self, images: torch.Tensor, geo_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Obtenir les features intermédiaires pour l'analyse.
        
        Args:
            images (torch.Tensor): Images d'entrée
            geo_data (torch.Tensor): Données géophysiques
            
        Returns:
            dict: Dictionnaire des features intermédiaires
        """
        # Encoder les images
        image_features = self.image_encoder(images)
        
        # Encoder les données géophysiques
        geo_features = self.geo_encoder(geo_data)
        
        # Fusionner les features
        output = self.fusion(image_features, geo_features)
        
        return {
            'image_features': image_features,
            'geo_features': geo_features,
            'output': output
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Compter les paramètres du modèle.
        
        Returns:
            dict: Dictionnaire avec le nombre de paramètres
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params
        }


# Fonctions utilitaires
def create_hybrid_model(num_classes: int = 2, 
                       image_model: str = "resnet18",
                       geo_input_dim: int = 4,
                       **kwargs) -> GeophysicalHybridNet:
    """
    Créer un modèle hybride avec configuration par défaut.
    
    Args:
        num_classes (int): Nombre de classes
        image_model (str): Modèle ResNet
        geo_input_dim (int): Dimension des données géophysiques
        **kwargs: Arguments supplémentaires pour le modèle
        
    Returns:
        GeophysicalHybridNet: Modèle hybride configuré
    """
    return GeophysicalHybridNet(
        num_classes=num_classes,
        image_model=image_model,
        geo_input_dim=geo_input_dim,
        **kwargs
    )


def get_model_summary(model: GeophysicalHybridNet) -> Dict[str, Any]:
    """
    Obtenir un résumé détaillé du modèle.
    
    Args:
        model (GeophysicalHybridNet): Modèle à analyser
        
    Returns:
        dict: Résumé du modèle
    """
    param_counts = model.count_parameters()
    
    summary = {
        'model_type': 'GeophysicalHybridNet',
        'image_model': model.image_model,
        'fusion_method': model.fusion_method,
        'num_classes': model.num_classes,
        'parameters': param_counts,
        'architecture': {
            'image_encoder': str(model.image_encoder),
            'geo_encoder': str(model.geo_encoder),
            'fusion_module': str(model.fusion)
        }
    }
    
    return summary





if __name__ == "__main__":
    # Test du modèle
    model = GeophysicalHybridNet(
        num_classes=2,
        image_model="resnet18",
        geo_input_dim=4,
        fusion_method="concatenation"
    )
    
    # Créer des données de test
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 64, 64)
    test_geo_data = torch.randn(batch_size, 4)
    
    # Test du forward pass
    with torch.no_grad():
        output = model(test_images, test_geo_data)
        print(f"✅ Modèle hybride testé avec succès!")
        print(f"Entrée images: {test_images.shape}")
        print(f"Entrée données géo: {test_geo_data.shape}")
        print(f"Sortie: {output.shape}")
        
        # Résumé du modèle
        summary = get_model_summary(model)
        print(f"\n📊 Résumé du modèle:")
        print(f"  - Type: {summary['model_type']}")
        print(f"  - Modèle d'images: {summary['image_model']}")
        print(f"  - Méthode de fusion: {summary['fusion_method']}")
        print(f"  - Classes: {summary['num_classes']}")
        print(f"  - Paramètres totaux: {summary['parameters']['total_parameters']:,}")
