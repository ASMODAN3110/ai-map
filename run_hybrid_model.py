#!/usr/bin/env python3
"""
Script pour charger et utiliser le modèle hybride sauvegardé (hybrid_model.pth).
Ce script montre comment charger le modèle et faire des prédictions.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import cv2
from PIL import Image

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def load_hybrid_model(model_path: str = "artifacts/models/hybrid_model.pth"):
    """
    Charger le modèle hybride sauvegardé.
    
    Args:
        model_path: Chemin vers le fichier .pth
        
    Returns:
        Modèle chargé et prêt à l'utilisation
    """
    print(f"🔄 Chargement du modèle hybride depuis: {model_path}")
    
    # Vérifier que le fichier existe
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Le fichier modèle n'existe pas: {model_path}")
    
    # Importer les classes nécessaires
    from src.model.geophysical_hybrid_net import GeophysicalHybridNet
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Afficher les informations du checkpoint
    print(f"📊 Informations du modèle:")
    print(f"   - Architecture: Modèle Hybride (Images + Données Géophysiques)")
    print(f"   - Classes: 2")
    print(f"   - Modèle d'images: ResNet18")
    print(f"   - Méthode de fusion: Concatenation")
    
    # Créer le modèle hybride
    model = GeophysicalHybridNet(
        num_classes=2,
        image_model="resnet18",
        pretrained=False,  # Utiliser les poids sauvegardés
        geo_input_dim=4,
        image_feature_dim=512,
        geo_feature_dim=256,
        fusion_hidden_dims=(512, 256),
        dropout=0.5,
        fusion_method="concatenation",
        freeze_backbone=False
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✅ Modèle hybride chargé avec succès!")
    return model

def create_sample_data():
    """
    Créer des données d'exemple pour tester le modèle hybride.
    
    Returns:
        Tuple (image, geo_data) pour le modèle hybride
    """
    print("🔧 Création de données d'exemple...")
    
    # Créer une image factice (3 canaux, 64x64)
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Créer des données géophysiques factices (4 dimensions)
    geo_data = np.random.rand(4).astype(np.float32)
    
    print(f"📊 Données d'exemple créées:")
    print(f"   - Image: {image.shape}, type: {image.dtype}")
    print(f"   - Données géophysiques: {geo_data.shape}, type: {geo_data.dtype}")
    
    return image, geo_data

def load_real_data():
    """
    Charger les données réelles pour le modèle hybride.
    
    Returns:
        Tuple (image, geo_data) pour le modèle hybride
    """
    print("🌍 Chargement des données réelles...")
    
    try:
        # Charger une image réelle
        image_paths = [
            "data/raw/images/resistivity/resis1.JPG",
            "data/raw/images/chargeability/char_1.PNG"
        ]
        
        # Trouver une image existante
        image_path = None
        for path in image_paths:
            if Path(path).exists():
                image_path = path
                break
        
        if image_path is None:
            print("❌ Aucune image réelle trouvée, utilisation des données d'exemple")
            return create_sample_data()
        
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            # Essayer avec PIL
            image = np.array(Image.open(image_path))
        
        # Redimensionner à 64x64
        image = cv2.resize(image, (64, 64))
        
        # Convertir BGR vers RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Créer des données géophysiques factices
        geo_data = np.random.rand(4).astype(np.float32)
        
        print(f"✅ Données réelles chargées:")
        print(f"   - Image: {image_path} -> {image.shape}")
        print(f"   - Données géophysiques: {geo_data.shape}")
        
        return image, geo_data
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données réelles: {e}")
        print("🔄 Utilisation des données d'exemple...")
        return create_sample_data()

def preprocess_data(image, geo_data):
    """
    Préprocesser les données pour le modèle hybride.
    
    Args:
        image: Image (numpy array)
        geo_data: Données géophysiques (numpy array)
        
    Returns:
        Tuple (image_tensor, geo_tensor) prêts pour le modèle
    """
    # Préprocesser l'image
    # Normaliser les pixels [0, 255] -> [0, 1]
    image_normalized = image.astype(np.float32) / 255.0
    
    # Convertir en tenseur PyTorch et réorganiser (H, W, C) -> (C, H, W)
    image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1)
    
    # Ajouter une dimension batch
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 64, 64)
    
    # Préprocesser les données géophysiques
    geo_tensor = torch.FloatTensor(geo_data).unsqueeze(0)  # (1, 4)
    
    return image_tensor, geo_tensor

def predict_with_model(model, image, geo_data):
    """
    Faire des prédictions avec le modèle hybride chargé.
    
    Args:
        model: Modèle hybride chargé
        image: Image (numpy array)
        geo_data: Données géophysiques (numpy array)
        
    Returns:
        Prédictions du modèle
    """
    print(f"🔮 Prédiction sur des données hybrides...")
    
    # Préprocesser les données
    image_tensor, geo_tensor = preprocess_data(image, geo_data)
    
    print(f"📊 Données préprocessées:")
    print(f"   - Image: {image_tensor.shape}")
    print(f"   - Données géophysiques: {geo_tensor.shape}")
    
    # Faire la prédiction
    with torch.no_grad():
        predictions = model(image_tensor, geo_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    print(f"🎯 Prédictions:")
    print(f"   - Classes prédites: {predicted_classes.numpy()}")
    print(f"   - Probabilités: {probabilities.numpy()}")
    
    return {
        'predictions': predictions.numpy(),
        'probabilities': probabilities.numpy(),
        'predicted_classes': predicted_classes.numpy()
    }

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Exécuter le modèle hybride sauvegardé')
    parser.add_argument('--model-path', default='artifacts/models/hybrid_model.pth',
                       help='Chemin vers le fichier modèle')
    parser.add_argument('--real-data', action='store_true',
                       help='Utiliser les données réelles au lieu des données d\'exemple')
    parser.add_argument('--verbose', action='store_true',
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    try:
        # Charger le modèle
        model = load_hybrid_model(args.model_path)
        
        # Charger les données
        if args.real_data:
            image, geo_data = load_real_data()
        else:
            image, geo_data = create_sample_data()
        
        # Faire des prédictions
        results = predict_with_model(model, image, geo_data)
        
        print("\n🎉 Exécution du modèle hybride terminée avec succès!")
        
        if args.verbose:
            print(f"\n📋 Résumé détaillé:")
            print(f"   - Modèle: Hybride (Images + Données Géophysiques)")
            print(f"   - Données: {'Réelles' if args.real_data else 'Exemple'}")
            print(f"   - Image: {image.shape}")
            print(f"   - Données géophysiques: {geo_data.shape}")
            print(f"   - Prédictions: {results['predicted_classes']}")
            print(f"   - Probabilités: {results['probabilities']}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
