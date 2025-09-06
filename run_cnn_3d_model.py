#!/usr/bin/env python3
"""
Script pour charger et utiliser le modèle CNN 3D sauvegardé (cnn_3d_model.pth).
Ce script montre comment charger le modèle et faire des prédictions.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def load_cnn_3d_model(model_path: str = "artifacts/models/cnn_3d_model.pth"):
    """
    Charger le modèle CNN 3D sauvegardé.
    
    Args:
        model_path: Chemin vers le fichier .pth
        
    Returns:
        Modèle chargé et prêt à l'utilisation
    """
    print(f"🔄 Chargement du modèle CNN 3D depuis: {model_path}")
    
    # Vérifier que le fichier existe
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Le fichier modèle n'existe pas: {model_path}")
    
    # Importer les classes nécessaires
    from src.model.geophysical_trainer import GeophysicalCNN3D
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Afficher les informations du checkpoint
    print(f"📊 Informations du modèle:")
    print(f"   - Époques d'entraînement: {checkpoint.get('epoch', 'N/A')}")
    
    # Afficher la loss si elle existe
    loss = checkpoint.get('loss', 'N/A')
    if isinstance(loss, (int, float)):
        print(f"   - Loss finale: {loss:.4f}")
    else:
        print(f"   - Loss finale: {loss}")
    
    # Afficher l'accuracy si elle existe
    accuracy = checkpoint.get('accuracy', 'N/A')
    if isinstance(accuracy, (int, float)):
        print(f"   - Accuracy finale: {accuracy:.4f}")
    else:
        print(f"   - Accuracy finale: {accuracy}")
    
    print(f"   - Architecture: CNN 3D")
    
    # Créer le modèle CNN 3D
    model = GeophysicalCNN3D(
        input_channels=4,
        num_classes=2
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Modèle CNN 3D chargé avec succès!")
    return model

def create_sample_3d_data():
    """
    Créer des données d'exemple pour tester le modèle CNN 3D.
    
    Returns:
        Données d'exemple au format (batch, channels, depth, height, width)
    """
    print("🔧 Création de données d'exemple 3D...")
    
    # Créer un volume 3D factice (4 canaux, 32x32x32)
    batch_size = 2
    channels = 4
    depth = 32
    height = 32
    width = 32
    
    # Générer des données aléatoires réalistes
    sample_data = np.random.randn(batch_size, channels, depth, height, width).astype(np.float32)
    
    print(f"📊 Données d'exemple créées:")
    print(f"   - Forme: {sample_data.shape}")
    print(f"   - Type: {sample_data.dtype}")
    print(f"   - Plage: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
    
    return sample_data

def load_real_3d_data():
    """
    Charger les données réelles pour le modèle CNN 3D.
    
    Returns:
        Données réelles au format (batch, channels, depth, height, width)
    """
    print("🌍 Chargement des données réelles 3D...")
    
    try:
        from src.data.data_processor import DataProcessor
        from src.model.geophysical_trainer import GeophysicalTrainer
        
        # Charger les données
        data_processor = DataProcessor()
        data_processor.load_data()
        
        # Créer le volume 3D
        volume_3d = data_processor.create_3d_volume()
        print(f"📊 Volume 3D créé: {volume_3d.shape}")
        
        # Convertir au format PyTorch (channels first)
        if len(volume_3d.shape) == 5:  # (batch, channels, depth, height, width)
            volume_tensor = torch.FloatTensor(volume_3d)
        elif len(volume_3d.shape) == 4:  # (channels, depth, height, width)
            volume_tensor = torch.FloatTensor(volume_3d).unsqueeze(0)  # Ajouter batch
        else:
            raise ValueError(f"Forme de volume 3D inattendue: {volume_3d.shape}")
        
        print(f"✅ Données réelles chargées: {volume_tensor.shape}")
        return volume_tensor.numpy()
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données réelles: {e}")
        print("🔄 Utilisation des données d'exemple...")
        return create_sample_3d_data()

def predict_with_model(model, input_data):
    """
    Faire des prédictions avec le modèle CNN 3D chargé.
    
    Args:
        model: Modèle CNN 3D chargé
        input_data: Données d'entrée (numpy array)
        
    Returns:
        Prédictions du modèle
    """
    print(f"🔮 Prédiction sur des données de forme: {input_data.shape}")
    
    # Convertir en tenseur PyTorch
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.FloatTensor(input_data)
    else:
        input_tensor = input_data
    
    # Vérifier que les données sont au bon format (batch, channels, depth, height, width)
    if len(input_tensor.shape) == 4:  # (channels, depth, height, width)
        input_tensor = input_tensor.unsqueeze(0)  # Ajouter dimension batch
    
    print(f"📊 Tenseur d'entrée: {input_tensor.shape}")
    
    # Faire la prédiction
    with torch.no_grad():
        predictions = model(input_tensor)
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
    parser = argparse.ArgumentParser(description='Exécuter le modèle CNN 3D sauvegardé')
    parser.add_argument('--model-path', default='artifacts/models/cnn_3d_model.pth',
                       help='Chemin vers le fichier modèle')
    parser.add_argument('--real-data', action='store_true',
                       help='Utiliser les données réelles au lieu des données d\'exemple')
    parser.add_argument('--verbose', action='store_true',
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    try:
        # Charger le modèle
        model = load_cnn_3d_model(args.model_path)
        
        # Charger les données
        if args.real_data:
            input_data = load_real_3d_data()
        else:
            input_data = create_sample_3d_data()
        
        # Faire des prédictions
        results = predict_with_model(model, input_data)
        
        print("\n🎉 Exécution du modèle CNN 3D terminée avec succès!")
        
        if args.verbose:
            print(f"\n📋 Résumé détaillé:")
            print(f"   - Modèle: CNN 3D")
            print(f"   - Données: {'Réelles' if args.real_data else 'Exemple'}")
            print(f"   - Forme d'entrée: {input_data.shape}")
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
