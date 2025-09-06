#!/usr/bin/env python3
"""
Script pour charger et utiliser le modèle CNN 2D sauvegardé (cnn_2d_model.pth).
Ce script montre comment charger le modèle et faire des prédictions.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def load_cnn_2d_model(model_path: str = "artifacts/models/cnn_2d_model.pth"):
    """
    Charger le modèle CNN 2D sauvegardé.
    
    Args:
        model_path: Chemin vers le fichier .pth
        
    Returns:
        Modèle chargé et prêt à l'utilisation
    """
    print(f"🔄 Chargement du modèle depuis: {model_path}")
    
    # Vérifier que le fichier existe
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Le fichier modèle n'existe pas: {model_path}")
    
    # Importer les classes nécessaires
    from src.model.geophysical_trainer import GeophysicalCNN2D, GeophysicalTrainer
    from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
    
    # Créer le modèle avec la même architecture que lors de l'entraînement
    model = GeophysicalCNN2D(
        input_channels=4,      # 4 canaux pour les dispositifs
        num_classes=2,         # 2 classes (binaire)
        grid_size=64,          # Taille de grille 64x64
        dropout_rate=0.3       # Taux de dropout
    )
    
    # Charger les poids du modèle
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    print("✅ Modèle CNN 2D chargé avec succès!")
    print(f"📊 Architecture: {model}")
    
    return model, checkpoint

def predict_with_model(model, input_data):
    """
    Faire des prédictions avec le modèle chargé.
    
    Args:
        model: Modèle CNN 2D chargé
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
    
    # Gérer le format des données selon la forme d'entrée
    if len(input_tensor.shape) == 3:
        # Format (height, width, channels) -> (channels, height, width)
        if input_tensor.shape[2] == 4:  # 4 canaux en dernière position
            input_tensor = input_tensor.permute(2, 0, 1)  # (4, 64, 64)
        input_tensor = input_tensor.unsqueeze(0)  # Ajouter dimension batch
    elif len(input_tensor.shape) == 4:
        # Format (batch, height, width, channels) -> (batch, channels, height, width)
        if input_tensor.shape[3] == 4:  # 4 canaux en dernière position
            input_tensor = input_tensor.permute(0, 3, 1, 2)  # (batch, 4, 64, 64)
    
    print(f"📊 Données formatées pour le modèle: {input_tensor.shape}")
    
    # Faire la prédiction
    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    print(f"📈 Prédictions: {predictions}")
    print(f"🎯 Classes prédites: {predicted_classes}")
    print(f"📊 Probabilités: {probabilities}")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'predicted_classes': predicted_classes
    }

def create_sample_data():
    """
    Créer des données d'exemple pour tester le modèle.
    
    Returns:
        Données d'exemple au format attendu par le modèle
    """
    print("📊 Création de données d'exemple...")
    
    # Créer des données factices avec la forme attendue: (channels, height, width)
    # Le modèle attend des données de forme (4, 64, 64)
    sample_data = np.random.rand(4, 64, 64).astype(np.float32)
    
    print(f"✅ Données d'exemple créées: {sample_data.shape}")
    return sample_data

def main():
    """
    Fonction principale pour exécuter le modèle CNN 2D.
    """
    print("🚀 EXÉCUTION DU MODÈLE CNN 2D")
    print("=" * 50)
    
    try:
        # 1. Charger le modèle
        model, checkpoint = load_cnn_2d_model()
        
        # 2. Afficher les informations du modèle
        print("\n📋 INFORMATIONS DU MODÈLE:")
        print("-" * 30)
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            print(f"📊 Époques d'entraînement: {len(history.get('epochs', []))}")
            if 'final_train_loss' in history:
                print(f"📉 Loss finale (train): {history['final_train_loss']:.4f}")
            if 'final_val_loss' in history:
                print(f"📉 Loss finale (val): {history['final_val_loss']:.4f}")
            if 'final_train_acc' in history:
                print(f"📈 Accuracy finale (train): {history['final_train_acc']:.2f}%")
            if 'final_val_acc' in history:
                print(f"📈 Accuracy finale (val): {history['final_val_acc']:.2f}%")
        
        # 3. Créer des données d'exemple
        print("\n🧪 TEST AVEC DES DONNÉES D'EXEMPLE:")
        print("-" * 40)
        sample_data = create_sample_data()
        
        # 4. Faire des prédictions
        results = predict_with_model(model, sample_data)
        
        # 5. Afficher les résultats
        print("\n🎯 RÉSULTATS:")
        print("-" * 20)
        print(f"✅ Prédiction réussie!")
        print(f"📊 Classes prédites: {results['predicted_classes'].numpy()}")
        print(f"📈 Probabilités: {results['probabilities'].numpy()}")
        
        print("\n" + "=" * 50)
        print("🎉 EXÉCUTION DU MODÈLE TERMINÉE AVEC SUCCÈS!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_and_predict_real_data():
    """
    Charger le modèle et faire des prédictions sur de vraies données.
    """
    print("🌍 CHARGEMENT ET PRÉDICTION SUR DONNÉES RÉELLES")
    print("=" * 60)
    
    try:
        # Charger le modèle
        model, checkpoint = load_cnn_2d_model()
        
        # Charger les données réelles du pipeline
        from main import phase2_data_processing
        
        print("📊 Chargement des données réelles...")
        processor, multi_device_tensor, volume_3d = phase2_data_processing()
        
        if multi_device_tensor is not None and len(multi_device_tensor) > 0:
            print(f"✅ Données réelles chargées: {multi_device_tensor.shape}")
            
            # Prendre quelques échantillons pour la prédiction
            sample_indices = np.random.choice(len(multi_device_tensor), min(3, len(multi_device_tensor)), replace=False)
            sample_data = multi_device_tensor[sample_indices]
            
            print(f"🔮 Prédiction sur {len(sample_data)} échantillons réels...")
            
            for i, sample in enumerate(sample_data):
                print(f"\n📊 Échantillon {i+1}:")
                results = predict_with_model(model, sample)
                
        else:
            print("⚠️  Aucune donnée réelle disponible, utilisation de données factices")
            sample_data = create_sample_data()
            results = predict_with_model(model, sample_data)
            
    except Exception as e:
        print(f"❌ Erreur avec les données réelles: {e}")
        print("🔄 Retour aux données factices...")
        model, _ = load_cnn_2d_model()
        sample_data = create_sample_data()
        results = predict_with_model(model, sample_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exécuter le modèle CNN 2D sauvegardé")
    parser.add_argument("--real-data", action="store_true", 
                       help="Utiliser les données réelles du pipeline")
    parser.add_argument("--model-path", type=str, 
                       default="artifacts/models/cnn_2d_model.pth",
                       help="Chemin vers le fichier modèle")
    
    args = parser.parse_args()
    
    if args.real_data:
        load_and_predict_real_data()
    else:
        main()
