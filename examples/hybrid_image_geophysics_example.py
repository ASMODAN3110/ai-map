"""
Exemple Complet : Modèle Hybride Images + Données Géophysiques
==============================================================

Ce script démontre l'utilisation complète du système hybride qui combine :
- Images géophysiques (résistivité, chargeabilité, etc.)
- Données géophysiques tabulaires (mesures, coordonnées, etc.)

Fonctionnalités démontrées :
1. Chargement et traitement d'images
2. Préparation des données hybrides
3. Création et entraînement du modèle hybride
4. Évaluation et prédictions
5. Sauvegarde et chargement du modèle
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import shutil

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import des modules du projet
from data.image_processor import GeophysicalImageProcessor, ImageAugmenter
from model.geophysical_hybrid_net import GeophysicalHybridNet, create_hybrid_model
from model.geophysical_image_trainer import GeophysicalImageTrainer, create_hybrid_trainer
from preprocessor.data_augmenter import GeophysicalDataAugmenter


def create_sample_images(output_dir: str, num_images: int = 20) -> tuple:
    """
    Créer des images d'exemple pour la démonstration.
    
    Args:
        output_dir (str): Dossier de sortie
        num_images (int): Nombre d'images à créer
        
    Returns:
        tuple: (chemins des images, labels correspondants)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    labels = []
    
    print(f"🎨 Création de {num_images} images d'exemple...")
    
    for i in range(num_images):
        # Créer des images avec des patterns différents selon la classe
        if i < num_images // 2:
            # Classe 0: Images avec des lignes horizontales (stratification)
            img = Image.new('RGB', (128, 128), color='darkblue')
            pixels = img.load()
            
            # Ajouter des lignes horizontales
            for y in range(0, 128, 20):
                for x in range(128):
                    pixels[x, y] = (255, 255, 255)  # Lignes blanches
            
            label = 0
            filename = f"stratification_{i:02d}.png"
        else:
            # Classe 1: Images avec des lignes verticales (fractures)
            img = Image.new('RGB', (128, 128), color='darkred')
            pixels = img.load()
            
            # Ajouter des lignes verticales
            for x in range(0, 128, 20):
                for y in range(128):
                    pixels[x, y] = (255, 255, 255)  # Lignes blanches
            
            label = 1
            filename = f"fracture_{i:02d}.png"
        
        # Sauvegarder l'image
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        image_paths.append(filepath)
        labels.append(label)
    
    print(f"✅ {len(image_paths)} images créées dans {output_dir}")
    return image_paths, labels


def create_sample_geophysical_data(num_samples: int = 20) -> tuple:
    """
    Créer des données géophysiques d'exemple.
    
    Args:
        num_samples (int): Nombre d'échantillons
        
    Returns:
        tuple: (données géophysiques, labels)
    """
    print(f"📊 Création de {num_samples} échantillons de données géophysiques...")
    
    # Données simulées
    np.random.seed(42)
    
    geo_data = []
    labels = []
    
    for i in range(num_samples):
        if i < num_samples // 2:
            # Classe 0: Données de stratification
            resistivity = np.random.normal(150, 30)  # Ohm.m
            chargeability = np.random.normal(25, 5)   # mV/V
            sp_potential = np.random.normal(12, 3)   # mV
            coordinates = np.random.uniform(100, 200, 2)  # X, Y
            
            label = 0
        else:
            # Classe 1: Données de fracturation
            resistivity = np.random.normal(80, 20)   # Ohm.m
            chargeability = np.random.normal(40, 8)   # mV/V
            sp_potential = np.random.normal(25, 6)   # mV
            coordinates = np.random.uniform(200, 300, 2)  # X, Y
            
            label = 1
        
        # Créer le vecteur de features
        features = [resistivity, chargeability, sp_potential, coordinates[0], coordinates[1]]
        geo_data.append(features)
        labels.append(label)
    
    print(f"✅ {len(geo_data)} échantillons de données géophysiques créés")
    return geo_data, labels


def demonstrate_image_processing(image_paths: list, output_dir: str):
    """
    Démontrer le traitement d'images.
    
    Args:
        image_paths (list): Chemins des images
        output_dir (str): Dossier de sortie
    """
    print("\n🖼️  Démonstration du traitement d'images...")
    
    # Créer le processeur d'images
    processor = GeophysicalImageProcessor(target_size=(64, 64), channels=3)
    
    # Traiter quelques images
    sample_images = image_paths[:3]
    
    for i, image_path in enumerate(sample_images):
        print(f"  Traitement de l'image {i+1}/{len(sample_images)}...")
        
        # Traiter l'image
        tensor = processor.process_image(image_path)
        print(f"    Forme du tenseur: {tensor.shape}")
        
        # Extraire des features
        features = processor.extract_geophysical_features(image_path)
        print(f"    Intensité moyenne: {features['mean_intensity']:.2f}")
        print(f"    Magnitude du gradient: {features['gradient_magnitude']:.2f}")
        
        # Sauvegarder l'image prétraitée
        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_path = os.path.join(output_dir, processed_filename)
        processor.save_processed_image(image_path, processed_path)
        print(f"    Image prétraitée sauvegardée: {processed_filename}")
    
    print("✅ Traitement d'images terminé")


def demonstrate_image_augmentation(image_paths: list, output_dir: str):
    """
    Démontrer l'augmentation d'images.
    
    Args:
        image_paths (list): Chemins des images
        output_dir (str): Dossier de sortie
    """
    print("\n🔄 Démonstration de l'augmentation d'images...")
    
    # Créer l'augmenteur
    augmenter = ImageAugmenter(random_seed=42)
    
    # Sélectionner une image pour la démonstration
    demo_image_path = image_paths[0]
    demo_image = Image.open(demo_image_path)
    
    # Techniques d'augmentation
    augmentations = ["rotation", "flip_horizontal", "brightness", "contrast"]
    
    print(f"  Augmentation de l'image: {os.path.basename(demo_image_path)}")
    print(f"  Techniques: {', '.join(augmentations)}")
    
    # Générer des augmentations
    augmented_images = augmenter.augment_image(demo_image, augmentations, num_augmentations=2)
    
    print(f"  {len(augmented_images)} versions créées (original + augmentations)")
    
    # Sauvegarder les images augmentées
    for i, aug_image in enumerate(augmented_images):
        filename = f"augmented_{i:02d}_{os.path.basename(demo_image_path)}"
        filepath = os.path.join(output_dir, filename)
        aug_image.save(filepath)
    
    # Résumé des augmentations
    summary = augmenter.get_augmentation_summary()
    print(f"  Total d'augmentations effectuées: {summary['total_augmentations']}")
    print(f"  Types d'augmentation: {', '.join(summary['augmentation_types'])}")
    
    print("✅ Augmentation d'images terminée")


def demonstrate_hybrid_model_creation():
    """
    Démontrer la création du modèle hybride.
    
    Returns:
        GeophysicalHybridNet: Modèle créé
    """
    print("\n🧠 Démonstration de la création du modèle hybride...")
    
    # Créer le modèle
    model = create_hybrid_model(
        num_classes=2,
        image_model="resnet18",
        geo_input_dim=5,
        fusion_method="concatenation"
    )
    
    print(f"  Modèle créé: {model.__class__.__name__}")
    print(f"  Classes de sortie: {model.num_classes}")
    print(f"  Modèle d'images: {model.image_model}")
    print(f"  Méthode de fusion: {model.fusion_method}")
    
    # Compter les paramètres
    param_counts = model.count_parameters()
    print(f"  Paramètres totaux: {param_counts['total_parameters']:,}")
    print(f"  Paramètres entraînables: {param_counts['trainable_parameters']:,}")
    
    # Test du forward pass
    print("  Test du forward pass...")
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 64, 64)
    test_geo_data = torch.randn(batch_size, 5)
    
    with torch.no_grad():
        output = model(test_images, test_geo_data)
        print(f"    Entrée images: {test_images.shape}")
        print(f"    Entrée données géo: {test_geo_data.shape}")
        print(f"    Sortie: {output.shape}")
    
    print("✅ Modèle hybride créé avec succès")
    return model


def demonstrate_hybrid_training(image_paths: list, geo_data: list, labels: list, 
                              output_dir: str, model: GeophysicalHybridNet):
    """
    Démontrer l'entraînement du modèle hybride.
    
    Args:
        image_paths (list): Chemins des images
        geo_data (list): Données géophysiques
        labels (list): Labels
        output_dir (str): Dossier de sortie
        model (GeophysicalHybridNet): Modèle à entraîner
        
    Returns:
        dict: Historique d'entraînement
    """
    print("\n🚀 Démonstration de l'entraînement du modèle hybride...")
    
    # Créer le trainer
    augmenter = GeophysicalDataAugmenter()
    trainer = create_hybrid_trainer(augmenter)
    
    print(f"  Trainer créé: {trainer.__class__.__name__}")
    print(f"  Device: {trainer.device}")
    
    # Préparer les données hybrides
    print("  Préparation des données hybrides...")
    train_loader, val_loader = trainer.prepare_hybrid_data(
        image_paths, geo_data, labels,
        test_size=0.3,
        augmentations=["rotation", "flip_horizontal"],
        num_augmentations=1
    )
    
    print(f"  Données d'entraînement: {len(train_loader.dataset)} échantillons")
    print(f"  Données de validation: {len(val_loader.dataset)} échantillons")
    
    # Entraîner le modèle (version courte pour la démonstration)
    print("  Début de l'entraînement...")
    history = trainer.train_hybrid_model(
        model, train_loader, val_loader,
        num_epochs=5,  # Court pour la démonstration
        learning_rate=0.001,
        patience=3
    )
    
    print("✅ Entraînement terminé")
    return history


def demonstrate_model_evaluation(trainer: GeophysicalImageTrainer, 
                               image_paths: list, geo_data: list, labels: list):
    """
    Démontrer l'évaluation du modèle.
    
    Args:
        trainer (GeophysicalImageTrainer): Trainer
        image_paths (list): Chemins des images
        geo_data (list): Données géophysiques
        labels (list): Labels
    """
    print("\n📊 Démonstration de l'évaluation du modèle...")
    
    # Préparer les données de test
    from sklearn.model_selection import train_test_split
    
    (X_img_train, X_img_test, 
     X_geo_train, X_geo_test, 
     y_train, y_test) = train_test_split(
        image_paths, geo_data, labels,
        test_size=0.3, random_state=42, stratify=labels
    )
    
    # Créer le loader de test
    test_loader = trainer.prepare_hybrid_data(
        X_img_test, X_geo_test, y_test,
        test_size=0.0  # Pas de split train/val pour le test
    )[1]  # Prendre le deuxième loader (validation)
    
    print(f"  Données de test: {len(test_loader.dataset)} échantillons")
    
    # Évaluer le modèle
    metrics = trainer.evaluate_hybrid_model(trainer.model, test_loader)
    
    print("  Résultats de l'évaluation:")
    print(f"    Test Loss: {metrics['test_loss']:.4f}")
    print(f"    Test Accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"    Échantillons corrects: {metrics['correct_predictions']}/{metrics['total_samples']}")
    
    print("✅ Évaluation terminée")


def demonstrate_model_saving_and_loading(model: GeophysicalHybridNet, output_dir: str):
    """
    Démontrer la sauvegarde et le chargement du modèle.
    
    Args:
        model (GeophysicalHybridNet): Modèle à sauvegarder
        output_dir (str): Dossier de sortie
    """
    print("\n💾 Démonstration de la sauvegarde et du chargement du modèle...")
    
    # Chemin de sauvegarde
    model_path = os.path.join(output_dir, "hybrid_model_example.pth")
    
    # Sauvegarder le modèle
    print(f"  Sauvegarde du modèle dans: {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': model.num_classes,
            'image_model': model.image_model,
            'fusion_method': model.fusion_method
        },
        'timestamp': '2024-01-01'
    }, model_path)
    
    # Charger le modèle
    print("  Chargement du modèle...")
    checkpoint = torch.load(model_path)
    
    # Créer un nouveau modèle avec la même configuration
    loaded_model = create_hybrid_model(
        num_classes=checkpoint['model_config']['num_classes'],
        image_model=checkpoint['model_config']['image_model'],
        fusion_method=checkpoint['model_config']['fusion_method']
    )
    
    # Charger les poids
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("  Vérification du modèle chargé...")
    
    # Test du modèle chargé
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 64, 64)
    test_geo_data = torch.randn(batch_size, 5)
    
    with torch.no_grad():
        original_output = model(test_images, test_geo_data)
        loaded_output = loaded_model(test_images, test_geo_data)
        
        # Vérifier que les sorties sont identiques
        output_diff = torch.abs(original_output - loaded_output).max()
        print(f"    Différence maximale entre modèles: {output_diff:.6f}")
        
        if output_diff < 1e-6:
            print("    ✅ Modèle chargé avec succès (sorties identiques)")
        else:
            print("    ⚠️  Différences détectées dans le modèle chargé")
    
    print("✅ Sauvegarde et chargement terminés")


def create_demo_report(output_dir: str, image_paths: list, geo_data: list, labels: list):
    """
    Créer un rapport de démonstration.
    
    Args:
        output_dir (str): Dossier de sortie
        image_paths (list): Chemins des images
        geo_data (list): Données géophysiques
        labels (list): Labels
    """
    print("\n📋 Création du rapport de démonstration...")
    
    report_path = os.path.join(output_dir, "demo_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE DÉMONSTRATION - MODÈLE HYBRIDE IMAGES + DONNÉES GÉOPHYSIQUES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DONNÉES UTILISÉES\n")
        f.write("-" * 30 + "\n")
        f.write(f"Nombre total d'images: {len(image_paths)}\n")
        f.write(f"Nombre total d'échantillons géophysiques: {len(geo_data)}\n")
        f.write(f"Nombre de classes: {len(set(labels))}\n")
        f.write(f"Distribution des classes: {dict(zip(*np.unique(labels, return_counts=True)))}\n\n")
        
        f.write("2. IMAGES CRÉÉES\n")
        f.write("-" * 20 + "\n")
        for i, path in enumerate(image_paths):
            filename = os.path.basename(path)
            label = labels[i]
            f.write(f"  {i+1:2d}. {filename} -> Classe {label}\n")
        f.write("\n")
        
        f.write("3. DONNÉES GÉOPHYSIQUES\n")
        f.write("-" * 30 + "\n")
        f.write("Features: [Résistivité, Chargeabilité, Potentiel SP, Coordonnée X, Coordonnée Y]\n")
        f.write("Unités: [Ohm.m, mV/V, mV, m, m]\n\n")
        
        f.write("4. FONCTIONNALITÉS DÉMONTRÉES\n")
        f.write("-" * 40 + "\n")
        f.write("✅ Traitement d'images géophysiques\n")
        f.write("✅ Augmentation d'images\n")
        f.write("✅ Création de modèle hybride\n")
        f.write("✅ Entraînement du modèle\n")
        f.write("✅ Évaluation et métriques\n")
        f.write("✅ Sauvegarde et chargement\n")
        f.write("✅ Rapport de démonstration\n\n")
        
        f.write("5. FICHIERS GÉNÉRÉS\n")
        f.write("-" * 25 + "\n")
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                f.write(f"  {filename} ({size:,} bytes)\n")
    
    print(f"  Rapport créé: {report_path}")
    print("✅ Rapport de démonstration terminé")


def main():
    """Fonction principale de démonstration."""
    print("🚀 DÉMONSTRATION COMPLÈTE - MODÈLE HYBRIDE IMAGES + DONNÉES GÉOPHYSIQUES")
    print("=" * 80)
    
    # Créer le dossier de sortie
    output_dir = "artifacts/demo_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Créer les données d'exemple
        print("\n📁 ÉTAPE 1: Création des données d'exemple")
        image_paths, image_labels = create_sample_images(output_dir, num_images=20)
        geo_data, geo_labels = create_sample_geophysical_data(num_samples=20)
        
        # Vérifier que les labels correspondent
        assert image_labels == geo_labels, "Les labels des images et données géo doivent correspondre"
        labels = image_labels
        
        # 2. Démontrer le traitement d'images
        print("\n📁 ÉTAPE 2: Traitement d'images")
        demonstrate_image_processing(image_paths, output_dir)
        
        # 3. Démontrer l'augmentation d'images
        print("\n📁 ÉTAPE 3: Augmentation d'images")
        demonstrate_image_augmentation(image_paths, output_dir)
        
        # 4. Créer le modèle hybride
        print("\n📁 ÉTAPE 4: Création du modèle hybride")
        model = demonstrate_hybrid_model_creation()
        
        # 5. Entraîner le modèle
        print("\n📁 ÉTAPE 5: Entraînement du modèle")
        history = demonstrate_hybrid_training(image_paths, geo_data, labels, output_dir, model)
        
        # 6. Évaluer le modèle
        print("\n📁 ÉTAPE 6: Évaluation du modèle")
        augmenter = GeophysicalDataAugmenter()
        trainer = create_hybrid_trainer(augmenter)
        trainer.model = model  # Assigner le modèle entraîné
        demonstrate_model_evaluation(trainer, image_paths, geo_data, labels)
        
        # 7. Sauvegarder et charger le modèle
        print("\n📁 ÉTAPE 7: Sauvegarde et chargement")
        demonstrate_model_saving_and_loading(model, output_dir)
        
        # 8. Créer le rapport
        print("\n📁 ÉTAPE 8: Rapport de démonstration")
        create_demo_report(output_dir, image_paths, geo_data, labels)
        
        print("\n🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Tous les fichiers ont été créés dans: {output_dir}")
        print("\n📋 Résumé des fonctionnalités testées:")
        print("  ✅ Traitement d'images géophysiques")
        print("  ✅ Augmentation d'images")
        print("  ✅ Modèle hybride CNN + données géophysiques")
        print("  ✅ Entraînement complet")
        print("  ✅ Évaluation et métriques")
        print("  ✅ Sauvegarde/chargement")
        print("  ✅ Pipeline end-to-end")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Vérifier que PyTorch est disponible
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} détecté")
    except ImportError:
        print("❌ PyTorch n'est pas installé. Installez-le avec: pip install torch torchvision")
        sys.exit(1)
    
    # Lancer la démonstration
    success = main()
    
    if success:
        print("\n🎯 Prochaines étapes suggérées:")
        print("  1. Tester avec vos propres données géophysiques")
        print("  2. Ajuster les hyperparamètres du modèle")
        print("  3. Expérimenter avec différentes méthodes de fusion")
        print("  4. Intégrer dans votre pipeline de production")
        print("  5. Ajouter des métriques d'évaluation personnalisées")
    else:
        print("\n🔧 Pour résoudre les problèmes:")
        print("  1. Vérifiez que toutes les dépendances sont installées")
        print("  2. Assurez-vous que les chemins sont corrects")
        print("  3. Vérifiez les permissions d'écriture")
        print("  4. Consultez les logs d'erreur détaillés")
