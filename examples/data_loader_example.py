#!/usr/bin/env python3
"""
Exemple d'utilisation des données organisées
==========================================

Ce script montre comment charger et utiliser les données
organisées pour l'entraînement et les tests.
"""

import os
from pathlib import Path
from src.data.image_processor import ImageAugmenter, GeophysicalImageProcessor

def load_training_data():
    """Charger les données d'entraînement."""
    train_path = Path("data/training/images")
    
    # Images de résistivité
    resistivity_files = list(train_path.glob("resistivity/*.JPG"))
    print(f"Images de résistivité d'entraînement: {len(resistivity_files)}")
    
    # Images de chargeabilité
    chargeability_files = list(train_path.glob("chargeability/*.JPG")) + \
                         list(train_path.glob("chargeability/*.PNG"))
    print(f"Images de chargeabilité d'entraînement: {len(chargeability_files)}")
    
    # Images de profils
    profile_files = list(train_path.glob("profiles/*.JPG"))
    print(f"Images de profils d'entraînement: {len(profile_files)}")
    
    return resistivity_files, chargeability_files, profile_files

def load_test_data():
    """Charger les données de test."""
    test_path = Path("data/test/images")
    
    # Images de résistivité
    resistivity_files = list(test_path.glob("resistivity/*.JPG"))
    print(f"Images de résistivité de test: {len(resistivity_files)}")
    
    # Images de chargeabilité
    chargeability_files = list(test_path.glob("chargeability/*.JPG")) + \
                         list(test_path.glob("chargeability/*.PNG"))
    print(f"Images de chargeabilité de test: {len(chargeability_files)}")
    
    # Images de profils
    profile_files = list(test_path.glob("profiles/*.JPG"))
    print(f"Images de profils de test: {len(profile_files)}")
    
    return resistivity_files, chargeability_files, profile_files

def demonstrate_augmentation():
    """Démontrer l'augmentation des données d'entraînement."""
    print("\n🚀 Démonstration de l'augmentation des données...")
    
    # Initialiser l'augmenteur
    augmenter = ImageAugmenter(random_seed=42)
    
    # Charger une image d'exemple
    train_path = Path("data/training/images/resistivity")
    example_image = next(train_path.glob("*.JPG"))
    
    print(f"Image d'exemple: {example_image.name}")
    
    # Augmenter l'image
    from PIL import Image
    image = Image.open(example_image)
    
    augmented_images = augmenter.augment_image(
        image,
        ["rotation", "flip_horizontal", "gaussian_noise", "elastic_deformation"],
        num_augmentations=3
    )
    
    print(f"Images augmentées créées: {len(augmented_images)}")
    
    return augmented_images

def demonstrate_image_processing():
    """Démontrer le traitement d'images géophysiques."""
    print("\n🔧 Démonstration du traitement d'images...")
    
    # Initialiser le processeur
    processor = GeophysicalImageProcessor(target_size=(64, 64), channels=3)
    
    # Charger une image d'exemple
    train_path = Path("data/training/images/resistivity")
    example_image = next(train_path.glob("*.JPG"))
    
    print(f"Traitement de l'image: {example_image.name}")
    
    try:
        # Traiter l'image
        tensor = processor.process_image(str(example_image))
        print(f"Image traitée avec succès: {tensor.shape}")
        
        # Sauvegarder l'image traitée
        output_path = Path("data/processed") / "example_processed.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir le tenseur en image
        import torchvision.utils as vutils
        vutils.save_image(tensor, str(output_path))
        print(f"Image traitée sauvegardée: {output_path}")
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
    
    return processor

def create_training_dataset():
    """Créer un dataset d'entraînement avec augmentation."""
    print("\n📊 Création du dataset d'entraînement...")
    
    # Charger les données d'entraînement
    train_res, train_char, train_prof = load_training_data()
    
    # Initialiser l'augmenteur
    augmenter = ImageAugmenter(random_seed=42)
    
    # Techniques d'augmentation pour images géophysiques
    geophysical_augmentations = [
        "rotation", "flip_horizontal", "gaussian_noise",
        "elastic_deformation", "geological_stratification"
    ]
    
    print(f"Dataset d'entraînement créé:")
    print(f"  - Images de résistivité: {len(train_res)}")
    print(f"  - Images de chargeabilité: {len(train_char)}")
    print(f"  - Images de profils: {len(train_prof)}")
    print(f"  - Techniques d'augmentation: {len(geophysical_augmentations)}")
    
    return {
        'resistivity': train_res,
        'chargeability': train_char,
        'profiles': train_prof,
        'augmentations': geophysical_augmentations,
        'augmenter': augmenter
    }

def main():
    """Fonction principale de démonstration."""
    print("📊 Chargement des données organisées...")
    print("=" * 50)
    
    # Charger les données d'entraînement
    train_res, train_char, train_prof = load_training_data()
    
    # Charger les données de test
    test_res, test_char, test_prof = load_test_data()
    
    print("\n📈 Résumé des données:")
    print(f"  Entraînement: {len(train_res) + len(train_char) + len(train_prof)} images")
    print(f"  Test: {len(test_res) + len(test_char) + len(test_prof)} images")
    
    # Démontrer l'augmentation
    augmented = demonstrate_augmentation()
    
    # Démontrer le traitement
    processor = demonstrate_image_processing()
    
    # Créer le dataset d'entraînement
    dataset = create_training_dataset()
    
    print("\n✅ Exemple d'utilisation terminé!")
    print("\n💡 Prochaines étapes:")
    print("  1. Utiliser ces données pour entraîner votre modèle CNN")
    print("  2. Appliquer l'augmentation pour enrichir le dataset")
    print("  3. Utiliser les données de test pour évaluer les performances")
    print("  4. Intégrer dans votre pipeline d'entraînement")

if __name__ == "__main__":
    main()
