#!/usr/bin/env python3
"""
Démonstration simple des méthodes de nettoyage avancées
======================================================
"""

import sys
import os
import numpy as np
from PIL import Image

# Ajouter le chemin du projet
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from data.image_processor import GeophysicalImageProcessor


def create_simple_test_image():
    """Créer une image de test simple."""
    # Image 128x128 avec des motifs géologiques
    img = Image.new('RGB', (128, 128), color='white')
    
    # Ajouter des couches colorées
    for i in range(4):
        y = i * 32
        color = ['#8B4513', '#CD853F', '#D2B48C', '#F4A460'][i]
        img.paste(color, (0, y, 128, y + 32))
    
    # Ajouter du bruit
    img_array = np.array(img)
    noise = np.random.normal(0, 50, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array)


def main():
    """Démonstration principale."""
    print("🧪 Démonstration des méthodes de nettoyage avancées")
    print("=" * 50)
    
    # Créer le processeur
    processor = GeophysicalImageProcessor(target_size=(128, 128), channels=3)
    print(f"✅ Processeur créé: {processor.target_size}")
    
    # Créer une image de test
    test_image = create_simple_test_image()
    print("✅ Image de test créée")
    
    # Test des méthodes de nettoyage
    print("\n🔍 Test des méthodes de nettoyage:")
    
    # 1. Réduction de bruit
    print("  - Réduction de bruit gaussienne...")
    cleaned_gaussian = processor.apply_noise_reduction(test_image, method="gaussian")
    print("    ✅ Succès")
    
    # 2. Correction d'artefacts
    print("  - Correction du bruit sel-et-poivre...")
    cleaned_artifacts = processor.correct_artifacts(test_image, "salt_pepper")
    print("    ✅ Succès")
    
    # 3. Amélioration du contraste
    print("  - Amélioration du contraste...")
    enhanced_contrast = processor.enhance_contrast(test_image, method="clahe")
    print("    ✅ Succès")
    
    # 4. Pipeline complet
    print("  - Pipeline de nettoyage complet...")
    cleaning_steps = ["noise_reduction", "contrast_enhancement"]
    final_cleaned = processor.apply_geophysical_specific_cleaning(test_image, cleaning_steps)
    print("    ✅ Succès")
    
    # Sauvegarder les résultats
    test_image.save("test_original.png")
    cleaned_gaussian.save("test_gaussian_cleaned.png")
    cleaned_artifacts.save("test_artifacts_cleaned.png")
    enhanced_contrast.save("test_contrast_enhanced.png")
    final_cleaned.save("test_final_cleaned.png")
    
    print("\n💾 Images sauvegardées:")
    print("  - test_original.png")
    print("  - test_gaussian_cleaned.png")
    print("  - test_artifacts_cleaned.png")
    print("  - test_contrast_enhanced.png")
    print("  - test_final_cleaned.png")
    
    print("\n🎉 Démonstration terminée avec succès!")


if __name__ == "__main__":
    main()
