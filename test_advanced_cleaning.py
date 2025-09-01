#!/usr/bin/env python3
"""
Script de test pour les méthodes de nettoyage avancées de GeophysicalImageProcessor
================================================================================

Ce script teste toutes les nouvelles fonctionnalités de nettoyage d'images
ajoutées au processeur d'images géophysiques.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Ajouter le chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.image_processor import GeophysicalImageProcessor, ImageAugmenter


def create_test_image(size=(256, 256), noise_level=0.3, artifacts=True):
    """
    Créer une image de test avec du bruit et des artefacts.
    
    Args:
        size: Taille de l'image (width, height)
        noise_level: Niveau de bruit (0.0 à 1.0)
        artifacts: Ajouter des artefacts artificiels
    """
    # Créer une image de base avec des motifs géologiques
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Dessiner des couches géologiques
    colors = ['#8B4513', '#CD853F', '#D2B48C', '#F4A460', '#DEB887']
    for i in range(5):
        y = i * size[1] // 5
        draw.rectangle([0, y, size[0], y + size[1] // 5], fill=colors[i])
    
    # Ajouter des fractures
    for _ in range(8):
        x1 = np.random.randint(0, size[0])
        y1 = np.random.randint(0, size[1])
        x2 = x1 + np.random.randint(-50, 50)
        y2 = y1 + np.random.randint(-50, 50)
        draw.line([x1, y1, x2, y2], fill='black', width=2)
    
    # Ajouter du bruit
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy_array)
    
    # Ajouter des artefacts si demandé
    if artifacts:
        # Lignes de balayage
        draw = ImageDraw.Draw(img)
        for i in range(0, size[1], 20):
            draw.line([0, i, size[0], i], fill='gray', width=1)
        
        # Bruit sel-et-poivre
        img_array = np.array(img)
        mask = np.random.random(img_array.shape[:2]) < 0.02
        img_array[mask] = [0, 0, 0]  # Points noirs
        mask = np.random.random(img_array.shape[:2]) < 0.02
        img_array[mask] = [255, 255, 255]  # Points blancs
        img = Image.fromarray(img_array)
    
    return img


def test_noise_reduction_methods(processor, test_image):
    """Tester toutes les méthodes de réduction de bruit."""
    print("\n🔍 Test des méthodes de réduction de bruit...")
    
    methods = ["gaussian", "median", "bilateral", "wiener", "non_local_means"]
    results = {}
    
    for method in methods:
        try:
            print(f"  - Test de {method}...")
            cleaned = processor.apply_noise_reduction(test_image, method=method)
            results[method] = cleaned
            
            # Calculer la réduction de bruit
            original_std = np.std(np.array(test_image))
            cleaned_std = np.std(np.array(cleaned))
            reduction = (original_std - cleaned_std) / original_std * 100
            
            print(f"    ✅ {method}: Réduction de bruit de {reduction:.1f}%")
            
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")
            results[method] = None
    
    return results


def test_artifact_correction(processor, test_image):
    """Tester la correction d'artefacts."""
    print("\n🔧 Test de la correction d'artefacts...")
    
    artifacts = ["scan_lines", "salt_pepper", "streaking", "banding"]
    results = {}
    
    for artifact in artifacts:
        try:
            print(f"  - Correction de {artifact}...")
            corrected = processor.correct_artifacts(test_image, artifact)
            results[artifact] = corrected
            print(f"    ✅ {artifact}: Corrigé avec succès")
            
        except Exception as e:
            print(f"    ❌ {artifact}: Erreur - {e}")
            results[artifact] = None
    
    return results


def test_contrast_enhancement(processor, test_image):
    """Tester l'amélioration du contraste."""
    print("\n✨ Test de l'amélioration du contraste...")
    
    methods = ["histogram_equalization", "adaptive_histogram", "clahe", "gamma_correction"]
    results = {}
    
    for method in methods:
        try:
            print(f"  - Test de {method}...")
            enhanced = processor.enhance_contrast(test_image, method=method)
            results[method] = enhanced
            
            # Calculer l'amélioration du contraste
            original_std = np.std(np.array(test_image))
            enhanced_std = np.std(np.array(enhanced))
            improvement = (enhanced_std - original_std) / original_std * 100
            
            print(f"    ✅ {method}: Amélioration du contraste de {improvement:.1f}%")
            
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")
            results[method] = None
    
    return results


def test_geophysical_cleaning_pipeline(processor, test_image):
    """Tester le pipeline de nettoyage géophysique complet."""
    print("\n🚀 Test du pipeline de nettoyage géophysique complet...")
    
    # Définir les étapes de nettoyage
    cleaning_steps = [
        "noise_reduction",
        "scan_lines_removal", 
        "contrast_enhancement",
        "salt_pepper_removal"
    ]
    
    try:
        # Appliquer le pipeline complet
        cleaned_image = processor.apply_geophysical_specific_cleaning(test_image, cleaning_steps)
        
        # Obtenir un résumé des améliorations
        summary = processor.get_cleaning_summary(test_image, cleaning_steps)
        
        print("  ✅ Pipeline de nettoyage appliqué avec succès!")
        print(f"  📊 Résumé des améliorations:")
        print(f"    - Réduction de bruit: {summary['noise_reduction']:.2f}")
        print(f"    - Amélioration du contraste: {summary['contrast_improvement']:.2f}")
        print(f"    - Amélioration des gradients: {summary['gradient_enhancement']:.2f}")
        print(f"    - Méthodes appliquées: {', '.join(summary['cleaning_methods_applied'])}")
        
        return cleaned_image, summary
        
    except Exception as e:
        print(f"  ❌ Erreur lors du pipeline de nettoyage: {e}")
        return None, None


def visualize_results(original_image, results_dict, title_prefix="Résultats"):
    """Visualiser les résultats du nettoyage."""
    if not results_dict:
        return
    
    # Compter le nombre d'images valides
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return
    
    n_images = len(valid_results) + 1  # +1 pour l'image originale
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Afficher l'image originale
    axes[0].imshow(original_image)
    axes[0].set_title(f"{title_prefix} - Original")
    axes[0].axis('off')
    
    # Afficher les résultats
    for i, (method, result) in enumerate(valid_results.items(), 1):
        if i < len(axes):
            axes[i].imshow(result)
            axes[i].set_title(f"{title_prefix} - {method}")
            axes[i].axis('off')
    
    # Masquer les axes vides
    for i in range(len(valid_results) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale de test."""
    print("🧪 Test des méthodes de nettoyage avancées de GeophysicalImageProcessor")
    print("=" * 70)
    
    # Créer le processeur
    processor = GeophysicalImageProcessor(target_size=(256, 256), channels=3)
    print(f"✅ Processeur initialisé: {processor.target_size}, {processor.channels} canaux")
    
    # Créer une image de test
    print("\n🎨 Création d'une image de test avec bruit et artefacts...")
    test_image = create_test_image(size=(256, 256), noise_level=0.4, artifacts=True)
    print(f"✅ Image de test créée: {test_image.size}")
    
    # Sauvegarder l'image de test
    test_image.save("test_image_original.png")
    print("💾 Image de test sauvegardée: test_image_original.png")
    
    # Test 1: Réduction de bruit
    noise_results = test_noise_reduction_methods(processor, test_image)
    
    # Test 2: Correction d'artefacts
    artifact_results = test_artifact_correction(processor, test_image)
    
    # Test 3: Amélioration du contraste
    contrast_results = test_contrast_enhancement(processor, test_image)
    
    # Test 4: Pipeline complet
    cleaned_image, cleaning_summary = test_geophysical_cleaning_pipeline(processor, test_image)
    
    # Sauvegarder l'image nettoyée
    if cleaned_image:
        cleaned_image.save("test_image_cleaned.png")
        print("💾 Image nettoyée sauvegardée: test_image_cleaned.png")
    
    # Visualiser les résultats
    print("\n📊 Affichage des résultats...")
    
    # Réduction de bruit
    visualize_results(test_image, noise_results, "Réduction de Bruit")
    
    # Correction d'artefacts
    visualize_results(test_image, artifact_results, "Correction d'Artefacts")
    
    # Amélioration du contraste
    visualize_results(test_image, contrast_results, "Amélioration du Contraste")
    
    # Pipeline complet
    if cleaned_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(test_image)
        ax1.set_title("Image Originale (avec bruit et artefacts)")
        ax1.axis('off')
        
        ax2.imshow(cleaned_image)
        ax2.set_title("Image Nettoyée (pipeline complet)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\n🎉 Tests terminés avec succès!")
    print("\n📁 Fichiers générés:")
    print("  - test_image_original.png: Image de test originale")
    print("  - test_image_cleaned.png: Image après nettoyage complet")
    
    if cleaning_summary:
        print(f"\n📈 Résumé des améliorations:")
        print(f"  - Réduction de bruit: {cleaning_summary['noise_reduction']:.2f}")
        print(f"  - Amélioration du contraste: {cleaning_summary['contrast_improvement']:.2f}")
        print(f"  - Amélioration des gradients: {cleaning_summary['gradient_enhancement']:.2f}")


if __name__ == "__main__":
    main()
