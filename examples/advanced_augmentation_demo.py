#!/usr/bin/env python3
"""
Démonstration des techniques d'augmentation avancées pour images géophysiques
============================================================================

Ce script démontre toutes les nouvelles fonctionnalités d'augmentation
ajoutées à l'ImageAugmenter pour les images géophysiques.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Ajouter le chemin du projet
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from data.image_processor import ImageAugmenter


def create_geophysical_test_image(size=(256, 256)):
    """
    Créer une image de test géophysique réaliste.
    
    Args:
        size: Taille de l'image (width, height)
    
    Returns:
        Image PIL avec motifs géologiques
    """
    # Créer une image de base
    img = Image.new('RGB', size, color='#F5F5DC')  # Couleur beige clair
    draw = ImageDraw.Draw(img)
    
    # Couches géologiques (strates)
    geological_layers = [
        {'color': '#8B4513', 'thickness': 0.15, 'name': 'Argile'},
        {'color': '#CD853F', 'thickness': 0.12, 'name': 'Sable'},
        {'color': '#D2B48C', 'thickness': 0.18, 'name': 'Calcaire'},
        {'color': '#F4A460', 'thickness': 0.20, 'name': 'Grès'},
        {'color': '#DEB887', 'thickness': 0.15, 'name': 'Schiste'},
        {'color': '#BC8F8F', 'thickness': 0.20, 'name': 'Granite'}
    ]
    
    current_y = 0
    for layer in geological_layers:
        layer_height = int(size[1] * layer['thickness'])
        draw.rectangle([0, current_y, size[0], current_y + layer_height], 
                      fill=layer['color'])
        current_y += layer_height
    
    # Ajouter des fractures et failles
    fractures = [
        {'start': (100, 50), 'end': (150, 200), 'width': 3},
        {'start': (300, 100), 'end': (350, 300), 'width': 2},
        {'start': (450, 150), 'end': (500, 400), 'width': 4},
        {'start': (200, 300), 'end': (250, 450), 'width': 2},
        {'start': (400, 350), 'end': (450, 500), 'width': 3}
    ]
    
    for fracture in fractures:
        draw.line([fracture['start'], fracture['end']], 
                 fill='black', width=fracture['width'])
    
    # Ajouter des plis géologiques
    for i in range(3):
        center_x = 100 + i * 150
        center_y = 200 + i * 50
        
        # Dessiner des courbes pour simuler des plis
        points = []
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            x = center_x + 30 * np.cos(rad)
            y = center_y + 20 * np.sin(rad)
            points.append((x, y))
        
        if len(points) > 2:
            draw.polygon(points, fill='#A0522D', outline='#8B4513')
    
    # Ajouter des inclusions minérales
    for _ in range(20):
        x = np.random.randint(50, size[0] - 50)
        y = np.random.randint(50, size[1] - 50)
        radius = np.random.randint(5, 15)
        color = np.random.choice(['#FFD700', '#C0C0C0', '#CD7F32', '#FF6B6B'])
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return img


def demonstrate_basic_augmentations(augmenter, test_image):
    """Démontrer les techniques d'augmentation de base."""
    print("\n🔧 Techniques d'augmentation de base...")
    
    basic_techniques = [
        "rotation", "flip_horizontal", "flip_vertical", 
        "brightness", "contrast"
    ]
    
    results = {}
    for technique in basic_techniques:
        try:
            augmented = augmenter.augment_image(test_image, [technique])
            results[technique] = augmented[1]  # Image augmentée (pas l'originale)
            print(f"  ✅ {technique}: Succès")
        except Exception as e:
            print(f"  ❌ {technique}: Erreur - {e}")
            results[technique] = None
    
    return results


def demonstrate_advanced_augmentations(augmenter, test_image):
    """Démontrer les techniques d'augmentation avancées."""
    print("\n🚀 Techniques d'augmentation avancées...")
    
    advanced_techniques = [
        "elastic_deformation", "color_jittering", "gaussian_noise",
        "blur_sharpen", "perspective_transform", "cutout"
    ]
    
    results = {}
    for technique in advanced_techniques:
        try:
            augmented = augmenter.augment_image(test_image, [technique])
            results[technique] = augmented[1]  # Image augmentée
            print(f"  ✅ {technique}: Succès")
        except Exception as e:
            print(f"  ❌ {technique}: Erreur - {e}")
            results[technique] = None
    
    return results


def demonstrate_geophysical_augmentations(augmenter, test_image):
    """Démontrer les techniques spécifiques aux images géophysiques."""
    print("\n🌍 Techniques d'augmentation géophysiques...")
    
    geophysical_techniques = [
        "geological_stratification", "fracture_patterns", 
        "mineral_inclusions", "weathering_effects", "sedimentary_layers"
    ]
    
    results = {}
    for technique in geophysical_techniques:
        try:
            augmented = augmenter.augment_image(test_image, [technique])
            results[technique] = augmented[1]  # Image augmentée
            print(f"  ✅ {technique}: Succès")
        except Exception as e:
            print(f"  ❌ {technique}: Erreur - {e}")
            results[technique] = None
    
    return results


def demonstrate_combined_augmentations(augmenter, test_image):
    """Démontrer des combinaisons d'augmentations."""
    print("\n🎯 Combinaisons d'augmentations...")
    
    # Pipeline d'augmentation géophysique complet
    geophysical_pipeline = [
        "elastic_deformation",      # Plis géologiques
        "geological_stratification", # Couches
        "fracture_patterns",        # Fractures
        "mineral_inclusions",       # Inclusions
        "weathering_effects"        # Altération
    ]
    
    try:
        augmented = augmenter.augment_image(test_image, geophysical_pipeline)
        print(f"  ✅ Pipeline géophysique complet: {len(augmented)} images créées")
        return augmented
    except Exception as e:
        print(f"  ❌ Pipeline géophysique: Erreur - {e}")
        return None


def create_augmentation_visualization(original_image, results_dict, title_prefix="Augmentation"):
    """Créer une visualisation des résultats d'augmentation."""
    if not results_dict:
        return
    
    # Compter le nombre d'images valides
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return
    
    n_images = len(valid_results) + 1  # +1 pour l'image originale
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Afficher l'image originale
    axes[0].imshow(original_image)
    axes[0].set_title(f"{title_prefix} - Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Afficher les résultats
    for i, (technique, result) in enumerate(valid_results.items(), 1):
        if i < len(axes):
            axes[i].imshow(result)
            axes[i].set_title(f"{title_prefix} - {technique}", fontsize=10)
            axes[i].axis('off')
    
    # Masquer les axes vides
    for i in range(len(valid_results) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def save_augmentation_results(results_dict, output_dir="augmented_images"):
    """Sauvegarder toutes les images augmentées."""
    print(f"\n💾 Sauvegarde des images augmentées dans '{output_dir}'...")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    for technique, image in results_dict.items():
        if image is not None:
            filename = f"augmented_{technique}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_files.append(filename)
            print(f"  ✅ {technique}: {filename}")
    
    print(f"  📁 {len(saved_files)} images sauvegardées")
    return saved_files


def main():
    """Fonction principale de démonstration."""
    print("🎨 Démonstration des techniques d'augmentation avancées pour images géophysiques")
    print("=" * 80)
    
    # 1. Initialiser l'augmenteur
    print("\n🔧 Initialisation de l'ImageAugmenter...")
    augmenter = ImageAugmenter(random_seed=42)
    print("✅ ImageAugmenter initialisé avec succès")
    
    # 2. Créer une image de test géophysique
    print("\n🎨 Création d'une image de test géophysique...")
    test_image = create_geophysical_test_image(size=(256, 256))
    print(f"✅ Image de test créée: {test_image.size}")
    
    # Sauvegarder l'image originale
    test_image.save("geophysical_test_original.png")
    print("💾 Image originale sauvegardée: geophysical_test_original.png")
    
    # 3. Démontrer les techniques de base
    basic_results = demonstrate_basic_augmentations(augmenter, test_image)
    
    # 4. Démontrer les techniques avancées
    advanced_results = demonstrate_advanced_augmentations(augmenter, test_image)
    
    # 5. Démontrer les techniques géophysiques
    geophysical_results = demonstrate_geophysical_augmentations(augmenter, test_image)
    
    # 6. Démontrer les combinaisons
    combined_results = demonstrate_combined_augmentations(augmenter, test_image)
    
    # 7. Obtenir le résumé des techniques disponibles
    summary = augmenter.get_augmentation_summary()
    print(f"\n📊 Résumé des augmentations:")
    print(f"  - Total d'augmentations effectuées: {summary['total_augmentations']}")
    print(f"  - Types d'augmentation utilisés: {', '.join(summary['augmentation_types'])}")
    print(f"  - Techniques disponibles: {len(summary['available_techniques'])}")
    
    # 8. Sauvegarder les résultats
    all_results = {**basic_results, **advanced_results, **geophysical_results}
    saved_files = save_augmentation_results(all_results)
    
    # 9. Créer des visualisations
    print("\n📊 Création des visualisations...")
    
    # Visualisation des techniques de base
    if basic_results:
        fig1 = create_augmentation_visualization(test_image, basic_results, "Techniques de Base")
        fig1.savefig("augmented_images/basic_augmentations.png", dpi=300, bbox_inches='tight')
        print("  ✅ Visualisation des techniques de base sauvegardée")
    
    # Visualisation des techniques avancées
    if advanced_results:
        fig2 = create_augmentation_visualization(test_image, advanced_results, "Techniques Avancées")
        fig2.savefig("augmented_images/advanced_augmentations.png", dpi=300, bbox_inches='tight')
        print("  ✅ Visualisation des techniques avancées sauvegardée")
    
    # Visualisation des techniques géophysiques
    if geophysical_results:
        fig3 = create_augmentation_visualization(test_image, geophysical_results, "Techniques Géophysiques")
        fig3.savefig("augmented_images/geophysical_augmentations.png", dpi=300, bbox_inches='tight')
        print("  ✅ Visualisation des techniques géophysiques sauvegardée")
    
    # 10. Résumé final
    print("\n🎉 Démonstration terminée avec succès!")
    print(f"\n📁 Fichiers générés:")
    print(f"  - geophysical_test_original.png: Image de test originale")
    print(f"  - augmented_images/: Dossier contenant toutes les images augmentées")
    print(f"  - Visualisations PNG des différentes catégories d'augmentation")
    
    print(f"\n🔍 Techniques d'augmentation testées:")
    print(f"  - Techniques de base: {len([r for r in basic_results.values() if r is not None])}/5")
    print(f"  - Techniques avancées: {len([r for r in advanced_results.values() if r is not None])}/6")
    print(f"  - Techniques géophysiques: {len([r for r in geophysical_results.values() if r is not None])}/5")
    
    print(f"\n💡 Utilisez ces techniques dans votre pipeline d'entraînement!")
    print(f"   Exemple: augmenter.augment_image(image, ['elastic_deformation', 'fracture_patterns'])")
    
    # Afficher quelques visualisations
    if basic_results:
        plt.figure(figsize=(15, 10))
        plt.suptitle("Techniques d'Augmentation de Base", fontsize=16, fontweight='bold')
        create_augmentation_visualization(test_image, basic_results, "Base")
        plt.show()


if __name__ == "__main__":
    main()
