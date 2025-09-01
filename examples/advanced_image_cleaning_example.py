#!/usr/bin/env python3
"""
Exemple d'utilisation des méthodes de nettoyage avancées
======================================================

Ce script démontre comment utiliser les nouvelles fonctionnalités de nettoyage
d'images du GeophysicalImageProcessor pour traiter des images géophysiques.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Ajouter le chemin du projet
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from data.image_processor import GeophysicalImageProcessor


def create_geophysical_test_image(size=(512, 512)):
    """
    Créer une image de test réaliste avec des motifs géologiques.
    
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


def add_realistic_noise_and_artifacts(image, noise_level=0.3):
    """
    Ajouter du bruit et des artefacts réalistes à l'image.
    
    Args:
        image: Image PIL à dégrader
        noise_level: Niveau de bruit (0.0 à 1.0)
    
    Returns:
        Image dégradée
    """
    img_array = np.array(image)
    
    # 1. Bruit gaussien
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    img_array = img_array + noise
    
    # 2. Bruit impulsionnel (sel-et-poivre)
    salt_pepper_mask = np.random.random(img_array.shape[:2]) < 0.01
    img_array[salt_pepper_mask] = [255, 255, 255]  # Points blancs
    
    salt_pepper_mask = np.random.random(img_array.shape[:2]) < 0.01
    img_array[salt_pepper_mask] = [0, 0, 0]  # Points noirs
    
    # 3. Lignes de balayage (artefacts d'acquisition)
    for i in range(0, img_array.shape[1], 25):
        img_array[:, i:i+2] = img_array[:, i:i+2] * 0.8  # Lignes sombres
    
    # 4. Bandes de moiré
    for i in range(0, img_array.shape[0], 30):
        img_array[i:i+3, :] = img_array[i:i+3, :] * 1.1  # Bandes claires
    
    # 5. Vignettage (assombrissement des coins)
    h, w = img_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    
    for y in range(h):
        for x in range(w):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            vignette_factor = 1 - 0.3 * (distance / max_distance)**2
            img_array[y, x] = img_array[y, x] * vignette_factor
    
    # Normaliser et convertir en uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def demonstrate_cleaning_pipeline(processor, degraded_image, output_dir="cleaned_images"):
    """
    Démontrer le pipeline de nettoyage complet.
    
    Args:
        processor: Instance de GeophysicalImageProcessor
        degraded_image: Image dégradée à nettoyer
        output_dir: Dossier de sortie pour les images nettoyées
    """
    print("\n🚀 Démonstration du pipeline de nettoyage complet...")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Réduction de bruit avec différentes méthodes
    print("  🔍 Étape 1: Réduction de bruit...")
    noise_methods = ["gaussian", "median", "bilateral", "wiener"]
    noise_results = {}
    
    for method in noise_methods:
        try:
            cleaned = processor.apply_noise_reduction(degraded_image, method=method)
            noise_results[method] = cleaned
            
            # Sauvegarder
            filename = f"noise_reduction_{method}.png"
            filepath = os.path.join(output_dir, filename)
            cleaned.save(filepath)
            print(f"    ✅ {method}: {filename}")
            
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")
    
    # 2. Correction d'artefacts
    print("  🔧 Étape 2: Correction d'artefacts...")
    artifact_types = ["scan_lines", "salt_pepper", "streaking", "banding"]
    artifact_results = {}
    
    for artifact in artifact_types:
        try:
            corrected = processor.correct_artifacts(degraded_image, artifact)
            artifact_results[artifact] = corrected
            
            # Sauvegarder
            filename = f"artifact_correction_{artifact}.png"
            filepath = os.path.join(output_dir, filename)
            corrected.save(filepath)
            print(f"    ✅ {artifact}: {filename}")
            
        except Exception as e:
            print(f"    ❌ {artifact}: Erreur - {e}")
    
    # 3. Amélioration du contraste
    print("  ✨ Étape 3: Amélioration du contraste...")
    contrast_methods = ["histogram_equalization", "adaptive_histogram", "clahe", "gamma_correction"]
    contrast_results = {}
    
    for method in contrast_methods:
        try:
            enhanced = processor.enhance_contrast(degraded_image, method=method)
            contrast_results[method] = enhanced
            
            # Sauvegarder
            filename = f"contrast_enhancement_{method}.png"
            filepath = os.path.join(output_dir, filename)
            enhanced.save(filepath)
            print(f"    ✅ {method}: {filename}")
            
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")
    
    # 4. Pipeline de nettoyage géophysique complet
    print("  🎯 Étape 4: Pipeline de nettoyage géophysique complet...")
    cleaning_steps = [
        "noise_reduction",      # Réduction de bruit
        "scan_lines_removal",   # Suppression des lignes de balayage
        "contrast_enhancement", # Amélioration du contraste
        "salt_pepper_removal"   # Suppression du bruit sel-et-poivre
    ]
    
    try:
        # Appliquer le pipeline complet
        final_cleaned = processor.apply_geophysical_specific_cleaning(
            degraded_image, cleaning_steps
        )
        
        # Sauvegarder l'image finale
        final_filename = "final_cleaned_image.png"
        final_filepath = os.path.join(output_dir, final_filename)
        final_cleaned.save(final_filepath)
        print(f"    ✅ Pipeline complet: {final_filename}")
        
        # Obtenir le résumé des améliorations
        summary = processor.get_cleaning_summary(degraded_image, cleaning_steps)
        print(f"    📊 Résumé des améliorations:")
        print(f"      - Réduction de bruit: {summary['noise_reduction']:.2f}")
        print(f"      - Amélioration du contraste: {summary['contrast_improvement']:.2f}")
        print(f"      - Amélioration des gradients: {summary['gradient_enhancement']:.2f}")
        
        return final_cleaned, summary
        
    except Exception as e:
        print(f"    ❌ Pipeline complet: Erreur - {e}")
        return None, None


def create_comparison_visualization(original, degraded, cleaned, output_dir):
    """
    Créer une visualisation comparative des résultats.
    
    Args:
        original: Image originale
        degraded: Image dégradée
        cleaned: Image nettoyée
        output_dir: Dossier de sortie
    """
    print("\n📊 Création de la visualisation comparative...")
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image originale
    axes[0].imshow(original)
    axes[0].set_title("Image Originale\n(Motifs géologiques propres)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Image dégradée
    axes[1].imshow(degraded)
    axes[1].set_title("Image Dégradée\n(Bruit + Artefacts)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Image nettoyée
    axes[2].imshow(cleaned)
    axes[2].set_title("Image Nettoyée\n(Pipeline complet)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder la visualisation
    comparison_file = os.path.join(output_dir, "comparison_visualization.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ Visualisation comparative sauvegardée: {comparison_file}")
    
    plt.show()


def main():
    """Fonction principale de démonstration."""
    print("🧪 Démonstration des méthodes de nettoyage avancées pour images géophysiques")
    print("=" * 80)
    
    # 1. Créer le processeur
    print("\n🔧 Initialisation du processeur d'images...")
    processor = GeophysicalImageProcessor(target_size=(512, 512), channels=3)
    print(f"✅ Processeur initialisé: {processor.target_size}, {processor.channels} canaux")
    
    # 2. Créer une image de test géophysique
    print("\n🎨 Création d'une image de test géophysique...")
    original_image = create_geophysical_test_image(size=(512, 512))
    print(f"✅ Image géophysique créée: {original_image.size}")
    
    # Sauvegarder l'image originale
    original_image.save("geophysical_original.png")
    print("💾 Image originale sauvegardée: geophysical_original.png")
    
    # 3. Ajouter du bruit et des artefacts
    print("\n🌫️ Ajout de bruit et d'artefacts réalistes...")
    degraded_image = add_realistic_noise_and_artifacts(original_image, noise_level=0.4)
    print("✅ Image dégradée créée avec bruit et artefacts")
    
    # Sauvegarder l'image dégradée
    degraded_image.save("geophysical_degraded.png")
    print("💾 Image dégradée sauvegardée: geophysical_degraded.png")
    
    # 4. Appliquer le pipeline de nettoyage
    output_dir = "cleaned_images"
    final_cleaned, cleaning_summary = demonstrate_cleaning_pipeline(
        processor, degraded_image, output_dir
    )
    
    # 5. Créer la visualisation comparative
    if final_cleaned is not None:
        create_comparison_visualization(
            original_image, degraded_image, final_cleaned, output_dir
        )
    
    # 6. Résumé final
    print("\n🎉 Démonstration terminée avec succès!")
    print(f"\n📁 Fichiers générés:")
    print(f"  - geophysical_original.png: Image géophysique originale")
    print(f"  - geophysical_degraded.png: Image avec bruit et artefacts")
    print(f"  - {output_dir}/: Dossier contenant toutes les images nettoyées")
    
    if cleaning_summary:
        print(f"\n📈 Résumé des améliorations finales:")
        print(f"  - Réduction de bruit: {cleaning_summary['noise_reduction']:.2f}")
        print(f"  - Amélioration du contraste: {cleaning_summary['contrast_improvement']:.2f}")
        print(f"  - Amélioration des gradients: {cleaning_summary['gradient_enhancement']:.2f}")
        print(f"  - Méthodes appliquées: {', '.join(cleaning_summary['cleaning_methods_applied'])}")
    
    print(f"\n💡 Utilisez ces méthodes dans votre pipeline de traitement d'images géophysiques!")
    print(f"   Exemple: processor.apply_noise_reduction(image, method='bilateral')")


if __name__ == "__main__":
    main()
