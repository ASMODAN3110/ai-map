#!/usr/bin/env python3
"""
Exemple d'utilisation du module GeophysicalDataAugmenter.
Ce script démontre comment utiliser les différentes techniques d'augmentation
pour améliorer l'entraînement des modèles CNN.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.data.data_processor import GeophysicalDataProcessor
from config import CONFIG


def create_sample_data():
    """Créer des données d'exemple pour la démonstration."""
    print("🔧 Création des données d'exemple...")
    
    # Créer une grille 2D d'exemple (64x64x4)
    height, width = CONFIG.processing.grid_2d
    channels = 4
    
    # Simuler des données géophysiques réalistes
    grid_2d = np.zeros((height, width, channels))
    
    # Canal 0: Résistivité (simuler un corps conducteur)
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    center_x, center_y = 0.5, 0.5
    radius = 0.2
    
    # Créer un corps conducteur circulaire
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    conductivity = np.exp(-distance / radius)
    grid_2d[:, :, 0] = conductivity * 1000  # Résistivité en Ohm.m
    
    # Canal 1: Chargeabilité (simuler un corps polarisable)
    chargeability = np.exp(-distance / (radius * 0.8)) * 150
    grid_2d[:, :, 1] = chargeability  # Chargeabilité en mV/V
    
    # Canaux 2-3: Coordonnées spatiales
    grid_2d[:, :, 2] = x * 1000  # Coordonnée X en mètres
    grid_2d[:, :, 3] = y * 1000  # Coordonnée Y en mètres
    
    # Créer un volume 3D d'exemple (32x32x32x4)
    depth, height, width = CONFIG.processing.grid_3d
    volume_3d = np.zeros((depth, height, width, channels))
    
    # Étendre la grille 2D en 3D
    for d in range(depth):
        volume_3d[d] = grid_2d
    
    # Créer un DataFrame d'exemple
    n_points = 200
    df = pd.DataFrame({
        'x': np.random.uniform(0, 1000, n_points),
        'y': np.random.uniform(0, 1000, n_points),
        'resistivity': np.random.lognormal(6, 1, n_points),  # Résistivité log-normale
        'chargeability': np.random.uniform(0, 200, n_points)
    })
    
    print(f"✅ Données d'exemple créées:")
    print(f"   - Grille 2D: {grid_2d.shape}")
    print(f"   - Volume 3D: {volume_3d.shape}")
    print(f"   - DataFrame: {df.shape}")
    
    return grid_2d, volume_3d, df


def demonstrate_2d_augmentation(augmenter, grid_2d):
    """Démontrer l'augmentation 2D."""
    print("\n🔄 Démonstration de l'augmentation 2D...")
    
    # Techniques d'augmentation recommandées pour les grilles 2D
    augmentations_2d = augmenter.get_recommended_augmentations("2d_grid")
    print(f"   Techniques recommandées: {augmentations_2d}")
    
    # Effectuer plusieurs augmentations
    augmented_grids = augmenter.augment_2d_grid(
        grid_2d, 
        augmentations_2d[:4],  # Utiliser les 4 premières techniques
        num_augmentations=3
    )
    
    print(f"   ✅ {len(augmented_grids)} grilles augmentées générées")
    
    # Afficher un résumé
    summary = augmenter.get_augmentation_summary()
    print(f"   Résumé: {summary['total_augmentations']} augmentations effectuées")
    
    return augmented_grids


def demonstrate_3d_augmentation(augmenter, volume_3d):
    """Démontrer l'augmentation 3D."""
    print("\n🔄 Démonstration de l'augmentation 3D...")
    
    # Techniques d'augmentation recommandées pour les volumes 3D
    augmentations_3d = augmenter.get_recommended_augmentations("3d_volume")
    print(f"   Techniques recommandées: {augmentations_3d}")
    
    # Effectuer plusieurs augmentations
    augmented_volumes = augmenter.augment_3d_volume(
        volume_3d, 
        augmentations_3d,
        num_augmentations=2
    )
    
    print(f"   ✅ {len(augmented_volumes)} volumes augmentés générés")
    
    return augmented_volumes


def demonstrate_dataframe_augmentation(augmenter, df):
    """Démontrer l'augmentation de DataFrame."""
    print("\n🔄 Démonstration de l'augmentation de DataFrame...")
    
    # Techniques d'augmentation recommandées pour les DataFrames
    augmentations_df = augmenter.get_recommended_augmentations("dataframe")
    print(f"   Techniques recommandées: {augmentations_df}")
    
    # Effectuer plusieurs augmentations
    augmented_dfs = augmenter.augment_dataframe(
        df, 
        augmentations_df,
        num_augmentations=2
    )
    
    print(f"   ✅ {len(augmented_dfs)} DataFrames augmentés générés")
    
    return augmented_dfs


def visualize_augmentations(original_grid, augmented_grids):
    """Visualiser les augmentations 2D."""
    print("\n📊 Visualisation des augmentations...")
    
    # Créer une figure avec les grilles originales et augmentées
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparaison des données originales et augmentées', fontsize=16)
    
    # Afficher la grille originale (canal de résistivité)
    im0 = axes[0, 0].imshow(original_grid[:, :, 0], cmap='viridis')
    axes[0, 0].set_title('Original - Résistivité')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Afficher la première augmentation
    im1 = axes[0, 1].imshow(augmented_grids[0][:, :, 0], cmap='viridis')
    axes[0, 1].set_title('Augmentation 1 - Résistivité')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Afficher la deuxième augmentation
    im2 = axes[1, 0].imshow(augmented_grids[1][:, :, 0], cmap='viridis')
    axes[1, 0].set_title('Augmentation 2 - Résistivité')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Afficher la troisième augmentation
    im3 = axes[1, 1].imshow(augmented_grids[2][:, :, 0], cmap='viridis')
    axes[1, 1].set_title('Augmentation 3 - Résistivité')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "data_augmentation_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   📁 Figure sauvegardée: {output_dir / 'data_augmentation_comparison.png'}")
    
    plt.show()


def demonstrate_advanced_features(augmenter):
    """Démontrer les fonctionnalités avancées."""
    print("\n🚀 Démonstration des fonctionnalités avancées...")
    
    # Validation des paramètres
    print("   🔍 Validation des paramètres d'augmentation:")
    
    valid_augmentations = ["rotation", "flip_horizontal"]
    is_valid = augmenter.validate_augmentation_parameters(valid_augmentations, "2d_grid")
    print(f"     - Augmentations valides: {is_valid}")
    
    invalid_augmentations = ["invalid_technique"]
    is_valid = augmenter.validate_augmentation_parameters(invalid_augmentations, "2d_grid")
    print(f"     - Augmentations invalides: {is_valid}")
    
    # Recommandations
    print("   💡 Recommandations d'augmentation:")
    for data_type in ["2d_grid", "3d_volume", "dataframe"]:
        recommendations = augmenter.get_recommended_augmentations(data_type)
        print(f"     - {data_type}: {recommendations}")
    
    # Historique et résumé
    print("   📋 Résumé des augmentations:")
    summary = augmenter.get_augmentation_summary()
    if "message" not in summary:
        print(f"     - Total: {summary['total_augmentations']}")
        print(f"     - Types: {summary['augmentation_types']}")
    else:
        print(f"     - {summary['message']}")


def main():
    """Fonction principale."""
    print("🚀 Démonstration du module GeophysicalDataAugmenter")
    print("=" * 60)
    
    try:
        # Créer l'augmenteur
        print("\n🔧 Initialisation de l'augmenteur...")
        augmenter = GeophysicalDataAugmenter(random_seed=42)
        print("   ✅ Augmenteur initialisé avec succès")
        
        # Créer des données d'exemple
        grid_2d, volume_3d, df = create_sample_data()
        
        # Démonstrations
        augmented_grids = demonstrate_2d_augmentation(augmenter, grid_2d)
        augmented_volumes = demonstrate_3d_augmentation(augmenter, volume_3d)
        augmented_dfs = demonstrate_dataframe_augmentation(augmenter, df)
        
        # Fonctionnalités avancées
        demonstrate_advanced_features(augmenter)
        
        # Visualisation
        visualize_augmentations(grid_2d, augmented_grids)
        
        # Résumé final
        print("\n🎉 Démonstration terminée avec succès!")
        print("\n📊 Statistiques finales:")
        final_summary = augmenter.get_augmentation_summary()
        if "message" not in final_summary:
            print(f"   - Total des augmentations: {final_summary['total_augmentations']}")
            print(f"   - Types d'augmentation utilisés: {list(final_summary['augmentation_types'].keys())}")
        
        print("\n💡 Utilisation dans votre pipeline:")
        print("   1. Intégrez l'augmenteur dans votre DataProcessor")
        print("   2. Appliquez l'augmentation avant l'entraînement")
        print("   3. Utilisez les techniques recommandées selon vos données")
        print("   4. Surveillez l'historique des augmentations")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
