#!/usr/bin/env python3
"""
Exemple d'utilisation du module d'entraînement géophysique avec l'augmenteur de données.

Ce script démontre comment :
1. Charger et nettoyer les données géophysiques
2. Créer un modèle CNN
3. Préparer les données avec augmentation
4. Entraîner le modèle
5. Évaluer les performances
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import os

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.model.geophysical_trainer import GeophysicalTrainer, GeophysicalCNN2D
from src.utils.logger import logger


def create_sample_2d_data(num_samples=10, grid_size=32, num_channels=4):
    """Créer des données 2D d'exemple pour la démonstration."""
    grids = []
    labels = []
    
    for i in range(num_samples):
        # Créer une grille avec des motifs géophysiques simulés
        grid = np.random.randn(num_channels, grid_size, grid_size)
        
        # Ajouter des motifs structurés
        if i % 2 == 0:
            # Anomalie circulaire (résistivité élevée)
            center_x, center_y = grid_size // 2, grid_size // 2
            for x in range(grid_size):
                for y in range(grid_size):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < grid_size // 4:
                        grid[0, x, y] += 2.0  # Résistivité élevée
        
        grids.append(grid)
        labels.append(i % 2)  # Classes binaires
    
    return grids, labels


def main():
    """Fonction principale de démonstration."""
    logger.info("=== Démonstration du module d'entraînement géophysique ===")
    
    # 1. Initialiser l'augmenteur et l'entraîneur
    logger.info("1. Initialisation des composants...")
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    trainer = GeophysicalTrainer(augmenter, device="auto")
    
    # 2. Créer des données d'exemple
    logger.info("2. Création de données d'exemple...")
    grids, labels = create_sample_2d_data(num_samples=20, grid_size=32, num_channels=4)
    logger.info(f"   Données créées: {len(grids)} grilles, {len(set(labels))} classes")
    
    # 3. Préparer les données avec augmentation
    logger.info("3. Préparation des données avec augmentation...")
    try:
        train_loader, val_loader = trainer.prepare_data_2d(
            grids=grids,
            labels=labels,
            augmentations=["rotation", "flip_horizontal", "gaussian_noise"],
            num_augmentations=3,
            test_size=0.2,
            random_state=42
        )
        logger.info("   Données préparées avec succès!")
    except Exception as e:
        logger.error(f"   Erreur lors de la préparation des données: {e}")
        return
    
    # 4. Créer le modèle
    logger.info("4. Création du modèle CNN 2D...")
    model = GeophysicalCNN2D(
        input_channels=4,
        num_classes=2,
        grid_size=32,
        dropout_rate=0.3
    )
    
    # Afficher le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Modèle créé: {total_params:,} paramètres totaux, {trainable_params:,} entraînables")
    
    # 5. Entraîner le modèle (version courte pour la démonstration)
    logger.info("5. Début de l'entraînement...")
    try:
        history = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,  # Peu d'époques pour la démonstration
            learning_rate=0.001,
            weight_decay=1e-5,
            patience=3
        )
        logger.info("   Entraînement terminé avec succès!")
        
        # 6. Afficher un résumé des augmentations
        logger.info("6. Résumé des augmentations utilisées...")
        aug_summary = trainer.get_augmentation_summary()
        logger.info(f"   Nombre total d'augmentations: {len(aug_summary)}")
        
        # 7. Sauvegarder le modèle
        logger.info("7. Sauvegarde du modèle...")
        trainer.save_model(model, "example_model.pth")
        
        # 8. Tracer l'historique d'entraînement
        logger.info("8. Tracé de l'historique d'entraînement...")
        trainer.plot_training_history(save_path="training_history.png")
        
    except Exception as e:
        logger.error(f"   Erreur lors de l'entraînement: {e}")
        return
    
    logger.info("=== Démonstration terminée avec succès! ===")
    logger.info("Fichiers générés:")
    logger.info("  - example_model.pth: Modèle entraîné")
    logger.info("  - training_history.png: Graphique d'entraînement")


if __name__ == "__main__":
    main()
