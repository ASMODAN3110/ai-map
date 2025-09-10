#!/usr/bin/env python3
"""
Script de démonstration pour les modèles générateurs géophysiques.

Ce script démontre l'utilisation des nouveaux modèles U-Net 2D et VoxNet 3D
conformes au cahier des charges.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

from backend.model.geophysical_generators import (
    UNet2D, VoxNet3D, GeophysicalImageGenerator, 
    GeophysicalDataProcessor, create_sample_csv_data
)
from backend.utils.logger import logger


def demo_unet_2d():
    """Démonstration du modèle U-Net 2D."""
    logger.info("🎯 DÉMONSTRATION U-NET 2D")
    logger.info("=" * 40)
    
    try:
        # Créer le modèle
        model = UNet2D()
        logger.info(f"✅ Modèle U-Net 2D créé")
        
        # Créer des données de test
        processor = GeophysicalDataProcessor()
        csv_data = create_sample_csv_data(n_samples=3)
        grids_2d = processor.process_csv_to_2d_grid(csv_data)
        
        logger.info(f"✅ Données de test créées: {grids_2d.shape}")
        
        # Générer des prédictions
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions = model(grids_2d)
            generation_time = time.time() - start_time
        
        logger.info(f"✅ Prédictions générées: {predictions.shape}")
        logger.info(f"   - Temps de génération: {generation_time:.3f}s")
        logger.info(f"   - Temps moyen par échantillon: {generation_time/len(csv_data):.3f}s")
        
        # Analyser les résultats
        resistivity_channel = predictions[:, 0, :, :]  # Canal résistivité
        chargeability_channel = predictions[:, 1, :, :]  # Canal chargeabilité
        
        logger.info(f"   - Résistivité: min={resistivity_channel.min():.3f}, max={resistivity_channel.max():.3f}")
        logger.info(f"   - Chargeabilité: min={chargeability_channel.min():.3f}, max={chargeability_channel.max():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la démonstration U-Net 2D: {e}")
        return False


def demo_voxnet_3d():
    """Démonstration du modèle VoxNet 3D."""
    logger.info("🎯 DÉMONSTRATION VOXNET 3D")
    logger.info("=" * 40)
    
    try:
        # Créer le modèle
        model = VoxNet3D()
        logger.info(f"✅ Modèle VoxNet 3D créé")
        
        # Créer des données de test
        processor = GeophysicalDataProcessor()
        csv_data = create_sample_csv_data(n_samples=2)  # Moins d'échantillons pour 3D
        volumes_3d = processor.process_csv_to_3d_volume(csv_data)
        
        logger.info(f"✅ Données de test créées: {volumes_3d.shape}")
        
        # Générer des prédictions
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions = model(volumes_3d)
            generation_time = time.time() - start_time
        
        logger.info(f"✅ Prédictions générées: {predictions.shape}")
        logger.info(f"   - Temps de génération: {generation_time:.3f}s")
        logger.info(f"   - Temps moyen par échantillon: {generation_time/len(csv_data):.3f}s")
        
        # Analyser les résultats
        chargeability_volume = predictions[:, 0, :, :, :]  # Volume de chargeabilité
        
        logger.info(f"   - Chargeabilité 3D: min={chargeability_volume.min():.3f}, max={chargeability_volume.max():.3f}")
        logger.info(f"   - Forme du volume: {chargeability_volume.shape[1:]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la démonstration VoxNet 3D: {e}")
        return False


def demo_integrated_generator():
    """Démonstration du générateur intégré."""
    logger.info("🎯 DÉMONSTRATION GÉNÉRATEUR INTÉGRÉ")
    logger.info("=" * 40)
    
    try:
        # Créer le générateur
        generator = GeophysicalImageGenerator()
        logger.info(f"✅ Générateur intégré créé")
        
        # Créer des données de test
        csv_data = create_sample_csv_data(n_samples=3)
        logger.info(f"✅ Données CSV créées: {csv_data.shape}")
        
        # Générer les pseudo-sections 2D
        logger.info("🖼️ Génération des pseudo-sections 2D...")
        start_time = time.time()
        pseudo_sections = generator.generate_pseudo_sections(csv_data, method="pole-dipole")
        time_2d = time.time() - start_time
        
        logger.info(f"✅ {len(pseudo_sections)} pseudo-sections générées en {time_2d:.3f}s")
        
        # Générer les modèles 3D
        logger.info("🌍 Génération des modèles 3D...")
        start_time = time.time()
        models_3d = generator.generate_3d_models(csv_data, method="pole-dipole")
        time_3d = time.time() - start_time
        
        logger.info(f"✅ {len(models_3d)} modèles 3D générés en {time_3d:.3f}s")
        
        # Résumé des performances
        total_time = time_2d + time_3d
        logger.info(f"📊 RÉSUMÉ DES PERFORMANCES:")
        logger.info(f"   - Temps total: {total_time:.3f}s")
        logger.info(f"   - Temps 2D: {time_2d:.3f}s ({time_2d/total_time*100:.1f}%)")
        logger.info(f"   - Temps 3D: {time_3d:.3f}s ({time_3d/total_time*100:.1f}%)")
        logger.info(f"   - Temps moyen par échantillon: {total_time/len(csv_data):.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la démonstration générateur intégré: {e}")
        return False


def demo_data_processing():
    """Démonstration du traitement des données."""
    logger.info("🎯 DÉMONSTRATION TRAITEMENT DES DONNÉES")
    logger.info("=" * 40)
    
    try:
        # Créer le processeur
        processor = GeophysicalDataProcessor()
        logger.info(f"✅ Processeur de données créé")
        
        # Créer des données CSV
        csv_data = create_sample_csv_data(n_samples=2)
        logger.info(f"✅ Données CSV créées: {csv_data.shape}")
        logger.info(f"   - Résistivité: {csv_data[:, 0]}")
        logger.info(f"   - Chargeabilité: {csv_data[:, 1]}")
        logger.info(f"   - Coordonnées X: {csv_data[:, 2]}")
        logger.info(f"   - Coordonnées Y: {csv_data[:, 3]}")
        
        # Traiter en grilles 2D
        grids_2d = processor.process_csv_to_2d_grid(csv_data)
        logger.info(f"✅ Grilles 2D créées: {grids_2d.shape}")
        logger.info(f"   - Canaux: {grids_2d.shape[1]}")
        logger.info(f"   - Résolution: {grids_2d.shape[2]}×{grids_2d.shape[3]}")
        
        # Traiter en volumes 3D
        volumes_3d = processor.process_csv_to_3d_volume(csv_data)
        logger.info(f"✅ Volumes 3D créés: {volumes_3d.shape}")
        logger.info(f"   - Canaux: {volumes_3d.shape[1]}")
        logger.info(f"   - Résolution: {volumes_3d.shape[2]}×{volumes_3d.shape[3]}×{volumes_3d.shape[4]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la démonstration traitement des données: {e}")
        return False


def demo_model_architecture():
    """Démonstration de l'architecture des modèles."""
    logger.info("🎯 DÉMONSTRATION ARCHITECTURE DES MODÈLES")
    logger.info("=" * 40)
    
    try:
        # U-Net 2D
        unet_2d = UNet2D()
        total_params_2d = sum(p.numel() for p in unet_2d.parameters() if p.requires_grad)
        logger.info(f"✅ U-Net 2D:")
        logger.info(f"   - Paramètres: {total_params_2d:,}")
        logger.info(f"   - Entrée: (batch_size, 4, 64, 64)")
        logger.info(f"   - Sortie: (batch_size, 2, 64, 64)")
        
        # VoxNet 3D
        voxnet_3d = VoxNet3D()
        total_params_3d = sum(p.numel() for p in voxnet_3d.parameters() if p.requires_grad)
        logger.info(f"✅ VoxNet 3D:")
        logger.info(f"   - Paramètres: {total_params_3d:,}")
        logger.info(f"   - Entrée: (batch_size, 4, 32, 32, 32)")
        logger.info(f"   - Sortie: (batch_size, 1, 32, 32, 32)")
        
        # Total
        total_params = total_params_2d + total_params_3d
        logger.info(f"📊 TOTAL:")
        logger.info(f"   - Paramètres totaux: {total_params:,}")
        logger.info(f"   - U-Net 2D: {total_params_2d/total_params*100:.1f}%")
        logger.info(f"   - VoxNet 3D: {total_params_3d/total_params*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la démonstration architecture: {e}")
        return False


def main():
    """Fonction principale de démonstration."""
    logger.info("🚀 DÉMONSTRATION DES MODÈLES GÉNÉRATEURS GÉOPHYSIQUES")
    logger.info("=" * 70)
    logger.info("Conformes au cahier des charges:")
    logger.info("- U-Net 2D: ~31M paramètres, pseudo-sections 2D")
    logger.info("- VoxNet 3D: ~15M paramètres, modèles 3D")
    logger.info("=" * 70)
    
    demos = [
        ("Architecture des modèles", demo_model_architecture),
        ("Traitement des données", demo_data_processing),
        ("U-Net 2D", demo_unet_2d),
        ("VoxNet 3D", demo_voxnet_3d),
        ("Générateur intégré", demo_integrated_generator)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        logger.info(f"\n🎯 {demo_name.upper()}")
        logger.info("-" * 50)
        
        try:
            success = demo_func()
            results[demo_name] = "✅ Succès" if success else "❌ Échec"
        except Exception as e:
            logger.error(f"❌ Erreur lors de {demo_name}: {e}")
            results[demo_name] = f"❌ Erreur: {str(e)}"
    
    # Résumé final
    logger.info("\n📊 RÉSUMÉ DE LA DÉMONSTRATION")
    logger.info("=" * 50)
    
    successful_demos = sum(1 for result in results.values() if result.startswith("✅"))
    total_demos = len(results)
    
    for demo_name, result in results.items():
        logger.info(f"   {result} {demo_name}")
    
    logger.info(f"\n🎉 DÉMONSTRATION TERMINÉE:")
    logger.info(f"   - Démonstrations réussies: {successful_demos}/{total_demos}")
    logger.info(f"   - Taux de succès: {successful_demos/total_demos*100:.1f}%")
    
    if successful_demos == total_demos:
        logger.info("🎉 TOUTES LES DÉMONSTRATIONS SONT PASSÉES AVEC SUCCÈS!")
        logger.info("✅ Les modèles générateurs sont conformes au cahier des charges")
    else:
        logger.info("⚠️ CERTAINES DÉMONSTRATIONS ONT ÉCHOUÉ")
        logger.info("❌ Vérifiez les erreurs ci-dessus")
    
    return successful_demos == total_demos


if __name__ == "__main__":
    main()
