#!/usr/bin/env python3
"""
Script pour corriger le pipeline et permettre l'exécution complète.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def create_dummy_training_data():
    """Créer des données d'entraînement factices pour tester le pipeline."""
    
    print("📊 Création de données d'entraînement factices...")
    
    # Créer des données d'entraînement 2D factices
    n_samples = 50
    n_channels = 4
    grid_size = 64
    
    # Données d'entraînement
    x_train = np.random.rand(n_samples, n_channels, grid_size, grid_size).astype(np.float32)
    x_test = np.random.rand(n_samples // 4, n_channels, grid_size, grid_size).astype(np.float32)
    
    # Labels factices
    y_train = np.random.randint(0, 2, n_samples)
    y_test = np.random.randint(0, 2, n_samples // 4)
    
    # Volume 3D factice
    volume_3d = np.random.rand(20, n_channels, 32, 32, 32).astype(np.float32)
    
    print(f"✅ Données créées:")
    print(f"  - x_train: {x_train.shape}")
    print(f"  - x_test: {x_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    print(f"  - volume_3d: {volume_3d.shape}")
    
    return x_train, x_test, y_train, y_test, volume_3d

def patch_main_for_testing():
    """Modifier temporairement main.py pour utiliser des données factices."""
    
    print("🔧 Modification temporaire de main.py...")
    
    # Lire le fichier main.py
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Sauvegarder l'original
    with open("main.py.backup", "w", encoding="utf-8") as f:
        f.write(content)
    
    # Modifier la fonction phase2_data_processing pour retourner des données factices
    old_phase2 = '''    def phase2_data_processing() -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Phase 2: Traitement des données et création des grilles spatiales.
        
        Returns:
            Tuple contenant (processor, multi_device_tensor, volume_3d)
        """
        logger.info("\\n📊 Phase 2: Traitement des données et création des grilles")
        logger.info("-" * 40)
        
        # Initialiser le processeur de données
        from src.data.data_processor import GeophysicalDataProcessor
        processor = GeophysicalDataProcessor()
        
        # Charger et valider les données nettoyées
        device_data = processor.load_and_validate()
        
        if not device_data:
            logger.warning("Aucune donnée de dispositif valide trouvée après le nettoyage")
            logger.info("Le pipeline continuera avec des données vides pour la démonstration")
        return processor, None, None
        
        # Créer les grilles spatiales
        spatial_grids = processor.create_spatial_grids()
        
        # Créer le tenseur multi-dispositifs pour l'entrée CNN
        multi_device_tensor = processor.create_multi_device_tensor()
        
        # Créer le volume 3D pour VoxNet
        volume_3d = processor.create_3d_volume()
        
        logger.info("✅ Traitement des données terminé avec succès")
        logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
        logger.info(f"Forme du volume 3D: {volume_3d.shape}")
                
        return processor, multi_device_tensor, volume_3d'''
    
    new_phase2 = '''    def phase2_data_processing() -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Phase 2: Traitement des données et création des grilles spatiales.
        
        Returns:
            Tuple contenant (processor, multi_device_tensor, volume_3d)
        """
        logger.info("\\n📊 Phase 2: Traitement des données et création des grilles")
        logger.info("-" * 40)
        
        # Initialiser le processeur de données
        from src.data.data_processor import GeophysicalDataProcessor
        processor = GeophysicalDataProcessor()
        
        # Charger et valider les données nettoyées
        device_data = processor.load_and_validate()
        
        if not device_data:
            logger.warning("Aucune donnée de dispositif valide trouvée après le nettoyage")
            logger.info("Le pipeline continuera avec des données factices pour la démonstration")
            
            # Créer des données factices pour la démonstration
            n_samples = 50
            n_channels = 4
            grid_size = 64
            
            # Tenseur multi-dispositifs factice
            multi_device_tensor = np.random.rand(n_samples, n_channels, grid_size, grid_size).astype(np.float32)
            
            # Volume 3D factice
            volume_3d = np.random.rand(20, n_channels, 32, 32, 32).astype(np.float32)
            
            logger.info("✅ Données factices créées pour la démonstration")
            logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
            logger.info(f"Forme du volume 3D: {volume_3d.shape}")
            
            return processor, multi_device_tensor, volume_3d
        
        # Créer les grilles spatiales
        spatial_grids = processor.create_spatial_grids()
        
        # Créer le tenseur multi-dispositifs pour l'entrée CNN
        multi_device_tensor = processor.create_multi_device_tensor()
        
        # Créer le volume 3D pour VoxNet
        volume_3d = processor.create_3d_volume()
        
        logger.info("✅ Traitement des données terminé avec succès")
        logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
        logger.info(f"Forme du volume 3D: {volume_3d.shape}")
                
        return processor, multi_device_tensor, volume_3d'''
    
    # Remplacer la fonction
    new_content = content.replace(old_phase2, new_phase2)
    
    # Sauvegarder la version modifiée
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ main.py modifié temporairement")

def restore_main():
    """Restaurer le fichier main.py original."""
    
    print("🔄 Restauration de main.py...")
    
    if Path("main.py.backup").exists():
        with open("main.py.backup", "r", encoding="utf-8") as f:
            content = f.read()
        
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        Path("main.py.backup").unlink()
        print("✅ main.py restauré")
    else:
        print("❌ Fichier de sauvegarde non trouvé")

def main():
    """Fonction principale."""
    print("🚀 Correction du pipeline AI-MAP pour l'exécution complète")
    print("=" * 60)
    
    try:
        # Modifier main.py temporairement
        patch_main_for_testing()
        
        print("\n✅ MODIFICATION TERMINÉE!")
        print("=" * 60)
        print("\n📋 Pour exécuter le pipeline complet:")
        print("1. python main.py --epochs 1 --verbose")
        print("2. python main.py --model hybrid --epochs 1")
        print("3. python main.py --model cnn_3d --epochs 1")
        print("\n🔄 Pour restaurer l'original:")
        print("   python fix_pipeline.py --restore")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        restore_main()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_main()
    else:
        main()
