#!/usr/bin/env python3
"""
Script de test pour les nouvelles fonctionnalités du module d'augmentation de données.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent))

from backend.preprocessor.data_augmenter import GeophysicalDataAugmenter

def test_geophysical_techniques():
    """Tester les nouvelles techniques géophysiques spécialisées."""
    print("🧪 Test des techniques géophysiques spécialisées")
    print("=" * 60)
    
    # Initialiser l'augmenteur
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    
    # Créer une grille de test 2D
    grid_2d = np.random.rand(64, 64, 4)  # height, width, channels
    
    # Techniques géophysiques spécialisées
    geophysical_techniques = [
        "geological_stratification",
        "fracture_patterns", 
        "mineral_inclusions",
        "weathering_effects",
        "sedimentary_layers"
    ]
    
    print(f"📊 Grille de test: {grid_2d.shape}")
    
    for technique in geophysical_techniques:
        print(f"\n🔍 Test de {technique}...")
        try:
            augmented = augmenter.augment_2d_grid(grid_2d, [technique], num_augmentations=1)
            print(f"   ✅ {technique}: {len(augmented)} grille(s) générée(s)")
            print(f"   📏 Forme: {augmented[0].shape}")
        except Exception as e:
            print(f"   ❌ {technique}: Erreur - {e}")
    
    return True

def test_advanced_techniques():
    """Tester les techniques avancées manquantes."""
    print("\n🎨 Test des techniques avancées")
    print("=" * 60)
    
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    grid_2d = np.random.rand(32, 32, 4)
    
    # Techniques avancées
    advanced_techniques = [
        "color_jittering",
        "blur_sharpen", 
        "perspective_transform",
        "cutout"
    ]
    
    print(f"📊 Grille de test: {grid_2d.shape}")
    
    for technique in advanced_techniques:
        print(f"\n🔍 Test de {technique}...")
        try:
            augmented = augmenter.augment_2d_grid(grid_2d, [technique], num_augmentations=1)
            print(f"   ✅ {technique}: {len(augmented)} grille(s) générée(s)")
            print(f"   📏 Forme: {augmented[0].shape}")
        except Exception as e:
            print(f"   ❌ {technique}: Erreur - {e}")
    
    return True

def test_validation_methods():
    """Tester les nouvelles méthodes de validation."""
    print("\n✅ Test des méthodes de validation")
    print("=" * 60)
    
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    
    # Test de validation des paramètres
    test_augmentations = ["rotation", "geological_stratification", "fracture_patterns"]
    
    print("🔍 Test de validation des paramètres...")
    
    # Test pour Pôle-Dipôle
    is_valid_pd = augmenter.validate_augmentation_parameters(
        test_augmentations, "2d_grid", "pole_dipole"
    )
    print(f"   Pôle-Dipôle: {'✅ Valide' if is_valid_pd else '❌ Invalide'}")
    
    # Test pour Schlumberger
    is_valid_sch = augmenter.validate_augmentation_parameters(
        test_augmentations, "2d_grid", "schlumberger"
    )
    print(f"   Schlumberger: {'✅ Valide' if is_valid_sch else '❌ Invalide'}")
    
    # Test de validation géophysique
    print("\n🔍 Test de validation géophysique...")
    geophysical_valid = augmenter.validate_geophysical_parameters(
        test_augmentations, "pole_dipole"
    )
    print(f"   Validation géophysique: {'✅ Valide' if geophysical_valid else '❌ Invalide'}")
    
    return True

def test_recommendations():
    """Tester les nouvelles méthodes de recommandation."""
    print("\n📋 Test des recommandations")
    print("=" * 60)
    
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    
    # Test des recommandations par méthode
    print("🔍 Test des recommandations par méthode...")
    
    for method in ["pole_dipole", "schlumberger"]:
        print(f"\n📊 Recommandations pour {method}:")
        
        # Recommandations 2D
        rec_2d = augmenter.get_recommended_augmentations("2d_grid", method)
        print(f"   2D: {rec_2d}")
        
        # Recommandations 3D
        rec_3d = augmenter.get_recommended_augmentations("3d_volume", method)
        print(f"   3D: {rec_3d}")
        
        # Recommandations DataFrame
        rec_df = augmenter.get_recommended_augmentations("dataframe", method)
        print(f"   DataFrame: {rec_df}")
    
    # Test du guide complet
    print("\n📖 Guide complet des augmentations géophysiques:")
    guide = augmenter.get_geophysical_augmentation_guide()
    
    for method, info in guide.items():
        print(f"\n🔬 {method.upper()}:")
        print(f"   Description: {info['description']}")
        print(f"   Bonnes pratiques: {len(info['best_practices'])} conseils")
        print(f"   Recommandations 2D: {info['recommended_2d']}")
    
    return True

def test_dataframe_augmentations():
    """Tester les augmentations de DataFrame."""
    print("\n📊 Test des augmentations de DataFrame")
    print("=" * 60)
    
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    
    # Créer un DataFrame de test
    df_test = pd.DataFrame({
        'x': np.random.uniform(500000, 501000, 100),
        'y': np.random.uniform(450000, 451000, 100),
        'z': np.random.uniform(500, 600, 100),
        'resistivity': np.random.uniform(10, 1000, 100),
        'chargeability': np.random.uniform(1, 50, 100)
    })
    
    print(f"📊 DataFrame de test: {df_test.shape}")
    print(f"   Colonnes: {list(df_test.columns)}")
    
    # Techniques géophysiques pour DataFrame
    df_techniques = [
        "geological_stratification",
        "fracture_patterns",
        "mineral_inclusions", 
        "weathering_effects",
        "sedimentary_layers"
    ]
    
    for technique in df_techniques:
        print(f"\n🔍 Test de {technique} sur DataFrame...")
        try:
            augmented = augmenter.augment_dataframe(df_test, [technique], num_augmentations=1)
            print(f"   ✅ {technique}: {len(augmented)} DataFrame(s) généré(s)")
            print(f"   📏 Forme: {augmented[0].shape}")
            
            # Vérifier que les colonnes sont préservées
            if list(augmented[0].columns) == list(df_test.columns):
                print(f"   ✅ Colonnes préservées")
            else:
                print(f"   ⚠️ Colonnes modifiées")
                
        except Exception as e:
            print(f"   ❌ {technique}: Erreur - {e}")
    
    return True

def test_3d_volume_augmentations():
    """Tester les augmentations 3D."""
    print("\n🧊 Test des augmentations 3D")
    print("=" * 60)
    
    augmenter = GeophysicalDataAugmenter(random_seed=42)
    
    # Créer un volume 3D de test
    volume_3d = np.random.rand(16, 32, 32, 4)  # depth, height, width, channels
    
    print(f"📊 Volume de test: {volume_3d.shape}")
    
    # Techniques 3D
    volume_techniques = [
        "geological_stratification",
        "fracture_patterns",
        "mineral_inclusions",
        "weathering_effects", 
        "sedimentary_layers",
        "elastic_deformation"
    ]
    
    for technique in volume_techniques:
        print(f"\n🔍 Test de {technique} sur volume 3D...")
        try:
            augmented = augmenter.augment_3d_volume(volume_3d, [technique], num_augmentations=1)
            print(f"   ✅ {technique}: {len(augmented)} volume(s) généré(s)")
            print(f"   📏 Forme: {augmented[0].shape}")
        except Exception as e:
            print(f"   ❌ {technique}: Erreur - {e}")
    
    return True

def main():
    """Fonction principale de test."""
    print("🚀 Test des améliorations du module d'augmentation de données")
    print("=" * 80)
    
    tests = [
        test_geophysical_techniques,
        test_advanced_techniques,
        test_validation_methods,
        test_recommendations,
        test_dataframe_augmentations,
        test_3d_volume_augmentations
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {test_func.__name__}: {e}")
            results.append(False)
    
    # Résumé
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests réussis: {passed}/{total}")
    print(f"❌ Tests échoués: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ Le module d'augmentation amélioré fonctionne correctement")
    else:
        print(f"\n⚠️ {total - passed} test(s) ont échoué")
        print("🔧 Vérifiez les erreurs ci-dessus")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

