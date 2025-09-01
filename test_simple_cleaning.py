#!/usr/bin/env python3
"""
Test simple des méthodes de nettoyage avancées
=============================================
"""

import sys
import os

# Ajouter le répertoire src au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from data.image_processor import GeophysicalImageProcessor
    print("✅ Import réussi de GeophysicalImageProcessor")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def test_basic_functionality():
    """Test des fonctionnalités de base."""
    print("\n🧪 Test des fonctionnalités de base...")
    
    # Créer le processeur
    processor = GeophysicalImageProcessor(target_size=(64, 64), channels=3)
    print(f"✅ Processeur créé: {processor.target_size}, {processor.channels} canaux")
    
    # Vérifier que les méthodes existent
    methods_to_check = [
        'apply_noise_reduction',
        'correct_artifacts', 
        'enhance_contrast',
        'apply_geophysical_specific_cleaning',
        'get_cleaning_summary'
    ]
    
    for method in methods_to_check:
        if hasattr(processor, method):
            print(f"✅ Méthode {method} disponible")
        else:
            print(f"❌ Méthode {method} manquante")
    
    return processor

def test_noise_reduction_methods(processor):
    """Test des méthodes de réduction de bruit."""
    print("\n🔍 Test des méthodes de réduction de bruit...")
    
    # Créer une image de test simple
    from PIL import Image
    import numpy as np
    
    # Image 64x64 avec du bruit
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    methods = ["gaussian", "median", "bilateral", "wiener", "non_local_means"]
    
    for method in methods:
        try:
            print(f"  - Test de {method}...")
            cleaned = processor.apply_noise_reduction(test_image, method=method)
            print(f"    ✅ {method}: Succès")
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")

def test_artifact_correction(processor):
    """Test de la correction d'artefacts."""
    print("\n🔧 Test de la correction d'artefacts...")
    
    from PIL import Image
    import numpy as np
    
    # Image de test
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    artifacts = ["scan_lines", "salt_pepper", "streaking", "banding"]
    
    for artifact in artifacts:
        try:
            print(f"  - Correction de {artifact}...")
            corrected = processor.correct_artifacts(test_image, artifact)
            print(f"    ✅ {artifact}: Succès")
        except Exception as e:
            print(f"    ❌ {artifact}: Erreur - {e}")

def test_contrast_enhancement(processor):
    """Test de l'amélioration du contraste."""
    print("\n✨ Test de l'amélioration du contraste...")
    
    from PIL import Image
    import numpy as np
    
    # Image de test
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    methods = ["histogram_equalization", "adaptive_histogram", "clahe", "gamma_correction"]
    
    for method in methods:
        try:
            print(f"  - Test de {method}...")
            enhanced = processor.enhance_contrast(test_image, method=method)
            print(f"    ✅ {method}: Succès")
        except Exception as e:
            print(f"    ❌ {method}: Erreur - {e}")

def test_pipeline_complet(processor):
    """Test du pipeline de nettoyage complet."""
    print("\n🚀 Test du pipeline de nettoyage complet...")
    
    from PIL import Image
    import numpy as np
    
    # Image de test
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    cleaning_steps = [
        "noise_reduction",
        "scan_lines_removal",
        "contrast_enhancement",
        "salt_pepper_removal"
    ]
    
    try:
        # Appliquer le pipeline
        cleaned_image = processor.apply_geophysical_specific_cleaning(test_image, cleaning_steps)
        print("  ✅ Pipeline de nettoyage appliqué avec succès!")
        
        # Obtenir le résumé
        summary = processor.get_cleaning_summary(test_image, cleaning_steps)
        print(f"  📊 Résumé obtenu: {len(summary)} métriques")
        
    except Exception as e:
        print(f"  ❌ Erreur lors du pipeline: {e}")

def main():
    """Fonction principale."""
    print("🧪 Test des méthodes de nettoyage avancées de GeophysicalImageProcessor")
    print("=" * 70)
    
    # Test 1: Fonctionnalités de base
    processor = test_basic_functionality()
    
    # Test 2: Réduction de bruit
    test_noise_reduction_methods(processor)
    
    # Test 3: Correction d'artefacts
    test_artifact_correction(processor)
    
    # Test 4: Amélioration du contraste
    test_contrast_enhancement(processor)
    
    # Test 5: Pipeline complet
    test_pipeline_complet(processor)
    
    print("\n🎉 Tous les tests sont terminés!")

if __name__ == "__main__":
    main()
