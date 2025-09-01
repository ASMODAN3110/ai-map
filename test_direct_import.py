#!/usr/bin/env python3
"""
Test direct des méthodes de nettoyage avancées
=============================================
"""

import sys
import os

# Ajouter le répertoire src au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path}")

# Test d'import direct
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "image_processor", 
        os.path.join(src_dir, "data", "image_processor.py")
    )
    image_processor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_processor_module)
    
    GeophysicalImageProcessor = image_processor_module.GeophysicalImageProcessor
    print("✅ Import direct réussi de GeophysicalImageProcessor")
    
except Exception as e:
    print(f"❌ Erreur d'import direct: {e}")
    sys.exit(1)

def test_advanced_cleaning_methods():
    """Test des méthodes de nettoyage avancées."""
    print("\n🧪 Test des méthodes de nettoyage avancées...")
    
    # Créer le processeur
    processor = GeophysicalImageProcessor(target_size=(64, 64), channels=3)
    print(f"✅ Processeur créé: {processor.target_size}, {processor.channels} canaux")
    
    # Vérifier que les nouvelles méthodes existent
    new_methods = [
        'apply_noise_reduction',
        'correct_artifacts', 
        'enhance_contrast',
        'apply_geophysical_specific_cleaning',
        'get_cleaning_summary'
    ]
    
    print("\n📋 Vérification des nouvelles méthodes:")
    for method in new_methods:
        if hasattr(processor, method):
            print(f"  ✅ {method}")
        else:
            print(f"  ❌ {method} - MANQUANTE")
    
    # Test rapide d'une méthode
    print("\n🔍 Test rapide de apply_noise_reduction...")
    try:
        from PIL import Image
        import numpy as np
        
        # Image de test simple
        test_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        
        # Test de la réduction de bruit gaussienne
        cleaned = processor.apply_noise_reduction(test_img, method="gaussian")
        print("  ✅ apply_noise_reduction avec méthode gaussian: SUCCÈS")
        
        # Test de la correction d'artefacts
        corrected = processor.correct_artifacts(test_img, "salt_pepper")
        print("  ✅ correct_artifacts avec salt_pepper: SUCCÈS")
        
        # Test de l'amélioration du contraste
        enhanced = processor.enhance_contrast(test_img, method="histogram_equalization")
        print("  ✅ enhance_contrast avec histogram_equalization: SUCCÈS")
        
        print("\n🎉 Tous les tests de base sont réussis!")
        
    except Exception as e:
        print(f"  ❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_cleaning_methods()
