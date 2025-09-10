#!/usr/bin/env python3
"""
Script pour identifier et corriger les tests qui échouent
"""

import sys
import os
import unittest
import traceback
import time
from pathlib import Path

# Configuration
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def run_test_with_details(test_file):
    """Exécute un test avec détails complets"""
    print(f"\n{'='*80}")
    print(f"🧪 EXÉCUTION DÉTAILLÉE: {test_file}")
    print(f"{'='*80}")
    
    try:
        # Importer le module de test
        module_name = test_file.replace('.py', '').replace('/', '.').replace('\\', '.')
        test_module = __import__(module_name, fromlist=[''])
        
        # Trouver la classe de test
        test_class = None
        for attr_name in dir(test_module):
            attr = getattr(test_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, unittest.TestCase) and 
                attr != unittest.TestCase):
                test_class = attr
                break
        
        if test_class is None:
            print(f"❌ Aucune classe de test trouvée dans {test_file}")
            return False
        
        # Créer la suite de tests
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Exécuter les tests avec output détaillé
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution de {test_file}: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔧 IDENTIFICATION ET CORRECTION DES TESTS QUI ÉCHOUENT")
    print("=" * 80)
    
    # Tests identifiés comme échouant d'après les résultats précédents
    failing_tests = [
        "test/unit/preprocessor/test_data_cleaner_load_device_data.py",
        "test/unit/preprocessor/test_data_cleaner_get_spatial_bounds_real_data.py", 
        "test/unit/preprocessor/test_data_cleaner_validate_columns.py"
    ]
    
    print(f"📋 {len(failing_tests)} tests identifiés comme échouant...")
    
    for i, test_file in enumerate(failing_tests, 1):
        print(f"\n[{i}/{len(failing_tests)}] Analyse de {test_file}...")
        
        success = run_test_with_details(test_file)
        
        if success:
            print(f"✅ {test_file}: Maintenant réussi !")
        else:
            print(f"❌ {test_file}: Nécessite des corrections")
    
    print(f"\n{'='*80}")
    print("🎯 ANALYSE TERMINÉE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

