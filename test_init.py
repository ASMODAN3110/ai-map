#!/usr/bin/env python3
"""
Script pour exécuter le test test_data_cleaner_init.py
"""

import sys
import os
from pathlib import Path
import traceback

# Configuration
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def main():
    print("🚀 TEST INITIALISATION DATA CLEANER")
    print("=" * 60)
    
    try:
        import unittest
        from test.unit.preprocessor.test_data_cleaner_init import TestDataCleanerInit
        
        print("✅ Module de test importé avec succès")
        
        # Vérifier les fichiers de données
        pd_file = Path("data/raw/PD.csv")
        s_file = Path("data/raw/S.csv")
        print(f"✅ PD.csv existe: {pd_file.exists()}")
        print(f"✅ S.csv existe: {s_file.exists()}")
        
        # Créer la suite de tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDataCleanerInit)
        print(f"✅ Suite de tests créée avec {suite.countTestCases()} tests")
        
        # Lister tous les tests
        print(f"\n📋 Tests disponibles:")
        for i, test in enumerate(suite, 1):
            print(f"  {i}. {test}")
        
        print(f"\n🧪 DÉMARRAGE DE L'EXÉCUTION")
        print("=" * 60)
        
        # Exécuter les tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Résumé détaillé
        print(f"\n" + "=" * 60)
        print(f"📊 RÉSUMÉ COMPLET DES RÉSULTATS")
        print(f"=" * 60)
        print(f"📈 Tests exécutés: {result.testsRun}")
        print(f"✅ Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"❌ Échecs: {len(result.failures)}")
        print(f"🚨 Erreurs: {len(result.errors)}")
        
        # Détails des échecs
        if result.failures:
            print(f"\n❌ DÉTAILS DES ÉCHECS ({len(result.failures)}):")
            print("-" * 40)
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"\n{i}. {test}")
                print(f"   Traceback: {traceback}")
        
        # Détails des erreurs
        if result.errors:
            print(f"\n🚨 DÉTAILS DES ERREURS ({len(result.errors)}):")
            print("-" * 40)
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"\n{i}. {test}")
                print(f"   Traceback: {traceback}")
        
        # Statut final
        print(f"\n" + "=" * 60)
        if result.wasSuccessful():
            print("🎉 TOUS LES TESTS ONT RÉUSSI !")
            print("✅ L'initialisation du DataCleaner fonctionne parfaitement")
        else:
            print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
            print("🔧 Vérifiez les détails ci-dessus pour plus d'informations")
        print("=" * 60)
            
    except Exception as e:
        print(f"🚨 Erreur lors de l'exécution: {e}")
        print(f"Traceback complet:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
