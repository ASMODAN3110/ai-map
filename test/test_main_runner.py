#!/usr/bin/env python3
"""
Script pour exécuter tous les tests du fichier main.py.
Inclut les tests unitaires et d'intégration.
"""

import unittest
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_main_tests():
    """Exécuter tous les tests pour le fichier main.py"""
    
    # Découvrir et charger les tests unitaires
    unit_loader = unittest.TestLoader()
    unit_suite = unit_loader.discover(
        start_dir=Path(__file__).parent / "unit",
        pattern="test_main.py",
        top_level_dir=Path(__file__).parent
    )
    
    # Découvrir et charger les tests d'intégration
    integration_loader = unittest.TestLoader()
    integration_suite = integration_loader.discover(
        start_dir=Path(__file__).parent / "integration",
        pattern="test_main_integration.py",
        top_level_dir=Path(__file__).parent
    )
    
    # Combiner les suites de tests
    combined_suite = unittest.TestSuite()
    combined_suite.addTest(unit_suite)
    combined_suite.addTest(integration_suite)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 80)
    print("🧪 EXÉCUTION DES TESTS POUR MAIN.PY")
    print("=" * 80)
    print()
    
    result = runner.run(combined_suite)
    
    # Afficher le résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 80)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print("\n❌ ÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ TOUS LES TESTS ONT RÉUSSI!")
        return True
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ!")
        return False

def run_specific_test_class(test_class_name):
    """Exécuter une classe de test spécifique"""
    loader = unittest.TestLoader()
    
    # Essayer de charger depuis les tests unitaires
    try:
        suite = loader.loadTestsFromName(f"test.unit.test_main.{test_class_name}")
        if suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2)
            return runner.run(suite)
    except:
        pass
    
    # Essayer de charger depuis les tests d'intégration
    try:
        suite = loader.loadTestsFromName(f"test.integration.test_main_integration.{test_class_name}")
        if suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2)
            return runner.run(suite)
    except:
        pass
    
    print(f"❌ Classe de test '{test_class_name}' non trouvée!")
    return None

def run_specific_test_method(test_class_name, test_method_name):
    """Exécuter une méthode de test spécifique"""
    loader = unittest.TestLoader()
    
    # Essayer de charger depuis les tests unitaires
    try:
        suite = loader.loadTestsFromName(f"test.unit.test_main.{test_class_name}.{test_method_name}")
        if suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2)
            return runner.run(suite)
    except:
        pass
    
    # Essayer de charger depuis les tests d'intégration
    try:
        suite = loader.loadTestsFromName(f"test.integration.test_main_integration.{test_class_name}.{test_method_name}")
        if suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2)
            return runner.run(suite)
    except:
        pass
    
    print(f"❌ Méthode de test '{test_class_name}.{test_method_name}' non trouvée!")
    return None

def main():
    """Fonction principale"""
    if len(sys.argv) == 1:
        # Exécuter tous les tests
        success = run_main_tests()
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 2:
        # Exécuter une classe de test spécifique
        test_class = sys.argv[1]
        result = run_specific_test_class(test_class)
        if result:
            sys.exit(0 if result.wasSuccessful() else 1)
        else:
            sys.exit(1)
    
    elif len(sys.argv) == 3:
        # Exécuter une méthode de test spécifique
        test_class = sys.argv[1]
        test_method = sys.argv[2]
        result = run_specific_test_method(test_class, test_method)
        if result:
            sys.exit(0 if result.wasSuccessful() else 1)
        else:
            sys.exit(1)
    
    else:
        print("Usage:")
        print("  python test_main_runner.py                    # Exécuter tous les tests")
        print("  python test_main_runner.py TestClass          # Exécuter une classe de test")
        print("  python test_main_runner.py TestClass test_method  # Exécuter une méthode de test")
        sys.exit(1)

if __name__ == "__main__":
    main()
