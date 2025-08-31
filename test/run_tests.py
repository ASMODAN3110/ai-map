#!/usr/bin/env python3
"""
Script principal pour exécuter tous les tests du projet AI-MAP

Usage:
    python test/run_tests.py                    # Exécuter tous les tests
    python test/run_tests.py --unit            # Tests unitaires uniquement
    python test/run_tests.py --integration     # Tests d'intégration uniquement
    python test/run_tests.py --e2e             # Tests end-to-end uniquement
    python test/run_tests.py --verbose         # Mode verbeux
"""

import sys
import argparse
import unittest
from pathlib import Path
from unittest import TestLoader, TestSuite, TextTestRunner

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_tests(test_dir: Path, pattern: str = "test_*.py") -> TestSuite:
    """Découvrir automatiquement tous les tests dans un répertoire"""
    loader = TestLoader()
    suite = TestSuite()
    
    if test_dir.exists():
        discovered = loader.discover(str(test_dir), pattern=pattern)
        suite.addTests(discovered)
    
    return suite


def run_unit_tests(verbose: bool = False) -> bool:
    """Exécuter tous les tests unitaires"""
    print("🧪 EXÉCUTION DES TESTS UNITAIRES")
    print("=" * 50)
    
    unit_dir = Path(__file__).parent / "unit"
    suite = discover_tests(unit_dir)
    
    if suite.countTestCases() == 0:
        print("⚠️ Aucun test unitaire trouvé")
        return True
    
    print(f"📊 {suite.countTestCases()} tests unitaires découverts")
    
    runner = TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests(verbose: bool = False) -> bool:
    """Exécuter tous les tests d'intégration"""
    print("\n🔗 EXÉCUTION DES TESTS D'INTÉGRATION")
    print("=" * 50)
    
    integration_dir = Path(__file__).parent / "integration"
    suite = discover_tests(integration_dir)
    
    if suite.countTestCases() == 0:
        print("⚠️ Aucun test d'intégration trouvé")
        return True
    
    print(f"📊 {suite.countTestCases()} tests d'intégration découverts")
    
    runner = TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_e2e_tests(verbose: bool = False) -> bool:
    """Exécuter tous les tests end-to-end"""
    print("\n🌐 EXÉCUTION DES TESTS END-TO-END")
    print("=" * 50)
    
    e2e_dir = Path(__file__).parent / "e2e"
    suite = discover_tests(e2e_dir)
    
    if suite.countTestCases() == 0:
        print("⚠️ Aucun test end-to-end trouvé")
        return True
    
    print(f"📊 {suite.countTestCases()} tests end-to-end découverts")
    
    runner = TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests(verbose: bool = False) -> bool:
    """Exécuter tous les tests"""
    print("🚀 EXÉCUTION DE TOUS LES TESTS AI-MAP")
    print("=" * 60)
    
    # Tests unitaires
    unit_success = run_unit_tests(verbose)
    
    # Tests d'intégration
    integration_success = run_integration_tests(verbose)
    
    # Tests end-to-end
    e2e_success = run_e2e_tests(verbose)
    
    # Résumé
    print("\n📋 RÉSUMÉ DES TESTS")
    print("=" * 30)
    print(f"🧪 Tests unitaires: {'✅ SUCCÈS' if unit_success else '❌ ÉCHEC'}")
    print(f"🔗 Tests d'intégration: {'✅ SUCCÈS' if integration_success else '❌ ÉCHEC'}")
    print(f"🌐 Tests end-to-end: {'✅ SUCCÈS' if e2e_success else '❌ ÉCHEC'}")
    
    overall_success = unit_success and integration_success and e2e_success
    
    if overall_success:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    else:
        print("\n💥 CERTAINS TESTS ONT ÉCHOUÉ!")
    
    return overall_success


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Exécuteur de tests pour AI-MAP")
    parser.add_argument("--unit", action="store_true", help="Exécuter uniquement les tests unitaires")
    parser.add_argument("--integration", action="store_true", help="Exécuter uniquement les tests d'intégration")
    parser.add_argument("--e2e", action="store_true", help="Exécuter uniquement les tests end-to-end")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    try:
        if args.unit:
            success = run_unit_tests(args.verbose)
        elif args.integration:
            success = run_integration_tests(args.verbose)
        elif args.e2e:
            success = run_e2e_tests(args.verbose)
        else:
            success = run_all_tests(args.verbose)
        
        # Code de sortie
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur lors de l'exécution des tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
