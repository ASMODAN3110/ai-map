#!/usr/bin/env python3
"""
Script pour exécuter tous les tests du data cleaner et corriger ceux qui échouent
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

def run_single_test(test_file):
    """Exécute un test individuel et retourne les résultats"""
    print(f"\n{'='*80}")
    print(f"🧪 EXÉCUTION: {test_file}")
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
            return {
                'file': test_file,
                'status': 'ERROR',
                'message': 'Aucune classe de test trouvée',
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
        
        # Créer la suite de tests
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Exécuter les tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        end_time = time.time()
        
        return {
            'file': test_file,
            'status': 'SUCCESS' if result.wasSuccessful() else 'FAILED',
            'message': 'Tous les tests ont réussi' if result.wasSuccessful() else f'{len(result.failures)} échecs, {len(result.errors)} erreurs',
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'execution_time': end_time - start_time,
            'failures_details': result.failures,
            'errors_details': result.errors
        }
        
    except Exception as e:
        return {
            'file': test_file,
            'status': 'ERROR',
            'message': f'Erreur d\'importation: {str(e)}',
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'execution_time': 0
        }

def main():
    print("🚀 EXÉCUTION DE TOUS LES TESTS DATA CLEANER")
    print("=" * 80)
    
    # Liste de tous les tests de data cleaner
    test_files = [
        "test/unit/preprocessor/test_data_cleaner_init.py",
        "test/unit/preprocessor/test_data_cleaner_load_device_data.py",
        "test/unit/preprocessor/test_data_cleaner_handle_missing_values.py",
        "test/unit/preprocessor/test_data_cleaner_normalize_geophysical_values.py",
        "test/unit/preprocessor/test_data_cleaner_get_value_ranges_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_get_spatial_bounds_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_calculate_coverage_area_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_validate_columns.py",
        "test/unit/preprocessor/test_data_cleaner_validate_csv_format_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_validate_spatial_coverage_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_clean_coordinates.py",
        "test/unit/preprocessor/test_data_cleaner_remove_outliers.py",
        "test/unit/preprocessor/test_data_cleaner_transform_coordinates_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_clean_device_data.py",
        "test/unit/preprocessor/test_data_cleaner_clean_all_devices.py",
        "test/unit/preprocessor/test_data_cleaner_clean_profile_data_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_clean_profile_files_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_find_device_files_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_create_dummy_data_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_generate_synthetic_geophysical_data_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_generate_synthetic_data_for_training_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_create_2d_grid_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_create_3d_volume_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_prepare_unet_2d_data_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_prepare_voxnet_3d_data_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_prepare_data_for_generators_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_prepare_data_for_generators_from_df_real_data.py",
        "test/unit/preprocessor/test_data_cleaner_comprehensive_fixtures.py"
    ]
    
    results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_time = 0
    
    print(f"📋 {len(test_files)} tests à exécuter...")
    
    # Exécuter chaque test
    for i, test_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] Exécution de {test_file}...")
        
        result = run_single_test(test_file)
        results.append(result)
        
        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']
        total_time += result.get('execution_time', 0)
        
        # Afficher le statut
        if result['status'] == 'SUCCESS':
            print(f"✅ {result['file']}: {result['message']} ({result['tests_run']} tests)")
        elif result['status'] == 'FAILED':
            print(f"❌ {result['file']}: {result['message']}")
        else:
            print(f"🚨 {result['file']}: {result['message']}")
    
    # Résumé final
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ FINAL")
    print(f"{'='*80}")
    print(f"⏱️  Temps total d'exécution: {total_time:.2f} secondes")
    print(f"📈 Tests exécutés: {total_tests}")
    print(f"✅ Succès: {len([r for r in results if r['status'] == 'SUCCESS'])}")
    print(f"❌ Échecs: {len([r for r in results if r['status'] == 'FAILED'])}")
    print(f"🚨 Erreurs: {len([r for r in results if r['status'] == 'ERROR'])}")
    
    # Détails des échecs
    failed_tests = [r for r in results if r['status'] == 'FAILED']
    if failed_tests:
        print(f"\n❌ TESTS EN ÉCHEC ({len(failed_tests)}):")
        print("-" * 40)
        for result in failed_tests:
            print(f"  • {result['file']}: {result['message']}")
    
    # Détails des erreurs
    error_tests = [r for r in results if r['status'] == 'ERROR']
    if error_tests:
        print(f"\n🚨 TESTS EN ERREUR ({len(error_tests)}):")
        print("-" * 40)
        for result in error_tests:
            print(f"  • {result['file']}: {result['message']}")
    
    print(f"\n{'='*80}")
    if total_failures == 0 and total_errors == 0:
        print("🎉 TOUS LES TESTS ONT RÉUSSI !")
    else:
        print(f"⚠️  {total_failures + total_errors} TESTS NÉCESSITENT DES CORRECTIONS")
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    results = main()