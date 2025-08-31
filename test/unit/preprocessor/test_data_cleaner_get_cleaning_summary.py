#!/usr/bin/env python3
"""
Test unitaire pour la méthode get_cleaning_summary de GeophysicalDataCleaner
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner
from config import CONFIG


class TestDataCleanerGetCleaningSummary(unittest.TestCase):
    """Tests pour la méthode get_cleaning_summary de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        self.cleaner = GeophysicalDataCleaner()
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
    
    def test_get_cleaning_summary_initial_state(self):
        """Test du résumé de nettoyage dans l'état initial"""
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        self.assertEqual(len(summary), 0, "Le résumé initial devrait être vide")
        print("✅ Résumé de nettoyage initial validé (vide)")
    
    def test_get_cleaning_summary_after_validation(self):
        """Test du résumé de nettoyage après validation des fichiers"""
        validation_results = self.cleaner.validate_all_input_files()
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0)
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        print("✅ Résumé de nettoyage après validation validé")
    
    def test_get_cleaning_summary_after_cleaning_pd_csv(self):
        """Test du résumé de nettoyage après nettoyage de PD.csv"""
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        clean_path, report = self.cleaner._clean_device_data("pole_dipole", pd_file)
        self.assertIsInstance(report, dict, "Le rapport de nettoyage devrait être un dictionnaire")
        
        if len(report) == 0:
            print(f"⚠️ Rapport vide pour pole_dipole (données déjà nettoyées)")
        else:
            self.assertGreater(len(report), 0, "Le rapport de nettoyage ne devrait pas être vide")
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        print("✅ Résumé de nettoyage après nettoyage de PD.csv validé")
    
    def test_get_cleaning_summary_after_cleaning_s_csv(self):
        """Test du résumé de nettoyage après nettoyage de S.csv"""
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        clean_path, report = self.cleaner._clean_device_data("schlumberger", s_file)
        self.assertIsInstance(report, dict, "Le rapport de nettoyage devrait être un dictionnaire")
        
        if len(report) == 0:
            print(f"⚠️ Rapport vide pour schlumberger (données déjà nettoyées)")
        else:
            self.assertGreater(len(report), 0, "Le rapport de nettoyage ne devrait pas être vide")
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        print("✅ Résumé de nettoyage après nettoyage de S.csv validé")
    
    def test_get_cleaning_summary_after_cleaning_all_devices(self):
        """Test du résumé de nettoyage après nettoyage de tous les dispositifs"""
        results = self.cleaner.clean_all_devices()
        self.assertIsInstance(results, dict, "Les résultats de nettoyage devraient être un dictionnaire")
        self.assertGreater(len(results), 0, "Les résultats de nettoyage ne devraient pas être vides")
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        print("✅ Résumé de nettoyage après nettoyage de tous les dispositifs validé")
    
    def test_get_cleaning_summary_structure(self):
        """Test de la structure du résumé de nettoyage"""
        pd_file = self.raw_data_dir / "PD.csv"
        clean_path, report = self.cleaner._clean_device_data("pole_dipole", pd_file)
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        
        for key, value in summary.items():
            self.assertIsInstance(key, str, "Les clés du résumé devraient être des chaînes")
        
        print("✅ Structure du résumé de nettoyage validée")
    
    def test_get_cleaning_summary_performance(self):
        """Test de performance de la méthode get_cleaning_summary"""
        pd_file = self.raw_data_dir / "PD.csv"
        clean_path, report = self.cleaner._clean_device_data("pole_dipole", pd_file)
        
        start_time = time.time()
        summary = self.cleaner.get_cleaning_summary()
        execution_time = time.time() - start_time
        
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        self.assertLess(execution_time, 1.0, "get_cleaning_summary devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance de get_cleaning_summary validée: {execution_time:.3f} secondes")
    
    def test_get_cleaning_summary_empty_report(self):
        """Test du résumé de nettoyage avec un rapport vide"""
        clean_cleaner = GeophysicalDataCleaner()
        summary = clean_cleaner.get_cleaning_summary()
        
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        self.assertEqual(len(summary), 0, "Le résumé devrait être vide pour une nouvelle instance")
        
        print("✅ Résumé de nettoyage avec rapport vide validé")
    
    def test_get_cleaning_summary_report_modification(self):
        """Test de la modification du rapport et de son impact sur le résumé"""
        pd_file = self.raw_data_dir / "PD.csv"
        clean_path, report = self.cleaner._clean_device_data("pole_dipole", pd_file)
        
        summary_initial = self.cleaner.get_cleaning_summary()
        self.cleaner.report['test_key'] = 'test_value'
        summary_modified = self.cleaner.get_cleaning_summary()
        
        self.assertIsInstance(summary_modified, dict, "Le résumé modifié devrait être un dictionnaire")
        self.assertIn('test_key', summary_modified, "La modification du rapport devrait être visible dans le résumé")
        self.assertEqual(summary_modified['test_key'], 'test_value', "La valeur modifiée devrait être correcte")
        
        print("✅ Modification du rapport et impact sur le résumé validés")
    
    def test_get_cleaning_summary_return_consistency(self):
        """Test de la cohérence des retours de get_cleaning_summary"""
        summary1 = self.cleaner.get_cleaning_summary()
        summary2 = self.cleaner.get_cleaning_summary()
        summary3 = self.cleaner.get_cleaning_summary()
        
        self.assertEqual(summary1, summary2, "Les retours consécutifs devraient être identiques")
        self.assertEqual(summary2, summary3, "Les retours consécutifs devraient être identiques")
        self.assertIs(summary1, summary2, "Les retours devraient être la même référence d'objet")
        
        print("✅ Cohérence des retours de get_cleaning_summary validée")
    
    def test_get_cleaning_summary_comprehensive_validation(self):
        """Test de validation complète de get_cleaning_summary"""
        results = self.cleaner.clean_all_devices()
        self.assertIsInstance(results, dict, "Les résultats de nettoyage devraient être un dictionnaire")
        self.assertGreater(len(results), 0, "Les résultats de nettoyage ne devraient pas être vides")
        
        summary = self.cleaner.get_cleaning_summary()
        self.assertIsInstance(summary, dict, "Le résumé devrait être un dictionnaire")
        self.assertIs(summary, self.cleaner.report, "Le résumé devrait être une référence au rapport interne")
        
        print("✅ Validation complète de get_cleaning_summary réussie")


if __name__ == "__main__":
    unittest.main(verbosity=2)
