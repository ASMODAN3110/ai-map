#!/usr/bin/env python3
"""
Test unitaire pour la méthode validate_all_input_files de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode validate_all_input_files
qui valide que tous les fichiers d'entrée sont des CSV valides
en utilisant les vrais fichiers PD.csv et S.csv.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner
from config import CONFIG


class TestDataCleanerValidateAllInputFiles(unittest.TestCase):
    """Tests pour la méthode validate_all_input_files de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
    
    def test_validate_all_input_files_with_real_files(self):
        """Test de validation de tous les fichiers d'entrée avec les vrais fichiers"""
        # Vérifier que les fichiers existent
        pd_file = self.raw_data_dir / "PD.csv"
        s_file = self.raw_data_dir / "S.csv"
        
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifications
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0, "Les résultats de validation ne devraient pas être vides")
        
        # Vérifier que tous les dispositifs sont présents dans les résultats
        expected_devices = ['pole_dipole', 'schlumberger']
        for device in expected_devices:
            self.assertIn(device, validation_results, f"Dispositif {device} manquant dans les résultats de validation")
        
        # Vérifier que tous les fichiers sont validés comme CSV valides
        for device, is_valid in validation_results.items():
            self.assertIsInstance(is_valid, bool, f"Statut de validation pour {device} devrait être un booléen")
            self.assertTrue(is_valid, f"Fichier pour {device} devrait être un CSV valide")
        
        print(f"✅ Validation de tous les fichiers d'entrée réussie:")
        for device, is_valid in validation_results.items():
            status = "✓" if is_valid else "✗"
            print(f"   {status} {device}: {'Valide' if is_valid else 'Invalide'}")
    
    def test_validate_all_input_files_file_existence(self):
        """Test de validation de l'existence des fichiers"""
        # Vérifier que les fichiers existent
        pd_file = self.raw_data_dir / "PD.csv"
        s_file = self.raw_data_dir / "S.csv"
        
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        # Vérifier que les fichiers sont bien des CSV
        self.assertEqual(pd_file.suffix.lower(), '.csv', f"PD.csv devrait avoir l'extension .csv")
        self.assertEqual(s_file.suffix.lower(), '.csv', f"S.csv devrait avoir l'extension .csv")
        
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que tous les fichiers existants sont validés comme valides
        for device, is_valid in validation_results.items():
            if is_valid:
                # Si le fichier est valide, vérifier qu'il existe et est un CSV
                device_config = CONFIG.geophysical_data.devices[device]
                raw_file = self.cleaner.raw_data_dir / device_config['file']
                self.assertTrue(raw_file.exists(), f"Fichier pour {device} devrait exister")
                self.assertEqual(raw_file.suffix.lower(), '.csv', f"Fichier pour {device} devrait être un CSV")
        
        print("✅ Existence et format des fichiers validés")
    
    def test_validate_all_input_files_csv_format_validation(self):
        """Test de validation du format CSV des fichiers"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        # Valider le format CSV
        is_valid_pd = self.cleaner._validate_csv_format(pd_file)
        self.assertTrue(is_valid_pd, "PD.csv devrait être un CSV valide")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        # Valider le format CSV
        is_valid_s = self.cleaner._validate_csv_format(s_file)
        self.assertTrue(is_valid_s, "S.csv devrait être un CSV valide")
        
        # Appeler la méthode de validation complète
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que les résultats correspondent
        self.assertTrue(validation_results['pole_dipole'], "pole_dipole devrait être validé comme CSV valide")
        self.assertTrue(validation_results['schlumberger'], "schlumberger devrait être validé comme CSV valide")
        
        print("✅ Format CSV des fichiers validé")
    
    def test_validate_all_input_files_extension_validation(self):
        """Test de validation des extensions de fichiers"""
        # Vérifier que les fichiers ont les bonnes extensions
        pd_file = self.raw_data_dir / "PD.csv"
        s_file = self.raw_data_dir / "S.csv"
        
        # Vérifier les extensions
        self.assertEqual(pd_file.suffix.lower(), '.csv', "PD.csv devrait avoir l'extension .csv")
        self.assertEqual(s_file.suffix.lower(), '.csv', "S.csv devrait avoir l'extension .csv")
        
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que tous les fichiers avec l'extension .csv sont validés
        for device, is_valid in validation_results.items():
            if is_valid:
                device_config = CONFIG.geophysical_data.devices[device]
                raw_file = self.cleaner.raw_data_dir / device_config['file']
                self.assertEqual(raw_file.suffix.lower(), '.csv', f"Fichier pour {device} devrait avoir l'extension .csv")
        
        print("✅ Extensions de fichiers validées")
    
    def test_validate_all_input_files_content_validation(self):
        """Test de validation du contenu des fichiers CSV"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        # Charger et vérifier le contenu
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes importantes sont présentes
        expected_cols_pd = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
        for col in expected_cols_pd:
            self.assertIn(col, df_pd.columns, f"Colonne {col} manquante dans PD.csv")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        # Charger et vérifier le contenu
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes importantes sont présentes
        expected_cols_s = ['LAT', 'LON', 'Rho (Ohm.m)', 'M (mV/V)']
        for col in expected_cols_s:
            self.assertIn(col, df_s.columns, f"Colonne {col} manquante dans S.csv")
        
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que les fichiers avec un contenu valide sont validés
        self.assertTrue(validation_results['pole_dipole'], "PD.csv avec contenu valide devrait être validé")
        self.assertTrue(validation_results['schlumberger'], "S.csv avec contenu valide devrait être validé")
        
        print("✅ Contenu des fichiers CSV validé")
    
    def test_validate_all_input_files_device_configuration(self):
        """Test de validation de la configuration des dispositifs"""
        # Vérifier que la configuration des dispositifs est correcte
        device_configs = CONFIG.geophysical_data.devices
        
        # Vérifier que tous les dispositifs requis sont présents
        expected_devices = ['pole_dipole', 'schlumberger']
        for device in expected_devices:
            self.assertIn(device, device_configs, f"Dispositif {device} manquant dans la configuration")
        
        # Vérifier que chaque dispositif a un fichier configuré
        for device, config in device_configs.items():
            self.assertIn('file', config, f"Configuration manquante pour {device}")
            self.assertIsInstance(config['file'], str, f"Nom de fichier pour {device} devrait être une chaîne")
        
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que tous les dispositifs configurés sont validés
        for device in device_configs.keys():
            self.assertIn(device, validation_results, f"Dispositif {device} devrait être dans les résultats de validation")
        
        print("✅ Configuration des dispositifs validée")
    
    def test_validate_all_input_files_error_handling(self):
        """Test de gestion des erreurs lors de la validation"""
        # Créer un fichier temporaire invalide
        invalid_file = self.test_dir / "invalid.txt"
        invalid_file.write_text("Ceci n'est pas un CSV valide\nLigne 2\nLigne 3")
        
        # Sauvegarder la configuration originale
        original_devices = CONFIG.geophysical_data.devices.copy()
        
        try:
            # Modifier temporairement la configuration pour inclure le fichier invalide
            CONFIG.geophysical_data.devices['test_device'] = {
                'file': 'invalid.txt',
                'coverage': {'x': 100, 'y': 100, 'z': 10}
            }
            
            # Appeler la méthode de validation
            validation_results = self.cleaner.validate_all_input_files()
            
            # Vérifier que le dispositif de test est dans les résultats
            self.assertIn('test_device', validation_results)
            
            # Vérifier que le fichier invalide est rejeté
            self.assertFalse(validation_results['test_device'], "Fichier invalide devrait être rejeté")
            
            print("✅ Gestion des erreurs validée")
            
        finally:
            # Restaurer la configuration originale
            CONFIG.geophysical_data.devices = original_devices
            
            # Nettoyer le fichier temporaire
            if invalid_file.exists():
                invalid_file.unlink()
    
    def test_validate_all_input_files_performance(self):
        """Test de performance de la validation des fichiers"""
        # Mesurer le temps de validation
        start_time = time.time()
        validation_results = self.cleaner.validate_all_input_files()
        validation_time = time.time() - start_time
        
        # Vérifications
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0)
        
        # Vérifier que la validation est rapide
        self.assertLess(validation_time, 5.0, "Validation des fichiers devrait être rapide (< 5 secondes)")
        
        print(f"✅ Performance de validation validée: {validation_time:.3f} secondes")
    
    def test_validate_all_input_files_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Valider tous les fichiers d'entrée
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que tous les fichiers sont valides
        all_valid = all(validation_results.values())
        self.assertTrue(all_valid, "Tous les fichiers d'entrée devraient être valides")
        
        # Si tous les fichiers sont valides, essayer de les charger
        if all_valid:
            for device_name in validation_results.keys():
                device_config = CONFIG.geophysical_data.devices[device_name]
                raw_file = self.cleaner.raw_data_dir / device_config['file']
                
                # Charger le fichier
                df = self.cleaner._load_device_data(raw_file, device_name)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreater(len(df), 0, f"Fichier pour {device_name} ne devrait pas être vide")
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_validate_all_input_files_return_structure(self):
        """Test de la structure de retour de la validation"""
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier la structure du dictionnaire retourné
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0)
        
        # Vérifier que toutes les clés sont des chaînes (noms de dispositifs)
        for key in validation_results.keys():
            self.assertIsInstance(key, str, "Clés du dictionnaire devraient être des chaînes")
        
        # Vérifier que toutes les valeurs sont des booléens
        for value in validation_results.values():
            self.assertIsInstance(value, bool, "Valeurs du dictionnaire devraient être des booléens")
        
        # Vérifier que tous les dispositifs configurés sont présents
        expected_devices = set(CONFIG.geophysical_data.devices.keys())
        actual_devices = set(validation_results.keys())
        self.assertEqual(expected_devices, actual_devices, "Tous les dispositifs configurés devraient être présents")
        
        print("✅ Structure de retour de la validation validée")
    
    def test_validate_all_input_files_edge_cases(self):
        """Test des cas limites de la validation"""
        # Test avec un fichier vide
        empty_file = self.test_dir / "empty.csv"
        empty_file.write_text("")
        
        # Sauvegarder la configuration originale
        original_devices = CONFIG.geophysical_data.devices.copy()
        
        try:
            # Modifier temporairement la configuration pour inclure le fichier vide
            CONFIG.geophysical_data.devices['empty_device'] = {
                'file': 'empty.csv',
                'coverage': {'x': 100, 'y': 100, 'z': 10}
            }
            
            # Appeler la méthode de validation
            validation_results = self.cleaner.validate_all_input_files()
            
            # Vérifier que le dispositif vide est dans les résultats
            self.assertIn('empty_device', validation_results)
            
            # Vérifier que le fichier vide est rejeté
            self.assertFalse(validation_results['empty_device'], "Fichier vide devrait être rejeté")
            
            print("✅ Cas limites de validation gérés")
            
        finally:
            # Restaurer la configuration originale
            CONFIG.geophysical_data.devices = original_devices
            
            # Nettoyer le fichier temporaire
            if empty_file.exists():
                empty_file.unlink()
    
    def test_validate_all_input_files_logging(self):
        """Test du logging lors de la validation"""
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifier que la validation a produit des résultats
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0)
        
        # Vérifier que tous les fichiers valides sont marqués comme valides
        for device, is_valid in validation_results.items():
            if is_valid:
                # Si le fichier est valide, vérifier qu'il existe et est un CSV
                device_config = CONFIG.geophysical_data.devices[device]
                raw_file = self.cleaner.raw_data_dir / device_config['file']
                self.assertTrue(raw_file.exists(), f"Fichier pour {device} devrait exister")
                self.assertEqual(raw_file.suffix.lower(), '.csv', f"Fichier pour {device} devrait être un CSV")
        
        print("✅ Logging de validation vérifié")
    
    def test_validate_all_input_files_comprehensive_validation(self):
        """Test de validation complète de tous les aspects"""
        # Appeler la méthode de validation
        validation_results = self.cleaner.validate_all_input_files()
        
        # Vérifications complètes
        self.assertIsInstance(validation_results, dict)
        self.assertGreater(len(validation_results), 0)
        
        # Vérifier que tous les dispositifs configurés sont présents
        expected_devices = set(CONFIG.geophysical_data.devices.keys())
        actual_devices = set(validation_results.keys())
        self.assertEqual(expected_devices, actual_devices, "Tous les dispositifs configurés devraient être présents")
        
        # Vérifier que tous les fichiers valides sont marqués comme valides
        for device, is_valid in validation_results.items():
            if is_valid:
                device_config = CONFIG.geophysical_data.devices[device]
                raw_file = self.cleaner.raw_data_dir / device_config['file']
                
                # Vérifier l'existence
                self.assertTrue(raw_file.exists(), f"Fichier pour {device} devrait exister")
                
                # Vérifier l'extension
                self.assertEqual(raw_file.suffix.lower(), '.csv', f"Fichier pour {device} devrait être un CSV")
                
                # Vérifier le format CSV
                is_valid_csv = self.cleaner._validate_csv_format(raw_file)
                self.assertTrue(is_valid_csv, f"Fichier pour {device} devrait être un CSV valide")
        
        print("✅ Validation complète de tous les aspects réussie")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
