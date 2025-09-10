#!/usr/bin/env python3
"""
Test unitaire pour la méthode _validate_csv_format de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _validate_csv_format
avec des fichiers CSV réels (PD.csv, S.csv) et des scénarios de validation.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner, read_csv_with_auto_separator, detect_csv_separator


class TestDataCleanerValidateCsvFormatRealData(unittest.TestCase):
    """Tests pour la méthode _validate_csv_format avec données réelles"""
    
    def setUp(self):
        """Configuration avant chaque test avec données réelles des fixtures"""
        # Utiliser les fichiers de données réels
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_validate_csv_format_pd_csv_real(self):
        """Test de validation du format CSV avec le vrai fichier PD.csv"""
        # Utiliser le vrai fichier PD.csv
        csv_file = self.raw_data_dir / "PD.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(csv_file)
        
        # Vérifications
        self.assertTrue(is_valid, "PD.csv devrait être un CSV valide")
        
        # Vérifier manuellement le contenu
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Vérifier qu'il y a des séparateurs (détection automatique)
        has_separators = any(',' in line or ';' in line for line in first_lines if line.strip())
        self.assertTrue(has_separators, "PD.csv devrait contenir des séparateurs CSV")
        
        # Tester la détection automatique des séparateurs
        detected_separator = detect_csv_separator(csv_file)
        self.assertIn(detected_separator, [',', ';', '\t'], f"Séparateur détecté invalide: {detected_separator}")
        
        print(f"✅ PD.csv validé: format CSV correct avec séparateurs virgule")
    
    def test_validate_csv_format_s_csv_real(self):
        """Test de validation du format CSV avec le vrai fichier S.csv"""
        # Utiliser le vrai fichier S.csv
        csv_file = self.raw_data_dir / "S.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(csv_file)
        
        # Vérifications
        self.assertTrue(is_valid, "S.csv devrait être un CSV valide")
        
        # Vérifier manuellement le contenu
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Vérifier qu'il y a des séparateurs (détection automatique)
        has_separators = any(',' in line or ';' in line for line in first_lines if line.strip())
        self.assertTrue(has_separators, "S.csv devrait contenir des séparateurs CSV")
        
        # Tester la détection automatique des séparateurs
        detected_separator = detect_csv_separator(csv_file)
        self.assertIn(detected_separator, [',', ';', '\t'], f"Séparateur détecté invalide: {detected_separator}")
        
        print(f"✅ S.csv validé: format CSV correct avec séparateurs détectés automatiquement")
    
    def test_read_csv_with_auto_separator_fixtures(self):
        """Test de lecture automatique des CSV avec les données des fixtures"""
        # Tester tous les fichiers CSV des fixtures
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        self.assertGreater(len(csv_files), 0, "Aucun fichier CSV trouvé dans les fixtures")
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    # Tester la lecture automatique
                    df = read_csv_with_auto_separator(csv_file)
                    
                    # Vérifications
                    self.assertIsInstance(df, pd.DataFrame)
                    self.assertGreater(len(df), 0, f"{csv_file.name} ne devrait pas être vide")
                    
                    # Vérifier la détection du séparateur
                    detected_separator = detect_csv_separator(csv_file)
                    self.assertIn(detected_separator, [',', ';', '\t'])
                    
                    print(f"✅ {csv_file.name}: {len(df)} lignes, séparateur '{detected_separator}'")
                    
                except Exception as e:
                    self.fail(f"Erreur lors de la lecture de {csv_file.name}: {e}")
    
    def test_validate_csv_format_profile_files_real(self):
        """Test de validation du format CSV avec les vrais fichiers de profils"""
        # Chercher les fichiers de profils dans les fixtures
        fixtures_dir = self.test_dir / "raw"
        if fixtures_dir.exists():
            profile_files = list(fixtures_dir.glob("profil*.csv"))
            
            for profile_file in profile_files:
                with self.subTest(file=profile_file.name):
                    # Appeler la méthode de validation
                    is_valid = self.cleaner._validate_csv_format(profile_file)
                    
                    # Vérifications
                    self.assertTrue(is_valid, f"{profile_file.name} devrait être un CSV valide")
                    
                    # Vérifier manuellement le contenu
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline() for _ in range(3)]
                    
                    # Vérifier qu'il y a des séparateurs
                    has_separators = any(',' in line or ';' in line for line in first_lines if line.strip())
                    self.assertTrue(has_separators, f"{profile_file.name} devrait contenir des séparateurs CSV")
                    
                    print(f"✅ {profile_file.name} validé: format CSV correct")
        else:
            self.skipTest("Dossier fixtures/raw non trouvé")
    
    def test_validate_csv_format_invalid_file(self):
        """Test de validation avec un fichier invalide"""
        # Créer un fichier temporaire invalide
        temp_file = self.test_dir / "invalid_file.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write("Ceci n'est pas un fichier CSV\n")
            f.write("Pas de séparateurs\n")
            f.write("Juste du texte\n")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(temp_file)
        
        # Vérifications
        self.assertFalse(is_valid, "Le fichier texte devrait être invalide")
        
        # Nettoyer
        temp_file.unlink()
        
        print(f"✅ Fichier invalide correctement rejeté")
    
    def test_validate_csv_format_empty_file(self):
        """Test de validation avec un fichier vide"""
        # Créer un fichier temporaire vide
        temp_file = self.test_dir / "empty_file.csv"
        temp_file.touch()
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(temp_file)
        
        # Vérifications
        self.assertFalse(is_valid, "Le fichier vide devrait être invalide")
        
        # Nettoyer
        temp_file.unlink()
        
        print(f"✅ Fichier vide correctement rejeté")
    
    def test_validate_csv_format_mixed_separators(self):
        """Test de validation avec des séparateurs mixtes"""
        # Créer un fichier temporaire avec séparateurs mixtes
        temp_file = self.test_dir / "mixed_separators.csv"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write("col1,col2;col3\n")
            f.write("val1,val2;val3\n")
            f.write("data1,data2;data3\n")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(temp_file)
        
        # Vérifications (devrait être valide car il y a des séparateurs)
        self.assertTrue(is_valid, "Le fichier avec séparateurs mixtes devrait être valide")
        
        # Nettoyer
        temp_file.unlink()
        
        print(f"✅ Fichier avec séparateurs mixtes correctement validé")
    
    def test_validate_csv_format_encoding_issues(self):
        """Test de validation avec des problèmes d'encodage"""
        # Créer un fichier temporaire avec des caractères spéciaux
        temp_file = self.test_dir / "encoding_test.csv"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write("col1;col2;col3\n")
                f.write("val1;val2;val3\n")
                f.write("données;mesures;géophysiques\n")  # Caractères accentués
            
            # Appeler la méthode de validation
            is_valid = self.cleaner._validate_csv_format(temp_file)
            
            # Vérifications
            self.assertTrue(is_valid, "Le fichier avec caractères spéciaux devrait être valide")
            
            print(f"✅ Fichier avec caractères spéciaux correctement validé")
            
        except UnicodeEncodeError:
            self.skipTest("Problème d'encodage sur ce système")
        finally:
            # Nettoyer
            if temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
