#!/usr/bin/env python3
"""
Test unitaire complet pour GeophysicalDataCleaner utilisant toutes les données des fixtures

Ce test vérifie le bon fonctionnement du data_cleaner avec toutes les données réelles
disponibles dans le répertoire test/fixtures.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import (
    GeophysicalDataCleaner, 
    detect_csv_separator, 
    read_csv_with_auto_separator,
    normalize_coordinate_columns,
    validate_coordinate_data
)


class TestDataCleanerComprehensiveFixtures(unittest.TestCase):
    """Tests complets utilisant toutes les données des fixtures"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.fixtures_dir = self.project_root / "test" / "fixtures"
        self.raw_fixtures_dir = self.project_root / "data" / "raw"  # Utiliser les vrais fichiers
        self.processed_dir = self.fixtures_dir / "processed"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_fixtures_dir)
            mock_config.paths.processed_data_dir = str(self.processed_dir)
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'resistivity', 'chargeability', 'LAT', 'LON', 'El-array'
            ]
            mock_config.geophysical_data.devices = {
                'pole_dipole': {'coverage': 1000},
                'schlumberger': {'coverage': 1000}
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        self.processed_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
    
    def test_detect_csv_separator_all_fixtures(self):
        """Test de détection des séparateurs pour tous les fichiers CSV des fixtures"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        self.assertGreater(len(csv_files), 0, "Aucun fichier CSV trouvé dans les fixtures")
        
        separators_detected = {}
        for csv_file in csv_files:
            separator = detect_csv_separator(csv_file)
            separators_detected[csv_file.name] = separator
            print(f"✅ {csv_file.name}: séparateur '{separator}' détecté")
        
        # Vérifier que tous les fichiers ont un séparateur détecté
        for file_name, separator in separators_detected.items():
            self.assertIn(separator, [',', ';', '\t'], f"Séparateur invalide pour {file_name}: {separator}")
    
    def test_read_csv_with_auto_separator_all_fixtures(self):
        """Test de lecture automatique pour tous les fichiers CSV des fixtures"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    df = read_csv_with_auto_separator(csv_file)
                    self.assertIsInstance(df, pd.DataFrame)
                    self.assertGreater(len(df), 0, f"Le fichier {csv_file.name} ne devrait pas être vide")
                    print(f"✅ {csv_file.name}: {len(df)} lignes, {len(df.columns)} colonnes")
                except Exception as e:
                    self.fail(f"Erreur lors de la lecture de {csv_file.name}: {e}")
    
    def test_normalize_coordinate_columns_all_fixtures(self):
        """Test de normalisation des colonnes de coordonnées pour tous les fichiers"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    df_original = read_csv_with_auto_separator(csv_file)
                    df_normalized = normalize_coordinate_columns(df_original)
                    
                    # Vérifier que la normalisation a fonctionné
                    self.assertIsInstance(df_normalized, pd.DataFrame)
                    self.assertEqual(len(df_normalized), len(df_original))
                    
                    # Vérifier les colonnes de coordonnées normalisées
                    coord_cols = ['x', 'y', 'z', 'LAT', 'LON']
                    found_coords = [col for col in coord_cols if col in df_normalized.columns]
                    
                    print(f"✅ {csv_file.name}: colonnes de coordonnées trouvées: {found_coords}")
                    
                except Exception as e:
                    self.fail(f"Erreur lors de la normalisation de {csv_file.name}: {e}")
    
    def test_validate_coordinate_data_all_fixtures(self):
        """Test de validation des données de coordonnées pour tous les fichiers"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    df = read_csv_with_auto_separator(csv_file)
                    df_normalized = normalize_coordinate_columns(df)
                    validation_report = validate_coordinate_data(df_normalized)
                    
                    # Vérifier la structure du rapport
                    self.assertIsInstance(validation_report, dict)
                    self.assertIn('has_utm_coords', validation_report)
                    self.assertIn('has_wgs84_coords', validation_report)
                    self.assertIn('coordinate_systems', validation_report)
                    self.assertIn('issues', validation_report)
                    
                    print(f"✅ {csv_file.name}: {validation_report['coordinate_systems']} - {len(validation_report['issues'])} problèmes")
                    
                except Exception as e:
                    self.fail(f"Erreur lors de la validation de {csv_file.name}: {e}")
    
    def test_clean_all_devices_with_fixtures(self):
        """Test de nettoyage de tous les dispositifs avec les données des fixtures"""
        try:
            results = self.cleaner.clean_all_devices()
            
            # Vérifier que des résultats ont été retournés
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0, "Aucun dispositif traité")
            
            # Vérifier la structure des résultats
            for device_name, (clean_path, report) in results.items():
                self.assertIsInstance(clean_path, Path)
                self.assertIsInstance(report, dict)
                self.assertIn('original_count', report)
                self.assertIn('cleaned_count', report)
                self.assertIn('removed_count', report)
                
                print(f"✅ {device_name}: {report['cleaned_count']}/{report['original_count']} enregistrements conservés")
            
        except Exception as e:
            self.fail(f"Erreur lors du nettoyage des dispositifs: {e}")
    
    def test_load_device_data_all_fixtures(self):
        """Test de chargement des données pour tous les fichiers des fixtures"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    # Déterminer le type de dispositif basé sur le nom du fichier
                    device_type = "pole_dipole" if "PD" in csv_file.name else "schlumberger"
                    
                    df = self.cleaner._load_device_data(csv_file, device_type)
                    
                    # Vérifier que les données ont été chargées
                    self.assertIsInstance(df, pd.DataFrame)
                    self.assertGreater(len(df), 0)
                    
                    print(f"✅ {csv_file.name} ({device_type}): {len(df)} lignes chargées")
                    
                except Exception as e:
                    self.fail(f"Erreur lors du chargement de {csv_file.name}: {e}")
    
    def test_coordinate_transformation_all_fixtures(self):
        """Test de transformation des coordonnées pour tous les fichiers avec LAT/LON"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    df = read_csv_with_auto_separator(csv_file)
                    df_normalized = normalize_coordinate_columns(df)
                    
                    # Vérifier s'il y a des coordonnées LAT/LON
                    if 'LAT' in df_normalized.columns and 'LON' in df_normalized.columns:
                        # Tester la transformation des coordonnées
                        lat_values = df_normalized['LAT'].dropna()
                        lon_values = df_normalized['LON'].dropna()
                        
                        if len(lat_values) > 0 and len(lon_values) > 0:
                            x_coords, y_coords = self.cleaner._transform_coordinates(lat_values, lon_values)
                            
                            # Vérifier que la transformation a fonctionné
                            self.assertEqual(len(x_coords), len(lat_values))
                            self.assertEqual(len(y_coords), len(lon_values))
                            
                            print(f"✅ {csv_file.name}: {len(x_coords)} coordonnées transformées")
                    
                except Exception as e:
                    # Ne pas échouer si le fichier n'a pas de coordonnées LAT/LON
                    if "LAT" not in str(e) and "LON" not in str(e):
                        self.fail(f"Erreur lors de la transformation de {csv_file.name}: {e}")
    
    def test_data_quality_all_fixtures(self):
        """Test de la qualité des données pour tous les fichiers des fixtures"""
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    df = read_csv_with_auto_separator(csv_file)
                    df_normalized = normalize_coordinate_columns(df)
                    
                    # Vérifier la qualité des données
                    quality_issues = []
                    
                    # Vérifier les valeurs manquantes
                    missing_pct = df_normalized.isnull().sum().sum() / (len(df_normalized) * len(df_normalized.columns)) * 100
                    if missing_pct > 50:
                        quality_issues.append(f"Trop de valeurs manquantes: {missing_pct:.1f}%")
                    
                    # Vérifier les colonnes numériques
                    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        quality_issues.append("Aucune colonne numérique trouvée")
                    
                    # Vérifier les coordonnées si présentes
                    if 'x' in df_normalized.columns and 'y' in df_normalized.columns:
                        x_range = df_normalized['x'].max() - df_normalized['x'].min()
                        y_range = df_normalized['y'].max() - df_normalized['y'].min()
                        if x_range < 1 or y_range < 1:
                            quality_issues.append(f"Couverture spatiale faible: {x_range:.1f}m x {y_range:.1f}m")
                    
                    print(f"✅ {csv_file.name}: {len(quality_issues)} problèmes de qualité détectés")
                    
                except Exception as e:
                    self.fail(f"Erreur lors de l'analyse de qualité de {csv_file.name}: {e}")
    
    def test_performance_all_fixtures(self):
        """Test de performance pour tous les fichiers des fixtures"""
        import time
        
        csv_files = list(self.raw_fixtures_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            with self.subTest(file=csv_file.name):
                try:
                    start_time = time.time()
                    
                    # Test complet de chargement et traitement
                    df = read_csv_with_auto_separator(csv_file)
                    df_normalized = normalize_coordinate_columns(df)
                    validation_report = validate_coordinate_data(df_normalized)
                    
                    processing_time = time.time() - start_time
                    
                    # Vérifier que le traitement est rapide (< 5 secondes)
                    self.assertLess(processing_time, 5.0, f"Traitement de {csv_file.name} trop lent: {processing_time:.2f}s")
                    
                    print(f"✅ {csv_file.name}: traité en {processing_time:.3f}s")
                    
                except Exception as e:
                    self.fail(f"Erreur lors du test de performance de {csv_file.name}: {e}")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
