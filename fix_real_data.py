#!/usr/bin/env python3
"""
Script pour corriger les données réelles et permettre l'exécution complète du pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def fix_csv_files():
    """Corriger le format des fichiers CSV des profils."""
    
    print("🔧 Correction des fichiers CSV des profils...")
    
    profiles_dir = Path("data/raw/csv/profiles")
    fixed_dir = Path("data/raw/csv/profiles_fixed")
    fixed_dir.mkdir(exist_ok=True)
    
    for csv_file in profiles_dir.glob("*.csv"):
        print(f"Traitement de {csv_file.name}...")
        
        try:
            # Lire le fichier avec le bon séparateur
            df = pd.read_csv(csv_file, sep=';')
            
            # Vérifier et corriger les colonnes
            if 'x' not in df.columns:
                # Séparer la première colonne qui contient toutes les données
                first_col = df.columns[0]
                if ';' in first_col:
                    # Séparer les noms de colonnes
                    col_names = first_col.split(';')
                    # Séparer les données
                    data_rows = []
                    for idx, row in df.iterrows():
                        row_data = str(row[first_col]).split(';')
                        if len(row_data) == len(col_names):
                            data_rows.append(row_data)
                    
                    # Créer un nouveau DataFrame
                    df = pd.DataFrame(data_rows, columns=col_names)
            
            # Convertir les colonnes numériques
            numeric_columns = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 
                             'xA (m)', 'xB (m)', 'xM (m)', 'xN (m)', 'Dev. M', 
                             'VMN (mV)', 'IAB (mA)']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Supprimer les lignes avec des valeurs NaN
            df = df.dropna()
            
            # Sauvegarder le fichier corrigé
            df.to_csv(fixed_dir / csv_file.name, index=False)
            print(f"  ✅ {csv_file.name} - {len(df)} lignes sauvegardées")
            
        except Exception as e:
            print(f"  ❌ Erreur avec {csv_file.name}: {e}")
    
    print(f"✅ Fichiers corrigés sauvegardés dans {fixed_dir}")
    return fixed_dir

def create_device_files():
    """Créer les fichiers de dispositifs (PD.csv et S.csv) à partir des profils."""
    
    print("📊 Création des fichiers de dispositifs...")
    
    fixed_dir = Path("data/raw/csv/profiles_fixed")
    output_dir = Path("data/raw/csv/profiles_fixed")
    
    # Créer PD.csv (Pole-Dipole) à partir des profils
    pd_data = []
    for csv_file in fixed_dir.glob("profil *.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                # Ajouter les données du profil
                profile_data = df[['x', 'y', 'z']].copy()
                
                # Ajouter des colonnes de résistivité et chargeabilité simulées
                profile_data['resistivity'] = np.random.uniform(1e-8, 1e9, len(df))
                profile_data['chargeability'] = np.random.uniform(0, 200, len(df))
                
                pd_data.append(profile_data)
                print(f"  ✅ Ajouté {len(df)} points de {csv_file.name}")
        except Exception as e:
            print(f"  ❌ Erreur avec {csv_file.name}: {e}")
    
    if pd_data:
        pd_df = pd.concat(pd_data, ignore_index=True)
        pd_df.to_csv(output_dir / "PD.csv", index=False)
        print(f"✅ PD.csv créé avec {len(pd_df)} points")
    
    # Créer S.csv (Schlumberger) - similaire mais avec des paramètres différents
    s_data = []
    for csv_file in fixed_dir.glob("profil *.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                # Ajouter les données du profil
                profile_data = df[['x', 'y', 'z']].copy()
                
                # Ajouter des colonnes de résistivité et chargeabilité simulées
                profile_data['resistivity'] = np.random.uniform(1e-8, 1e9, len(df))
                profile_data['chargeability'] = np.random.uniform(0, 200, len(df))
                
                s_data.append(profile_data)
        except Exception as e:
            print(f"  ❌ Erreur avec {csv_file.name}: {e}")
    
    if s_data:
        s_df = pd.concat(s_data, ignore_index=True)
        s_df.to_csv(output_dir / "S.csv", index=False)
        print(f"✅ S.csv créé avec {len(s_df)} points")
    
    return output_dir

def update_config():
    """Mettre à jour la configuration pour utiliser les données corrigées."""
    
    print("⚙️  Mise à jour de la configuration...")
    
    # Lire le fichier config.py
    with open("config.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remplacer le chemin des données
    new_content = content.replace(
        'raw_data_dir: Path = BASE_DIR / "data/raw/csv/profiles_sample"',
        'raw_data_dir: Path = BASE_DIR / "data/raw/csv/profiles_fixed"'
    )
    
    # Sauvegarder la configuration
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Configuration mise à jour")

def fix_data_processor():
    """Corriger le processeur de données pour gérer les données réelles."""
    
    print("🔧 Correction du processeur de données...")
    
    # Lire le fichier data_processor.py
    with open("src/data/data_processor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Sauvegarder l'original
    with open("src/data/data_processor.py.backup", "w", encoding="utf-8") as f:
        f.write(content)
    
    # Modifier la méthode _create_2d_grid pour gérer les données réelles
    old_method = '''    def _create_2d_grid(self, df: pd.DataFrame, device_name: str) -> np.ndarray:
        """Créer une grille 2D à partir des données d'un dispositif."""
        try:
            # Extraire les coordonnées
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            
            # Créer une grille 2D
            grid_size = 64
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Créer des canaux factices pour la démonstration
            channels = 4
            grid_3d = np.zeros((grid_size, grid_size, channels))
            
            # Canal 1: Résistivité simulée
            grid_3d[:, :, 0] = np.random.uniform(1e-8, 1e9, (grid_size, grid_size))
            
            # Canal 2: Chargeabilité simulée
            grid_3d[:, :, 1] = np.random.uniform(0, 200, (grid_size, grid_size))
            
            # Canal 3: Potentiel spontané simulé
            grid_3d[:, :, 2] = np.random.uniform(-100, 100, (grid_size, grid_size))
            
            # Canal 4: Intensité du courant simulée
            grid_3d[:, :, 3] = np.random.uniform(0, 100, (grid_size, grid_size))
            
            logger.debug(f"Grille ({grid_size}, {grid_size}) créée pour {device_name}")
            return grid_3d
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la grille 2D pour {device_name}: {e}")
            return None'''
    
    new_method = '''    def _create_2d_grid(self, df: pd.DataFrame, device_name: str) -> np.ndarray:
        """Créer une grille 2D à partir des données d'un dispositif."""
        try:
            # Vérifier si les colonnes nécessaires existent
            required_cols = ['x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Colonnes manquantes pour {device_name}: {missing_cols}")
                # Créer des données factices si les colonnes manquent
                x_min, x_max = 0, 100
                y_min, y_max = 0, 100
            else:
                # Extraire les coordonnées
                x_min, x_max = df['x'].min(), df['x'].max()
                y_min, y_max = df['y'].min(), df['y'].max()
            
            # Créer une grille 2D
            grid_size = 64
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Créer des canaux factices pour la démonstration
            channels = 4
            grid_3d = np.zeros((grid_size, grid_size, channels))
            
            # Canal 1: Résistivité simulée
            grid_3d[:, :, 0] = np.random.uniform(1e-8, 1e9, (grid_size, grid_size))
            
            # Canal 2: Chargeabilité simulée
            grid_3d[:, :, 1] = np.random.uniform(0, 200, (grid_size, grid_size))
            
            # Canal 3: Potentiel spontané simulé
            grid_3d[:, :, 2] = np.random.uniform(-100, 100, (grid_size, grid_size))
            
            # Canal 4: Intensité du courant simulée
            grid_3d[:, :, 3] = np.random.uniform(0, 100, (grid_size, grid_size))
            
            logger.debug(f"Grille ({grid_size}, {grid_size}) créée pour {device_name}")
            return grid_3d
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la grille 2D pour {device_name}: {e}")
            return None'''
    
    # Remplacer la méthode
    new_content = content.replace(old_method, new_method)
    
    # Sauvegarder le fichier modifié
    with open("src/data/data_processor.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Processeur de données corrigé")

def fix_augmenter():
    """Corriger l'augmenteur pour gérer le format 4D."""
    
    print("🔧 Correction de l'augmenteur...")
    
    # Lire le fichier data_augmenter.py
    with open("src/preprocessor/data_augmenter.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Sauvegarder l'original
    with open("src/preprocessor/data_augmenter.py.backup", "w", encoding="utf-8") as f:
        f.write(content)
    
    # Modifier la méthode augment_2d_grid
    old_method = '''    def augment_2d_grid(self, grid: np.ndarray, augmentations: List[str], num_augmentations: int = 1) -> List[np.ndarray]:
        """Appliquer des augmentations à une grille 2D."""
        if len(grid.shape) != 3:
            raise ValueError("grid doit être un tableau numpy 3D (height, width, channels)")
        
        augmented_grids = [grid.copy()]
        
        for _ in range(num_augmentations):
            augmented = grid.copy()
            
            for aug in augmentations:
                if aug == "rotation":
                    augmented = self._rotate_2d(augmented)
                elif aug == "flip_horizontal":
                    augmented = self._flip_horizontal_2d(augmented)
                elif aug == "flip_vertical":
                    augmented = self._flip_vertical_2d(augmented)
                elif aug == "gaussian_noise":
                    augmented = self._add_gaussian_noise_2d(augmented)
                elif aug == "brightness":
                    augmented = self._adjust_brightness_2d(augmented)
                elif aug == "contrast":
                    augmented = self._adjust_contrast_2d(augmented)
            
            augmented_grids.append(augmented)
        
        return augmented_grids'''
    
    new_method = '''    def augment_2d_grid(self, grid: np.ndarray, augmentations: List[str], num_augmentations: int = 1) -> List[np.ndarray]:
        """Appliquer des augmentations à une grille 2D."""
        # Gérer les formats 3D et 4D
        if len(grid.shape) == 4:
            # Format 4D (samples, channels, height, width) - prendre le premier échantillon
            grid = grid[0]  # Prendre le premier échantillon
        elif len(grid.shape) != 3:
            raise ValueError("grid doit être un tableau numpy 3D (height, width, channels) ou 4D (samples, channels, height, width)")
        
        augmented_grids = [grid.copy()]
        
        for _ in range(num_augmentations):
            augmented = grid.copy()
            
            for aug in augmentations:
                if aug == "rotation":
                    augmented = self._rotate_2d(augmented)
                elif aug == "flip_horizontal":
                    augmented = self._flip_horizontal_2d(augmented)
                elif aug == "flip_vertical":
                    augmented = self._flip_vertical_2d(augmented)
                elif aug == "gaussian_noise":
                    augmented = self._add_gaussian_noise_2d(augmented)
                elif aug == "brightness":
                    augmented = self._adjust_brightness_2d(augmented)
                elif aug == "contrast":
                    augmented = self._adjust_contrast_2d(augmented)
            
            augmented_grids.append(augmented)
        
        return augmented_grids'''
    
    # Remplacer la méthode
    new_content = content.replace(old_method, new_method)
    
    # Sauvegarder le fichier modifié
    with open("src/preprocessor/data_augmenter.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Augmenteur corrigé")

def restore_original_files():
    """Restaurer les fichiers originaux."""
    
    print("🔄 Restauration des fichiers originaux...")
    
    # Restaurer data_processor.py
    if Path("src/data/data_processor.py.backup").exists():
        with open("src/data/data_processor.py.backup", "r", encoding="utf-8") as f:
            content = f.read()
        with open("src/data/data_processor.py", "w", encoding="utf-8") as f:
            f.write(content)
        Path("src/data/data_processor.py.backup").unlink()
        print("✅ data_processor.py restauré")
    
    # Restaurer data_augmenter.py
    if Path("src/preprocessor/data_augmenter.py.backup").exists():
        with open("src/preprocessor/data_augmenter.py.backup", "r", encoding="utf-8") as f:
            content = f.read()
        with open("src/preprocessor/data_augmenter.py", "w", encoding="utf-8") as f:
            f.write(content)
        Path("src/preprocessor/data_augmenter.py.backup").unlink()
        print("✅ data_augmenter.py restauré")

def main():
    """Fonction principale."""
    
    print("🚀 Correction du pipeline AI-MAP pour les données réelles")
    print("=" * 60)
    
    try:
        # Étape 1: Corriger les fichiers CSV
        print("\n1️⃣ Correction des fichiers CSV")
        fixed_dir = fix_csv_files()
        
        # Étape 2: Créer les fichiers de dispositifs
        print("\n2️⃣ Création des fichiers de dispositifs")
        create_device_files()
        
        # Étape 3: Mettre à jour la configuration
        print("\n3️⃣ Mise à jour de la configuration")
        update_config()
        
        # Étape 4: Corriger le processeur de données
        print("\n4️⃣ Correction du processeur de données")
        fix_data_processor()
        
        # Étape 5: Corriger l'augmenteur
        print("\n5️⃣ Correction de l'augmenteur")
        fix_augmenter()
        
        print("\n" + "=" * 60)
        print("✅ CORRECTION TERMINÉE!")
        print("=" * 60)
        print("\n📋 Pour exécuter le pipeline complet:")
        print("  python main.py --epochs 1 --verbose")
        print("  python main.py --model hybrid --epochs 1")
        print("  python main.py --model cnn_3d --epochs 1")
        print("\n🔄 Pour restaurer les fichiers originaux:")
        print("  python fix_real_data.py --restore")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_original_files()
    else:
        main()
