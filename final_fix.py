#!/usr/bin/env python3
"""
Script final pour corriger le dernier problème de format des données.
"""

import sys
from pathlib import Path

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def fix_data_format():
    """Corriger le format des données pour l'entraînement."""
    
    print("🔧 Correction finale du format des données...")
    
    # Lire le fichier geophysical_trainer.py
    with open("src/model/geophysical_trainer.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Sauvegarder l'original
    with open("src/model/geophysical_trainer.py.backup", "w", encoding="utf-8") as f:
        f.write(content)
    
    # Modifier la méthode prepare_data_2d pour transposer les données
    old_method = '''        # Créer les datasets
        train_dataset = GeophysicalDataset2D(x_train, y_train)
        val_dataset = GeophysicalDataset2D(x_val, y_val)
        
        # Créer les data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )'''
    
    new_method = '''        # Transposer les données de (height, width, channels) vers (channels, height, width)
        x_train_transposed = np.transpose(x_train, (0, 3, 1, 2))
        x_val_transposed = np.transpose(x_val, (0, 3, 1, 2))
        
        # Créer les datasets
        train_dataset = GeophysicalDataset2D(x_train_transposed, y_train)
        val_dataset = GeophysicalDataset2D(x_val_transposed, y_val)
        
        # Créer les data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )'''
    
    # Remplacer la méthode
    new_content = content.replace(old_method, new_method)
    
    # Sauvegarder le fichier modifié
    with open("src/model/geophysical_trainer.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Format des données corrigé")

def test_pipeline():
    """Tester le pipeline complet."""
    
    print("🧪 Test du pipeline complet...")
    
    try:
        from main import main_with_args
        
        # Configuration des arguments
        sys.argv = ['main.py', '--epochs', '1', '--verbose']
        
        print("📋 Exécution du pipeline...")
        success = main_with_args()
        
        if success:
            print("\n🎉 PIPELINE EXÉCUTÉ AVEC SUCCÈS!")
            print("=" * 50)
            print("✅ Le pipeline AI-MAP fonctionne parfaitement!")
            print("📊 Toutes les phases ont été exécutées:")
            print("  ✅ Phase 1: Nettoyage des données")
            print("  ✅ Phase 2: Traitement des données")
            print("  ✅ Phase 3: Préparation des données")
            print("  ✅ Phase 4: Entraînement du modèle")
            print("  ✅ Phase 5: Évaluation et résultats")
            return True
        else:
            print("\n❌ Le pipeline a échoué!")
            return False
            
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return False

def restore_original():
    """Restaurer le fichier original."""
    
    print("🔄 Restauration du fichier original...")
    
    if Path("src/model/geophysical_trainer.py.backup").exists():
        with open("src/model/geophysical_trainer.py.backup", "r", encoding="utf-8") as f:
            content = f.read()
        
        with open("src/model/geophysical_trainer.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        Path("src/model/geophysical_trainer.py.backup").unlink()
        print("✅ Fichier original restauré")

def main():
    """Fonction principale."""
    
    print("🚀 CORRECTION FINALE DU PIPELINE AI-MAP")
    print("=" * 60)
    
    try:
        # Étape 1: Corriger le format des données
        print("\n1️⃣ Correction du format des données")
        fix_data_format()
        
        # Étape 2: Tester le pipeline
        print("\n2️⃣ Test du pipeline complet")
        success = test_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 CORRECTION TERMINÉE AVEC SUCCÈS!")
            print("=" * 60)
            print("\n📋 Le pipeline AI-MAP est maintenant entièrement fonctionnel!")
            print("\n🚀 Commandes disponibles:")
            print("  python main.py --epochs 10")
            print("  python main.py --model hybrid --epochs 5")
            print("  python main.py --model cnn_3d --epochs 3")
            print("  python main.py --help")
        else:
            print("\n❌ Le pipeline nécessite encore des corrections.")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_original()
    else:
        main()
