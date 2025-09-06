# Guide d'Entraînement AI-MAP

Ce guide explique comment utiliser le pipeline d'entraînement AI-MAP pour entraîner différents types de modèles géophysiques.

## 🚀 Démarrage Rapide

### Exécution Simple (Mode Par Défaut)
```bash
python main.py
```
Cette commande exécute le pipeline complet avec les paramètres par défaut :
- Modèle : CNN 2D
- Époques : 50
- Learning rate : 0.001
- Batch size : 32

### Exécution avec Arguments
```bash
python main.py --model cnn_2d --epochs 100 --learning-rate 0.0001
```

## 📋 Types de Modèles Disponibles

### 1. CNN 2D (`cnn_2d`) - ✅ Opérationnel
**Usage :** Classification de grilles géophysiques 2D
```bash
# Entraînement
python main.py --model cnn_2d --epochs 100 --batch-size 32

# Exécution du modèle sauvegardé
python run_cnn_2d_model.py
python run_cnn_2d_model.py --real-data
```

**Caractéristiques :**
- Input : Grilles 2D (4, 64, 64) - 4 canaux pour les dispositifs
- Architecture : CNN géophysique 2D
- Sortie : Classification binaire (2 classes)
- Paramètres : ~2M paramètres entraînables
- Fichier : `cnn_2d_model.pth`

### 2. CNN 3D (`cnn_3d`) - ✅ Opérationnel
**Usage :** Classification de volumes géophysiques 3D
```bash
# Entraînement
python main.py --model cnn_3d --epochs 80 --batch-size 16

# Exécution du modèle sauvegardé
python run_cnn_3d_model.py
python run_cnn_3d_model.py --real-data
```

**Caractéristiques :**
- Input : Volumes 3D (4, 32, 32, 32) - 4 canaux multi-dispositifs
- Architecture : CNN géophysique 3D
- Sortie : Classification binaire (2 classes)
- Paramètres : ~1.5M paramètres entraînables
- Fichier : `cnn_3d_model.pth`

### 3. Modèle Hybride (`hybrid`) - ✅ Opérationnel
**Usage :** Combinaison d'images et de données géophysiques
```bash
# Entraînement
python main.py --model hybrid --epochs 60 --learning-rate 0.0005

# Exécution du modèle sauvegardé
python run_hybrid_model.py
python run_hybrid_model.py --real-data --verbose
```

**Caractéristiques :**
- Input : Images (3, 64, 64) + Données géophysiques (4,)
- Architecture : ResNet18 + Encodeur géophysique + Fusion
- Sortie : Classification binaire (2 classes)
- Paramètres : ~12M paramètres entraînables
- Fichier : `hybrid_model.pth`


### 4. Modèle DataFrame (`dataframe`)
**Usage :** Classification de données tabulaires géophysiques
```bash
python main.py --model dataframe --epochs 120 --batch-size 64
```

**Caractéristiques :**
- Input : DataFrames avec features géophysiques
- Architecture : Réseau fully connected
- Optimisé pour : Données tabulaires structurées

## ⚙️ Paramètres d'Entraînement

### Paramètres Principaux

| Paramètre | Option | Défaut | Description |
|-----------|--------|--------|-------------|
| `--model` | `-m` | `cnn_2d` | Type de modèle à entraîner |
| `--epochs` | `-e` | `50` | Nombre d'époques d'entraînement |
| `--learning-rate` | `-lr` | `0.001` | Taux d'apprentissage |
| `--batch-size` | `-b` | `32` | Taille du batch |
| `--patience` | `-p` | `10` | Patience pour early stopping |
| `--device` | `-d` | `auto` | Device (auto/cpu/cuda) |

### Options de Pipeline

| Option | Description |
|--------|-------------|
| `--skip-cleaning` | Passer la phase de nettoyage des données |
| `--skip-processing` | Passer la phase de traitement des données |
| `--skip-training` | Passer la phase d'entraînement |
| `--output-dir` | `-o` | Répertoire de sortie personnalisé |
| `--verbose` | `-v` | Mode verbeux |

## 📝 Exemples d'Utilisation

### 1. Entraînement CNN 2D Standard
```bash
python main.py --model cnn_2d --epochs 100 --learning-rate 0.001 --batch-size 32
```

### 2. Entraînement CNN 3D avec GPU
```bash
python main.py --model cnn_3d --epochs 80 --device cuda --batch-size 16 --patience 15
```

### 3. Entraînement Modèle Hybride
```bash
python main.py --model hybrid --epochs 60 --learning-rate 0.0005 --verbose
```

### 4. Entraînement DataFrame avec Early Stopping
```bash
python main.py --model dataframe --epochs 200 --patience 20 --batch-size 64
```

### 5. Test du Pipeline (sans entraînement)
```bash
python main.py --skip-training --verbose
```

### 6. Entraînement avec Répertoire Personnalisé
```bash
python main.py --model cnn_2d --epochs 100 --output-dir ./my_models/
```

## 🔧 Configuration Avancée

### Variables d'Environnement
```bash
export CUDA_VISIBLE_DEVICES=0  # Utiliser GPU 0
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Ajouter le répertoire au PYTHONPATH
```

### Fichier de Configuration
Les paramètres par défaut sont définis dans `config.py`. Vous pouvez modifier :
- `CNNConfig.epochs` : Nombre d'époques par défaut
- `CNNConfig.learning_rate` : Taux d'apprentissage par défaut
- `CNNConfig.batch_size` : Taille de batch par défaut

## 📊 Monitoring et Résultats

### Fichiers de Sortie
- **Modèles :** `artifacts/models/[model_type]_model.pth`
- **Résumé :** `artifacts/training_summary.json`
- **Logs :** Affichage en temps réel dans le terminal

### Métriques Surveillées
- Loss d'entraînement et de validation
- Accuracy d'entraînement et de validation
- Learning rate (avec scheduler)
- Early stopping

### Exemple de Sortie
```
🚀 Starting AI-MAP Pipeline with CLI arguments
============================================================
Configuration:
  - Modèle: cnn_2d
  - Époques: 100
  - Learning rate: 0.001
  - Batch size: 32
  - Patience: 10
  - Device: cuda
  - Output: artifacts/models/
============================================================

📋 Phase 1: Nettoyage et prétraitement des données
--------------------------------------------------
✅ Nettoyage des données terminé avec succès

📊 Phase 2: Traitement des données et création des grilles
--------------------------------------------------
✅ Traitement des données terminé avec succès
Forme du tenseur multi-dispositifs: (100, 4, 64, 64)
Forme du volume 3D: (50, 4, 32, 32, 32)

🤖 Phase 4: Entraînement du modèle CNN_2D
--------------------------------------------------
Epoch 0/100
  Train Loss: 0.6931, Train Acc: 50.00%
  Val Loss: 0.6931, Val Acc: 50.00%
  LR: 0.001000
--------------------------------------------------
...
```

## 🐛 Dépannage

### Erreurs Communes

1. **CUDA out of memory**
   ```bash
   python main.py --model cnn_3d --batch-size 8  # Réduire batch size
   ```

2. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Données manquantes**
   ```bash
   python main.py --skip-cleaning --skip-processing  # Utiliser données factices
   ```

### Mode Debug
```bash
python main.py --verbose --skip-training  # Voir les détails sans entraîner
```

## 📈 Optimisation des Performances

### Pour GPU
- Utiliser `--device cuda`
- Ajuster `--batch-size` selon la mémoire GPU
- Modèles 3D : batch size plus petit (8-16)

### Pour CPU
- Utiliser `--device cpu`
- Batch size plus petit (16-32)
- Modèles 2D recommandés

### Early Stopping
- Ajuster `--patience` selon la complexité du modèle
- Modèles simples : patience 5-10
- Modèles complexes : patience 15-25

## 🔄 Workflow Recommandé

1. **Test initial :**
   ```bash
   python main.py --skip-training --verbose
   ```

2. **Entraînement rapide :**
   ```bash
   python main.py --model cnn_2d --epochs 10
   ```

3. **Entraînement complet :**
   ```bash
   python main.py --model cnn_2d --epochs 100 --patience 15
   ```

4. **Comparaison de modèles :**
   ```bash
   python main.py --model cnn_2d --epochs 50
   python main.py --model cnn_3d --epochs 50
   python main.py --model hybrid --epochs 50
   ```

## 📚 Ressources Supplémentaires

- `README.md` : Documentation générale du projet
- `config.py` : Configuration détaillée
- `src/model/` : Implémentation des modèles
- `examples/` : Exemples d'utilisation avancés
