# 🗺️ AI-MAP - Générateur d'Images Géophysiques

## 📋 Description

AI-MAP est un projet de génération automatique d'images géophysiques utilisant des réseaux de neurones convolutifs (U-Net 2D et VoxNet 3D). Le projet transforme des données CSV géophysiques en pseudo-sections 2D et modèles 3D.

## 🏗️ Architecture

### **Modèles Conformes au Cahier des Charges**
- **U-Net 2D** : ~31M paramètres, génération de pseudo-sections 2D
- **VoxNet 3D** : ~15M paramètres, génération de modèles 3D
- **Générateur Intégré** : Pipeline complet U-Net 2D + VoxNet 3D

### **Structure du Projet**
```
ai-map/
├── 📁 scripts/           # Scripts d'exécution
├── 📁 backend/           # Code backend Python
├── 📁 ai-map-frontend/   # Application React/TypeScript
├── 📁 test/              # Tests unitaires et d'intégration
├── 📁 docs/              # Documentation
├── 📁 config/            # Fichiers de configuration
├── 📁 data/              # Données géophysiques
├── 📁 artifacts/         # Modèles entraînés
└── 📁 output/            # Résultats de génération
```

## 🚀 Démarrage Rapide

### **1. Installation**
```bash
# Cloner le projet
git clone <repository-url>
cd ai-map

# Installer les dépendances
pip install -r config/requirements.txt
```

### **2. Démarrage du Projet**
```bash
# Script de démarrage interactif
python scripts/start_project.py

# Ou directement le serveur API
python scripts/run_api_server.py
```

### **3. Interface Web**
- Ouvrir http://localhost:8000
- Uploader des données CSV géophysiques
- Sélectionner la méthode (Pole-Dipole, Schlumberger)
- Générer les visualisations

## 📚 Documentation

- [Guide d'Installation](docs/README_INSTALLATION.md)
- [Guide de Génération d'Images](docs/GUIDE_GENERATION_IMAGES.md)
- [Migration des Générateurs](docs/GUIDE_MIGRATION_GENERATEURS.md)
- [Tests](docs/README_TESTS.md)
- [Nettoyage des Données](docs/README_DATA_CLEANING.md)

## 🧪 Tests

```bash
# Tests unitaires
python test/run_tests.py

# Tests d'intégration
python scripts/test_api_integration.py

# Tests des générateurs
python scripts/test_generators.py
```

## 🤖 Entraînement des Modèles

```bash
# Entraînement U-Net 2D
python scripts/train_generators.py --model unet_2d

# Entraînement VoxNet 3D
python scripts/train_generators.py --model voxnet_3d

# Entraînement intégré
python scripts/train_generators.py --model integrated
```

## 📊 Données Supportées

- **Pole-Dipole** : Données de résistivité et chargeabilité
- **Schlumberger** : Données de résistivité et chargeabilité
- **Format** : CSV avec colonnes x, y, resistivity, chargeability

## 🔧 Configuration

- **Fichiers de config** : `config/`
- **Requirements** : `config/requirements.txt`
- **Configuration** : `config/config.py`

## 📈 Résultats

Le projet génère :
- **Pseudo-sections 2D** : Visualisations de résistivité/chargeabilité
- **Modèles 3D** : Volumes de chargeabilité
- **Cartes d'iso-résistivité** : Cartes de contour
- **Métadonnées** : Informations sur la génération

## 🎯 Conformité

✅ **100% Conforme au Cahier des Charges**
- Architecture U-Net 2D + VoxNet 3D
- Paramètres conformes (~46M total)
- Génération d'images géophysiques
- Interface web moderne
- Tests complets

## 📞 Support

Pour toute question ou problème, consultez la documentation dans le dossier `docs/`.

---

**AI-MAP** - Génération d'Images Géophysiques par IA 🚀