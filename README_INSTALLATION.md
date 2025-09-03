# 🚀 Guide d'Installation Rapide - AI-Map

**Installation et configuration du projet AI-Map en 5 minutes !**

## ⚡ Installation Express

### **1. Cloner le Projet**
```bash
git clone <repository-url>
cd ai-map
```

### **2. Créer l'Environnement Virtuel**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### **3. Installer les Dépendances**
```bash
# Installation minimale (recommandée pour commencer)
pip install -r requirements-minimal.txt

# OU installation complète
pip install -r requirements.txt
```

### **4. Vérifier l'Installation**
```bash
python -c "import torch, numpy, pandas; print('✅ Installation réussie!')"
```

## 🔧 Installation Détaillée

### **Prérequis Système**
- **Python** : 3.9+ (recommandé 3.11+)
- **RAM** : 8GB minimum (16GB recommandé)
- **Espace disque** : 2GB minimum
- **OS** : Windows 10+, macOS 10.15+, Ubuntu 18.04+

### **Options d'Installation**

#### **🥇 Installation Minimale (Recommandée)**
```bash
pip install -r requirements-minimal.txt
```
**Inclut :** PyTorch, NumPy, Pandas, scikit-learn, matplotlib, pytest

#### **🥈 Installation Complète**
```bash
pip install -r requirements.txt
```
**Inclut :** Toutes les fonctionnalités + visualisation avancée + géospatial

#### **🥉 Installation Développement**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
**Inclut :** Outils de qualité de code + debugging + profiling

## 🧪 Validation de l'Installation

### **Test Rapide**
```bash
# Lancer les tests unitaires
python -m pytest test/unit/model/ -v

# Lancer les tests d'intégration
python -m pytest test/integration/ -v
```

### **Test Complet**
```bash
# Tous les tests avec couverture
python -m pytest --cov=src --cov-report=html test/
```

## 🐛 Résolution des Problèmes

### **Erreur : "Module not found"**
```bash
# Vérifier le PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou installer en mode développement
pip install -e .
```

### **Erreur : "CUDA not available"**
```bash
# Installer PyTorch CPU uniquement
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **Erreur : "Permission denied"**
```bash
# Utiliser --user flag
pip install --user -r requirements-minimal.txt

# Ou utiliser sudo (Linux/Mac)
sudo pip install -r requirements-minimal.txt
```

## 🚀 Premiers Pas

### **1. Exploration des Données**
```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir le notebook d'exploration
notebooks/phase1/01_data_exploration.ipynb
```

### **2. Test du Pipeline**
```bash
# Lancer le pipeline principal
python main.py

# Ou tester une fonctionnalité spécifique
python -c "
from src.model.geophysical_trainer import GeophysicalTrainer
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter

augmenter = GeophysicalDataAugmenter()
trainer = GeophysicalTrainer(augmenter, device='cpu')
print('✅ Pipeline initialisé avec succès!')
"
```

### **3. Visualisation des Tests**
```bash
# Ouvrir le rapport de couverture
# (après avoir lancé les tests avec --cov-report=html)
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

## 📊 Vérification de la Couverture

### **Statistiques Attendues**
- **Tests unitaires** : 115+ tests ✅
- **Tests d'intégration** : 5 tests ✅
- **Couverture totale** : 100% ✅
- **Temps d'exécution** : < 5 minutes ✅

### **Commandes de Vérification**
```bash
# Résumé rapide
python -m pytest --cov=src --cov-report=term-missing test/

# Rapport détaillé HTML
python -m pytest --cov=src --cov-report=html test/

# Rapport XML (pour CI/CD)
python -m pytest --cov=src --cov-report=xml test/
```

## 🔗 Liens Utiles

### **Documentation**
- **README principal** : `README.md`
- **Guide des tests** : `README_TESTS.md`
- **Guide d'entraînement** : `README_TRAINING.md`
- **Guide de nettoyage** : `README_DATA_CLEANING.md`
- **Guide d'augmentation** : `README_DATA_AUGMENTATION.md`

### **Structure du Projet**
```
ai-map/
├── 📁 src/                    # Code source
├── 📁 data/                   # Données
├── 📁 test/                   # Tests (100% couverture)
├── 📁 notebooks/              # Notebooks Jupyter
├── 📁 requirements/            # Dépendances
├── 📖 README.md               # Documentation principale
├── 🚀 main.py                 # Point d'entrée
└── ⚙️ config.py               # Configuration
```

## 🎯 Prochaines Étapes

### **Après l'Installation :**
1. **Explorer les données** : `data/raw/PD.csv`, `data/raw/S.csv`
2. **Lancer les tests** : Vérifier la couverture 100%
3. **Tester le pipeline** : Exécuter `main.py`
4. **Modifier le code** : Les tests vous protègent !
5. **Contribuer** : Ajouter de nouvelles fonctionnalités

### **Développement :**
1. **Tests unitaires** : Maintenir la couverture 100%
2. **Tests d'intégration** : Valider les workflows complets
3. **Documentation** : Mettre à jour les README
4. **Code quality** : Utiliser Black, Flake8, MyPy

---

## 🎉 Félicitations !

**Votre environnement AI-Map est maintenant prêt avec :**
- ✅ **Couverture de tests 100%**
- ✅ **Pipeline d'entraînement complet**
- ✅ **Processeur de données géophysiques**
- ✅ **Modèles CNN 2D/3D**
- ✅ **Augmentation de données spécialisée**

**🚀 Prêt à révolutionner l'analyse géophysique !**
