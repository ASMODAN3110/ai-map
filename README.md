# 🗺️ AI-MAP: Intelligence Artificielle pour l'Analyse Géophysique

## 📋 Description du Projet

**AI-MAP** est un système d'intelligence artificielle révolutionnaire pour l'analyse géophysique, capable de traiter automatiquement des données multi-dispositifs et de générer des modèles 2D/3D précis du sous-sol.

### 🎯 Objectifs Principaux

- **Innovation Technologique** : Système CNN multi-dispositifs pour la géophysique
- **Automatisation** : Réduction de 80% du temps de traitement manuel
- **Précision** : Amélioration de 25% de la précision d'inversion
- **Accessibilité** : Interface web intuitive pour utilisateurs non-experts

## 🏗️ Architecture du Projet

Ce projet s'inspire de l'architecture **EMUT** (Emotion Analysis) mais est adapté pour les données géophysiques.

### Structure des Dossiers

```
ai-map/
├── 📁 src/                    # Code source principal
│   ├── 📁 preprocessor/       # Nettoyage et validation des données
│   ├── 📁 data/              # Traitement et préparation des données
│   ├── 📁 model/             # Modèles CNN (U-Net 2D, VoxNet 3D)
│   └── 📁 utils/             # Utilitaires et logging
├── 📁 data/                   # Données du projet
│   ├── 📁 raw/               # Données brutes des dispositifs
│   ├── 📁 processed/         # Données nettoyées
│   └── 📁 intermediate/      # Données intermédiaires
├── 📁 notebooks/              # Notebooks Jupyter de développement
├── 📁 artifacts/              # Modèles et résultats sauvegardés
├── 📁 requirements/           # Dépendances Python
├── 📁 test/                  # Tests unitaires
├── ⚙️ config.py              # Configuration centralisée
├── 🚀 main.py                # Point d'entrée principal
└── 📖 README.md              # Ce fichier
```

## 🔬 Dispositifs Géophysiques Supportés

| Dispositif | Fichier | Mesures | Couverture | Caractéristiques |
|------------|---------|---------|------------|------------------|
| **Pôle-Pôle** | `profil 1.csv` | 164 | 950m × 450m | Exploration profonde |
| **Pôle-Dipôle** | `PD_Line1s.dat` | 144 | 1000m × modéré | Résolution latérale élevée |
| **Schlumberger 6** | `PRO 6 COMPLET.csv` | 469 | 945m × 94m | Résolution verticale élevée |
| **Schlumberger 7** | `PRO 7 COMPLET.csv` | ~100 | 180m × 31m | Profil court |

## 🧠 Modèles CNN

### U-Net 2D
- **Entrée** : Tenseur (64×64×4) - 4 canaux pour les dispositifs
- **Sortie** : 2 canaux (résistivité vraie, chargeabilité vraie)
- **Paramètres** : ~31M paramètres entraînables

### VoxNet 3D
- **Entrée** : Tenseur (32×32×32×4) - Volume 3D multi-canaux
- **Sortie** : Volume 3D de chargeabilité
- **Paramètres** : ~15M paramètres entraînables

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.9+
- pip ou conda
- Git

### Installation

```bash
# Cloner le projet
git clone <repository-url>
cd ai-map

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements/requirements_phase1.txt
```

### Utilisation

```bash
# Lancer le pipeline principal
python main.py

# Ou lancer depuis un notebook
jupyter notebook notebooks/phase1/01_data_exploration.ipynb
```

## 📊 Pipeline de Traitement

### Phase 1: Prétraitement (✅ Implémentée)
1. **Nettoyage des données** : Validation, suppression des valeurs manquantes
2. **Transformation des coordonnées** : LAT/LON → UTM
3. **Normalisation** : Résistivité (log), chargeabilité (min-max)
4. **Création des grilles spatiales** : 2D (64×64) et 3D (32×32×32)
5. **Augmentation des données** : Techniques géométriques, bruit, variations (✅ Nouveau!)

### Phase 2: Modèles CNN (🔄 En cours)
1. **Implémentation U-Net 2D**
2. **Implémentation VoxNet 3D**
3. **Pipeline d'entraînement**

### Phase 3: Application Web (📋 Planifiée)
1. **Backend Flask** : API REST
2. **Frontend React** : Interface utilisateur
3. **Base de données** : PostgreSQL + PostGIS

## 🛠️ Technologies Utilisées

- **Python** : Langage principal
- **TensorFlow/Keras** : Deep Learning
- **Pandas/NumPy** : Traitement des données
- **PyProj** : Transformations géospatiales
- **Scikit-learn** : Préprocessing et validation
- **Matplotlib/Seaborn** : Visualisation
- **Flask** : Backend web (futur)
- **React** : Frontend web (futur)

## 📈 Métriques de Performance

- **Temps de traitement** : < 5 minutes
- **Précision d'inversion** : > 90%
- **Couverture de tests** : > 90%
- **Disponibilité système** : > 99%

## 🧪 Tests

```bash
# Lancer tous les tests
pytest test/

# Lancer un test spécifique
pytest test/test_preprocessing.py

# Avec couverture
pytest --cov=src test/
```

## 📚 Documentation

- **Configuration** : `config.py` avec docstrings détaillés
- **Code source** : Docstrings et commentaires en français
- **Notebooks** : Exemples d'utilisation et tutoriels
- **Logs** : Système de logging coloré et configurable

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est développé dans le cadre d'un mémoire de fin d'études.
**Document confidentiel - Usage académique uniquement**

## 👥 Auteurs

- **DEMESSONG LEKEUFACK** - Chef de projet
- **ABDOULRAHIM MOMO ABOUBAKAR** - Développeur

## 📅 Dates

- **Début** : Juillet 2025
- **Phase 1** : ✅ Terminée
- **Phase 2** : 🔄 En cours
- **Phase 3** : 📋 Planifiée
- **Livraison finale** : Décembre 2025

## 🆘 Support

Pour toute question ou problème :
1. Consulter la documentation dans le code
2. Vérifier les logs d'erreur
3. Ouvrir une issue sur le repository
4. Contacter l'équipe de développement

---

**🎯 AI-MAP : Révolutionner l'analyse géophysique par l'intelligence artificielle**
