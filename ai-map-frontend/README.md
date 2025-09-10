# AI-MAP Frontend

Interface web moderne pour l'analyse géophysique multi-dispositifs avec intelligence artificielle.

## 🚀 Technologies

- **React 18** - Bibliothèque UI moderne
- **TypeScript** - Typage statique
- **Vite** - Build tool rapide
- **Tailwind CSS** - Framework CSS utilitaire
- **Radix UI** - Composants primitifs accessibles
- **Lucide React** - Icônes vectorielles

## 🎨 Fonctionnalités

### Interface Géophysique
- **Upload de fichiers** - Support drag & drop pour .dat, .csv, .txt
- **Sélection de méthodes** - Pôle-Dipôle, Schlumberger
- **Modèles d'IA** - CNN 2D, CNN 3D, Modèle Hybride
- **Visualisations** - Pseudo-sections 2D, modèles 3D, cartes de résistivité

### Workflow d'Analyse
1. **Sélection de méthode** de mesure géophysique
2. **Upload de données** via drag & drop
3. **Choix du modèle** d'IA (CNN 2D/3D/Hybride)
4. **Analyse automatisée** avec suivi de progression
5. **Visualisation des résultats** en 2D/3D
6. **Export des données** et rapports

## 🛠️ Installation

```bash
# Installer les dépendances
npm install

# Démarrer le serveur de développement
npm run dev

# Build pour la production
npm run build

# Prévisualiser le build
npm run preview
```

## 📁 Structure du Projet

```
src/
├── components/          # Composants React
│   ├── ui/             # Composants UI de base
│   ├── FileUpload.tsx  # Upload de fichiers
│   ├── ModelSelector.tsx # Sélection de modèles
│   └── AnalysisResults.tsx # Affichage des résultats
├── lib/                # Utilitaires
├── types/              # Types TypeScript
├── App.tsx             # Composant principal
└── main.tsx            # Point d'entrée
```

## 🎯 Intégration Backend

L'interface est conçue pour s'intégrer avec le backend Python :

- **API REST** pour l'upload de fichiers
- **WebSocket** pour le suivi de progression
- **Endpoints** pour les modèles CNN 2D/3D/Hybride
- **Visualisations** avec les données réelles

## 🎨 Thème Géophysique

Palette de couleurs spécialisée :
- **Primary** : Bleu géophysique
- **Secondary** : Vert scientifique  
- **Accent** : Orange de données
- **Mode sombre** optimisé pour les environnements de travail

## 📱 Responsive Design

Interface adaptative pour :
- **Desktop** - Expérience complète
- **Tablet** - Interface optimisée
- **Mobile** - Version simplifiée

## 🔧 Configuration

### Variables d'environnement
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

### Personnalisation
- Modifier `tailwind.config.js` pour les couleurs
- Adapter `src/types/index.ts` pour les types de données
- Configurer `vite.config.ts` pour les alias de chemins

## 🚀 Déploiement

```bash
# Build de production
npm run build

# Les fichiers sont dans dist/
# Déployer sur votre serveur web
```

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.
