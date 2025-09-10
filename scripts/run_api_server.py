#!/usr/bin/env python3
"""
Script pour démarrer le serveur API AI-MAP.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importer et exécuter le serveur API
from api_server import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Démarrage du serveur API AI-MAP...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
