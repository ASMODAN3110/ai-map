#!/usr/bin/env python3
"""
Serveur API FastAPI pour la génération d'images géophysiques.
Intègre le générateur d'images avec le frontend React.
"""

import sys
import os
import io
import base64
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

from backend.model.geophysical_generators import GeophysicalImageGenerator
from backend.utils.logger import logger

# Initialiser l'application FastAPI
app = FastAPI(
    title="AI-MAP Image Generator API",
    description="API pour la génération d'images géophysiques à partir de données CSV",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le générateur d'images
image_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialiser le générateur d'images au démarrage."""
    global image_generator
    try:
        image_generator = GeophysicalImageGenerator()
        logger.info("🚀 Générateur d'images initialisé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation du générateur: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint de base."""
    return {
        "message": "AI-MAP Image Generator API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "healthy",
        "generator_ready": image_generator is not None
    }

@app.post("/api/generate-images")
async def generate_images(
    file: UploadFile = File(...),
    method: str = Form("pole-dipole"),
    samples: int = Form(3)
):
    """
    Générer des images géophysiques à partir d'un fichier CSV.
    
    Args:
        file: Fichier CSV uploadé
        method: Méthode géophysique (pole-dipole, schlumberger)
        samples: Nombre d'échantillons à traiter
        
    Returns:
        Résultats de génération avec images en base64
    """
    try:
        logger.info(f"📊 Génération d'images pour {file.filename}")
        logger.info(f"   - Méthode: {method}")
        logger.info(f"   - Échantillons: {samples}")
        
        # Vérifier que le générateur est initialisé
        if image_generator is None:
            raise HTTPException(status_code=500, detail="Générateur d'images non initialisé")
        
        # Lire le fichier CSV
        contents = await file.read()
        
        # Créer un DataFrame à partir du contenu
        csv_buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_buffer)
        
        # Sélectionner les colonnes pertinentes
        relevant_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['resist', 'charge', 'x', 'y', 'depth', 'prof']):
                relevant_columns.append(col)
        
        if len(relevant_columns) < 4:
            # Utiliser les premières colonnes numériques
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            relevant_columns = numeric_columns[:4].tolist()
        
        logger.info(f"Colonnes sélectionnées: {relevant_columns}")
        
        # Extraire les données
        csv_data = df[relevant_columns].values.astype(np.float32)
        
        # Nettoyer les données (supprimer les NaN)
        csv_data = csv_data[~np.isnan(csv_data).any(axis=1)]
        
        # Limiter le nombre d'échantillons
        if len(csv_data) > samples:
            csv_data = csv_data[:samples]
        
        logger.info(f"✅ Données CSV traitées: {csv_data.shape}")
        
        # Générer les pseudo-sections 2D
        pseudo_sections = image_generator.generate_pseudo_sections(csv_data, method)
        
        # Générer les modèles 3D
        models_3d = image_generator.generate_3d_models(csv_data, method)
        
        # Préparer la réponse
        response_data = {
            "success": True,
            "method": method,
            "num_samples": len(csv_data),
            "visualizations": {
                "pseudo_sections": pseudo_sections,
                "chargeability_maps": pseudo_sections,  # Utiliser les pseudo-sections pour les cartes de chargeabilité
                "model_3d": models_3d[0] if models_3d else ""
            },
            "metadata": {
                "columns_used": relevant_columns,
                "data_shape": csv_data.shape,
                "processing_time": "N/A"  # Pourrait être calculé
            }
        }
        
        logger.info(f"✅ Génération terminée: {len(pseudo_sections)} images générées")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.post("/api/generate-sample-images")
async def generate_sample_images(
    method: str = Form("pole-dipole"),
    samples: int = Form(3)
):
    """
    Générer des images d'exemple avec des données factices.
    
    Args:
        method: Méthode géophysique
        samples: Nombre d'échantillons
        
    Returns:
        Résultats de génération avec images en base64
    """
    try:
        logger.info(f"🧪 Génération d'images d'exemple")
        logger.info(f"   - Méthode: {method}")
        logger.info(f"   - Échantillons: {samples}")
        
        # Vérifier que le générateur est initialisé
        if image_generator is None:
            raise HTTPException(status_code=500, detail="Générateur d'images non initialisé")
        
        # Créer des données d'exemple
        from backend.model.geophysical_generators import create_sample_csv_data
        csv_data = create_sample_csv_data(n_samples=samples)
        
        # Générer les pseudo-sections 2D
        pseudo_sections = image_generator.generate_pseudo_sections(csv_data, method)
        
        # Générer les modèles 3D
        models_3d = image_generator.generate_3d_models(csv_data, method)
        
        # Préparer la réponse
        response_data = {
            "success": True,
            "method": method,
            "num_samples": len(csv_data),
            "visualizations": {
                "pseudo_sections": pseudo_sections,
                "chargeability_maps": pseudo_sections,  # Utiliser les pseudo-sections pour les cartes de chargeabilité
                "model_3d": models_3d[0] if models_3d else ""
            },
            "metadata": {
                "data_type": "sample",
                "data_shape": csv_data.shape,
                "processing_time": "N/A"
            }
        }
        
        logger.info(f"✅ Génération d'exemple terminée: {len(pseudo_sections)} images générées")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération d'exemple: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.get("/api/methods")
async def get_available_methods():
    """Obtenir la liste des méthodes géophysiques disponibles."""
    return {
        "methods": [
            {
                "id": "pole-dipole",
                "name": "Pôle-Dipôle",
                "description": "Méthode de résistivité électrique pour l'exploration"
            },
            {
                "id": "schlumberger",
                "name": "Schlumberger",
                "description": "Méthode de résistivité électrique pour l'exploration"
            }
        ]
    }

@app.get("/api/models")
async def get_available_models():
    """Obtenir la liste des modèles disponibles."""
    return {
        "models": [
            {
                "id": "unet-2d",
                "name": "U-Net 2D",
                "description": "Générateur de pseudo-sections 2D avec ~31M paramètres",
                "architecture": "U-Net avec connexions résiduelles",
                "input": "Grilles 2D (64×64×4 canaux)",
                "output": "Pseudo-sections (résistivité + chargeabilité)",
                "parameters": "~31M",
                "speed": "Rapide"
            },
            {
                "id": "voxnet-3d",
                "name": "VoxNet 3D",
                "description": "Générateur de modèles 3D avec ~15M paramètres",
                "architecture": "VoxNet avec déconvolutions 3D",
                "input": "Volumes 3D (32×32×32×4 canaux)",
                "output": "Modèles 3D de chargeabilité",
                "parameters": "~15M",
                "speed": "Moyen"
            },
            {
                "id": "integrated-generator",
                "name": "Générateur Intégré",
                "description": "Système complet U-Net 2D + VoxNet 3D",
                "architecture": "Pipeline de génération multi-échelle",
                "input": "Données CSV géophysiques",
                "output": "Pseudo-sections 2D + Modèles 3D",
                "parameters": "~46M total",
                "speed": "Optimisé"
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Démarrage du serveur API AI-MAP")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
