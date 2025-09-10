# Guide d'Intégration Frontend-Backend

Ce guide explique comment connecter l'interface React au backend Python d'AI-MAP.

## 🔗 Architecture d'Intégration

```
Frontend (React) ←→ API REST ←→ Backend (Python)
     ↓
WebSocket ←→ Suivi en temps réel
```

## 🚀 Étapes d'Intégration

### 1. Configuration de l'API Backend

Créez un serveur FastAPI dans le backend Python :

```python
# backend/api/main.py
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json

app = FastAPI(title="AI-MAP API", version="2.0.0")

# CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"success": True, "status": "healthy"}

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    # Traitement des fichiers uploadés
    file_ids = []
    for file in files:
        # Sauvegarder le fichier et générer un ID
        file_id = await save_uploaded_file(file)
        file_ids.append(file_id)
    
    return {"success": True, "data": {"fileIds": file_ids}}

@app.post("/api/analysis/start")
async def start_analysis(analysis_params: dict):
    # Démarrer l'analyse avec les modèles Python
    analysis_id = await start_geophysical_analysis(analysis_params)
    return {"success": True, "data": {"analysisId": analysis_id}}

@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    # Retourner le statut de l'analyse
    status = await get_analysis_progress(analysis_id)
    return {"success": True, "data": status}

@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    await websocket.accept()
    
    # Envoyer les mises à jour de progression
    while True:
        progress = await get_analysis_progress(analysis_id)
        await websocket.send_text(json.dumps({
            "type": "progress",
            "analysisId": analysis_id,
            "data": progress
        }))
        
        if progress["status"] == "completed":
            break
            
        await asyncio.sleep(1)
```

### 2. Intégration des Modèles Python

```python
# backend/api/analysis.py
import asyncio
from typing import Dict, Any
from backend.model.geophysical_trainer import GeophysicalTrainer
from backend.model.geophysical_hybrid_net import GeophysicalHybridNet

class AnalysisManager:
    def __init__(self):
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
    
    async def start_analysis(self, analysis_id: str, params: dict):
        """Démarrer une nouvelle analyse"""
        method = params.get("method")
        model_type = params.get("model")
        file_ids = params.get("fileIds", [])
        
        # Initialiser l'analyse
        self.active_analyses[analysis_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Initialisation...",
            "method": method,
            "model": model_type,
            "files": file_ids
        }
        
        # Lancer l'analyse en arrière-plan
        asyncio.create_task(self._run_analysis(analysis_id, params))
        
        return analysis_id
    
    async def _run_analysis(self, analysis_id: str, params: dict):
        """Exécuter l'analyse géophysique"""
        try:
            # Charger les données
            await self._update_progress(analysis_id, 20, "Chargement des données...")
            data = await self._load_data(params["fileIds"])
            
            # Traitement des données
            await self._update_progress(analysis_id, 40, "Traitement des données...")
            processed_data = await self._process_data(data, params["method"])
            
            # Exécution du modèle
            await self._update_progress(analysis_id, 60, "Analyse IA en cours...")
            results = await self._run_model(processed_data, params["model"])
            
            # Génération des visualisations
            await self._update_progress(analysis_id, 80, "Génération des visualisations...")
            visualizations = await self._generate_visualizations(results)
            
            # Finalisation
            await self._update_progress(analysis_id, 100, "Analyse terminée!")
            self.active_analyses[analysis_id]["status"] = "completed"
            self.active_analyses[analysis_id]["results"] = {
                "data": results,
                "visualizations": visualizations
            }
            
        except Exception as e:
            self.active_analyses[analysis_id]["status"] = "error"
            self.active_analyses[analysis_id]["error"] = str(e)
    
    async def _run_model(self, data, model_type: str):
        """Exécuter le modèle d'IA approprié"""
        if model_type == "cnn-2d":
            # Utiliser run_cnn_2d_model.py
            return await self._run_cnn_2d(data)
        elif model_type == "cnn-3d":
            # Utiliser run_cnn_3d_model.py
            return await self._run_cnn_3d(data)
        elif model_type == "hybrid":
            # Utiliser run_hybrid_model.py
            return await self._run_hybrid(data)
    
    async def _run_cnn_2d(self, data):
        """Exécuter le modèle CNN 2D"""
        # Intégrer avec run_cnn_2d_model.py
        import subprocess
        result = subprocess.run([
            "python", "run_cnn_2d_model.py", "--real-data"
        ], capture_output=True, text=True)
        return result.stdout
    
    async def _update_progress(self, analysis_id: str, progress: int, message: str):
        """Mettre à jour le progrès de l'analyse"""
        if analysis_id in self.active_analyses:
            self.active_analyses[analysis_id]["progress"] = progress
            self.active_analyses[analysis_id]["message"] = message

# Instance globale
analysis_manager = AnalysisManager()
```

### 3. Configuration du Frontend

Mettez à jour le fichier `.env` :

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

### 4. Intégration dans les Composants

```typescript
// src/hooks/useAnalysis.ts
import { useState, useCallback } from 'react'
import { apiClient, wsClient } from '@/lib/api'
import { AnalysisResult, AnalysisProgress } from '@/types'

export function useAnalysis() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState<AnalysisProgress | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const startAnalysis = useCallback(async (params: {
    method: string
    model: string
    files: File[]
  }) => {
    setIsAnalyzing(true)
    
    try {
      // 1. Upload des fichiers
      const uploadResponse = await apiClient.uploadFiles(params.files)
      if (!uploadResponse.success) {
        throw new Error(uploadResponse.error)
      }

      // 2. Démarrer l'analyse
      const analysisResponse = await apiClient.startAnalysis({
        method: params.method,
        model: params.model,
        fileIds: uploadResponse.data!.fileIds
      })
      
      if (!analysisResponse.success) {
        throw new Error(analysisResponse.error)
      }

      const analysisId = analysisResponse.data!.analysisId

      // 3. Suivre la progression via WebSocket
      wsClient.connect(analysisId)
      wsClient.onMessage('progress', (data) => {
        setProgress(data)
      })
      
      wsClient.onMessage('result', (data) => {
        setResult(data)
        setIsAnalyzing(false)
        wsClient.disconnect()
      })

    } catch (error) {
      console.error('Analysis failed:', error)
      setIsAnalyzing(false)
    }
  }, [])

  return {
    isAnalyzing,
    progress,
    result,
    startAnalysis
  }
}
```

## 🔧 Scripts d'Intégration

### Script de démarrage complet

```bash
#!/bin/bash
# start-full-stack.sh

# Démarrer le backend Python
cd backend
python -m uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Attendre que le backend soit prêt
sleep 5

# Démarrer le frontend React
cd ../ai-map-frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend démarré sur http://localhost:8000"
echo "Frontend démarré sur http://localhost:3000"
echo "Appuyez sur Ctrl+C pour arrêter"

# Attendre l'interruption
trap "kill $BACKEND_PID $FRONTEND_PID" INT
wait
```

## 📊 Flux de Données

1. **Upload** : Frontend → API → Sauvegarde fichiers
2. **Analyse** : API → Modèles Python → Traitement
3. **Progression** : WebSocket → Frontend (temps réel)
4. **Résultats** : API → Frontend → Visualisation

## 🧪 Tests d'Intégration

```python
# tests/test_integration.py
import pytest
import requests
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["success"] == True

def test_upload_files():
    files = {"files": ("test.csv", "content", "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    assert "fileIds" in response.json()["data"]
```

## 🚀 Déploiement

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
  
  frontend:
    build: ./ai-map-frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://backend:8000
    depends_on:
      - backend
```

## 📝 Notes Importantes

1. **CORS** : Configurez correctement les origines autorisées
2. **WebSocket** : Gestion des reconnexions automatiques
3. **Gestion d'erreurs** : Messages d'erreur clairs pour l'utilisateur
4. **Performance** : Optimisation des uploads de gros fichiers
5. **Sécurité** : Validation des fichiers uploadés

Cette intégration permet une expérience utilisateur fluide avec un backend Python robuste pour l'analyse géophysique.
