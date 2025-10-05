# api/main.py

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import fotometria, orbital, estelar, falsos_positivos, ensemble, judge

# Crear aplicación FastAPI
app = FastAPI(
    title="NASA Space Apps 2025 - Exoplanet Detection API",
    description="API para detección de exoplanetas usando modelos especialistas de deep learning + Juez Final",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(fotometria.router)
app.include_router(orbital.router)
app.include_router(estelar.router)
app.include_router(falsos_positivos.router)
app.include_router(ensemble.router)
app.include_router(judge.router)  # ← NUEVO: Juez Final


@app.get("/")
async def root():
    """Endpoint raíz del API."""
    return {
        "message": "NASA Space Apps 2025 - Exoplanet Detection API",
        "version": "2.0.0",
        "models": {
            "specialists": {
                "fotometria": "/fotometria/predict",
                "orbital": "/orbital/predict",
                "estelar": "/estelar/predict",
                "falsos_positivos": "/falsos-positivos/predict"
            },
            "aggregators": {
                "ensemble": "/ensemble/predict",
                "judge": "/judge/predict"  # ← NUEVO
            }
        },
        "docs": "/docs",
        "description": "Sistema jerárquico: 4 especialistas → Ensemble (promedio) + Judge (regresión logística)"
    }


@app.get("/health")
async def health():
    """Endpoint de salud general del API."""
    return {
        "status": "healthy",
        "api": "exoplanet-detection",
        "version": "2.0.0",
        "models": {
            "specialists": 4,
            "aggregators": 2
        }
    }


if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
