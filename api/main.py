# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import fotometria, orbital, estelar, falsos_positivos, ensemble

# Crear aplicación FastAPI
app = FastAPI(
    title="NASA Space Apps 2025 - Exoplanet Detection API",
    description="API para detección de exoplanetas usando modelos especialistas de deep learning",
    version="1.0.0",
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


@app.get("/")
async def root():
    """Endpoint raíz del API."""
    return {
        "message": "NASA Space Apps 2025 - Exoplanet Detection API",
        "version": "1.0.0",
        "endpoints": {
            "fotometria": "/fotometria/predict",
            "orbital": "/orbital/predict",
            "estelar": "/estelar/predict",
            "falsos_positivos": "/falsos-positivos/predict",
            "ensemble": "/ensemble/predict"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Endpoint de salud general del API."""
    return {
        "status": "healthy",
        "api": "exoplanet-detection",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
