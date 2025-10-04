# api/routes/ensemble.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from api.utils.preprocessing import preprocess_input, validate_input
from api.services import ensemble_service

router = APIRouter(prefix="/ensemble", tags=["Ensemble"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]


@router.post("/predict")
async def predict_ensemble(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando todos los modelos especialistas.
    Combina las predicciones de los 4 modelos para dar una predicción final.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción combinada de todos los modelos
    """
    try:
        # Validar entrada
        is_valid, error_msg = validate_input(request.data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preprocesar datos
        processed_data = preprocess_input(request.data)
        
        # Realizar predicción con todos los modelos
        prediction = ensemble_service.predict_ensemble(processed_data)
        
        return {
            "status": "success",
            "result": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Verifica que todos los servicios del ensemble estén funcionando."""
    try:
        from api.services import fotometria_service, orbital_service, estelar_service, falsos_positivos_service
        
        fotometria_service.load_model()
        orbital_service.load_model()
        estelar_service.load_model()
        falsos_positivos_service.load_model()
        
        return {
            "status": "healthy",
            "service": "ensemble",
            "models": ["fotometria", "orbital", "estelar", "falsos_positivos"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
