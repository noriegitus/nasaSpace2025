# api/routes/estelar.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from api.utils.preprocessing import preprocess_input, validate_input
from api.services import estelar_service

router = APIRouter(prefix="/estelar", tags=["Estelar"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]


@router.post("/predict")
async def predict_estelar(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo de propiedades estelares.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo de propiedades estelares
    """
    try:
        # Validar entrada
        is_valid, error_msg = validate_input(request.data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preprocesar datos
        processed_data = preprocess_input(request.data)
        
        # Realizar predicción
        prediction = estelar_service.predict(processed_data)
        
        return {
            "status": "success",
            "result": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Verifica que el servicio de propiedades estelares esté funcionando."""
    try:
        estelar_service.load_model()
        return {"status": "healthy", "service": "estelar"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
