# api/routes/falsos_positivos.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from api.utils.preprocessing import preprocess_input, validate_input
from api.services import falsos_positivos_service

router = APIRouter(prefix="/falsos-positivos", tags=["Falsos Positivos"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]


@router.post("/predict")
async def predict_falsos_positivos(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo de detección de falsos positivos.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo de detección de falsos positivos
    """
    try:
        # Validar entrada
        is_valid, error_msg = validate_input(request.data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preprocesar datos
        processed_data = preprocess_input(request.data)
        
        # Realizar predicción
        prediction = falsos_positivos_service.predict(processed_data)
        
        return {
            "status": "success",
            "result": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Verifica que el servicio de detección de falsos positivos esté funcionando."""
    try:
        falsos_positivos_service.load_model()
        return {"status": "healthy", "service": "falsos_positivos"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
