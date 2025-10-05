# api/routes/orbital.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from api.utils.preprocessing import preprocess_input, validate_input
from api.utils.feature_groups import get_base_features, get_feature_group
from api.services import orbital_service

router = APIRouter(prefix="/orbital", tags=["Orbital"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "koi_period": 365.25,
                    "koi_period_err1": 0.5,
                    "koi_period_err2": 0.5,
                    "koi_time0bk": 120.5,
                    "koi_time0bk_err1": 0.1,
                    "koi_time0bk_err2": 0.1
                }
            }
        }


@router.post("/predict")
async def predict_orbital(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo orbital.
    
    Este endpoint evalúa las características orbitales del candidato.
    Considera el período orbital y el tiempo de tránsito central.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo orbital incluyendo:
        - score: Probabilidad de ser un tránsito planetario real
        - predicción: CONFIRMED o FALSE POSITIVE
        - confianza: Nivel de confianza en la predicción (0-1)
        
    Raises:
        HTTPException (400): Si faltan características requeridas
        HTTPException (500): Si hay errores en el procesamiento o predicción
    """
    try:
        # Validar datos de entrada usando feature_groups
        is_valid, error_msg = validate_input(request.data, 'orbital')
        if not is_valid:
            logging.error(f"Error de validación: {error_msg}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": error_msg,
                    "required_features": {
                        "base_features": get_base_features('orbital'),
                        "all_features": get_feature_group('orbital')
                    }
                }
            )

        # Preprocesar datos
        try:
            processed_data = preprocess_input(request.data)
            logging.info("Datos preprocesados exitosamente")
        except Exception as e:
            logging.error(f"Error en preprocesamiento: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error al preprocesar datos: {str(e)}"
            )

        # Realizar predicción
        try:
            prediction = orbital_service.predict(processed_data)
            logging.info(f"Predicción exitosa: {prediction}")
            
            return {
                "status": "success",
                "result": prediction
            }
            
        except ValueError as e:
            logging.error(f"Error en predicción: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
            
        except Exception as e:
            logging.error(f"Error interno en predicción: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error interno del servidor al realizar la predicción"
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error no manejado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )


@router.get("/features")
async def get_required_features():
    """
    Retorna las características requeridas por el modelo orbital.
    
    Returns:
        Lista de características base y derivadas necesarias
    """
    return {
        "base_features": get_base_features('orbital'),
        "derived_features": [f for f in get_feature_group('orbital') 
                           if f not in get_base_features('orbital')],
        "example_input": PredictionRequest.Config.schema_extra["example"]["data"]
    }


@router.get("/health")
async def health_check():
    """
    Verifica que el servicio orbital esté funcionando.
    
    Comprueba:
    1. Carga del modelo
    2. Acceso a características
    3. Preprocesamiento
    """
    try:
        # Verificar carga del modelo
        orbital_service.load_model()
        
        # Verificar acceso a características
        features = get_feature_group('orbital')
        if not features:
            raise ValueError("No se pudieron obtener las características del modelo")
            
        return {
            "status": "healthy",
            "service": "orbital",
            "features_count": len(features)
        }
        
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Servicio no disponible: {str(e)}"
        )
