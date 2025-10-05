# api/routes/fotometria.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from api.utils.preprocessing import preprocess_input, validate_input
from api.utils.feature_groups import get_base_features, get_feature_group
from api.services import fotometria_service

router = APIRouter(prefix="/fotometria", tags=["Fotometría"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "koi_duration": 2.5,
                    "koi_duration_err1": 0.1,
                    "koi_duration_err2": 0.1,
                    "koi_depth": 100,
                    "koi_depth_err1": 5,
                    "koi_depth_err2": 5,
                    "koi_impact": 0.5,
                    "koi_model_snr": 50
                }
            }
        }


@router.post("/predict")
async def predict_fotometria(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo de fotometría.
    
    Este endpoint evalúa la señal fotométrica del tránsito planetario.
    Considera características como la duración, profundidad, impacto y SNR del modelo.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo de fotometría incluyendo:
        - score: Probabilidad de ser un tránsito planetario real
        - predicción: CONFIRMED o FALSE POSITIVE
        - confianza: Nivel de confianza en la predicción (0-1)
        
    Raises:
        HTTPException (400): Si faltan características requeridas
        HTTPException (500): Si hay errores en el procesamiento o predicción
    """
    try:
        # Validar datos de entrada usando feature_groups
        is_valid, error_msg = validate_input(request.data, 'fotometria')
        if not is_valid:
            logging.error(f"Error de validación: {error_msg}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": error_msg,
                    "required_features": {
                        "base_features": get_base_features('fotometria'),
                        "all_features": get_feature_group('fotometria')
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
            prediction = fotometria_service.predict(processed_data)
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
    Retorna las características requeridas por el modelo de fotometría.
    
    Returns:
        Lista de características base y derivadas necesarias
    """
    return {
        "base_features": get_base_features('fotometria'),
        "derived_features": [f for f in get_feature_group('fotometria') 
                           if f not in get_base_features('fotometria')],
        "example_input": PredictionRequest.Config.schema_extra["example"]["data"]
    }


@router.get("/health")
async def health_check():
    """
    Verifica que el servicio de fotometría esté funcionando.
    
    Comprueba:
    1. Carga del modelo
    2. Acceso a características
    3. Preprocesamiento
    """
    try:
        # Verificar carga del modelo
        fotometria_service.load_model()
        
        # Verificar acceso a características
        features = get_feature_group('fotometria')
        if not features:
            raise ValueError("No se pudieron obtener las características del modelo")
            
        return {
            "status": "healthy",
            "service": "fotometria",
            "features_count": len(features)
        }
        
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Servicio no disponible: {str(e)}"
        )
