# api/routes/estelar.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from api.utils.preprocessing import preprocess_input, validate_input
from api.utils.feature_groups import get_base_features, get_feature_group
from api.services import estelar_service

router = APIRouter(prefix="/estelar", tags=["Estelar"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "koi_srad": 1.0,
                    "koi_srad_err1": 0.1,
                    "koi_srad_err2": 0.1,
                    "koi_steff": 5778,
                    "koi_steff_err1": 100,
                    "koi_steff_err2": 100,
                    "koi_slogg": 4.44,
                    "koi_slogg_err1": 0.1,
                    "koi_slogg_err2": 0.1,
                    "koi_prad": 1.0,
                    "koi_prad_err1": 0.1,
                    "koi_prad_err2": 0.1,
                    "koi_insol": 1.0,
                    "koi_insol_err1": 0.1,
                    "koi_insol_err2": 0.1,
                    "koi_teq": 288,
                    "koi_kepmag": 12.0
                }
            }
        }


@router.post("/predict")
async def predict_estelar(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo de propiedades estelares.
    
    Este endpoint evalúa las características de la estrella anfitriona.
    Considera radio, temperatura, gravedad superficial, radio planetario,
    insolación, temperatura de equilibrio y magnitud Kepler.
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo estelar incluyendo:
        - score: Probabilidad de ser un tránsito planetario real
        - predicción: CONFIRMED o FALSE POSITIVE
        - confianza: Nivel de confianza en la predicción (0-1)
        
    Raises:
        HTTPException (400): Si faltan características requeridas
        HTTPException (500): Si hay errores en el procesamiento o predicción
    """
    try:
        # Validar datos de entrada usando feature_groups
        is_valid, error_msg = validate_input(request.data, 'estelar')
        if not is_valid:
            logging.error(f"Error de validación: {error_msg}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": error_msg,
                    "required_features": {
                        "base_features": get_base_features('estelar'),
                        "all_features": get_feature_group('estelar')
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
            prediction = estelar_service.predict(processed_data)
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
    Retorna las características requeridas por el modelo estelar.
    
    Returns:
        Lista de características base y derivadas necesarias
    """
    return {
        "base_features": get_base_features('estelar'),
        "derived_features": [f for f in get_feature_group('estelar') 
                           if f not in get_base_features('estelar')],
        "example_input": PredictionRequest.Config.schema_extra["example"]["data"]
    }


@router.get("/health")
async def health_check():
    """
    Verifica que el servicio estelar esté funcionando.
    
    Comprueba:
    1. Carga del modelo
    2. Acceso a características
    3. Preprocesamiento
    """
    try:
        # Verificar carga del modelo
        estelar_service.load_model()
        
        # Verificar acceso a características
        features = get_feature_group('estelar')
        if not features:
            raise ValueError("No se pudieron obtener las características del modelo")
            
        return {
            "status": "healthy",
            "service": "estelar",
            "features_count": len(features)
        }
        
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Servicio no disponible: {str(e)}"
        )
