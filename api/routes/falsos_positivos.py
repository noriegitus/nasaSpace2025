# api/routes/falsos_positivos.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from api.utils.preprocessing import preprocess_input, validate_input
from api.utils.feature_groups import get_base_features, get_feature_group
from api.services import falsos_positivos_service

router = APIRouter(prefix="/falsos-positivos", tags=["Falsos Positivos"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "koi_fpflag_nt": 0,  # No Transit Seen
                    "koi_fpflag_ss": 0,  # Stellar Eclipse
                    "koi_fpflag_co": 0,  # Centroid Offset
                    "koi_fpflag_ec": 0   # Ephemeris Match Indicates Contamination
                }
            }
        }


@router.post("/predict")
async def predict_falsos_positivos(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el modelo de detección de falsos positivos.
    
    Este endpoint evalúa las banderas de falsos positivos de Kepler:
    - NT: No Transit Seen
    - SS: Stellar Eclipse
    - CO: Centroid Offset
    - EC: Ephemeris Match Indicates Contamination
    
    Args:
        request: Objeto con los datos del exoplaneta candidato
        
    Returns:
        Predicción del modelo de falsos positivos incluyendo:
        - score: Probabilidad de ser un falso positivo
        - predicción: FALSE POSITIVE o CONFIRMED
        - confianza: Nivel de confianza en la predicción (0-1)
        
    Raises:
        HTTPException (400): Si faltan características requeridas
        HTTPException (500): Si hay errores en el procesamiento o predicción
    """
    try:
        # Validar datos de entrada usando feature_groups
        is_valid, error_msg = validate_input(request.data, 'falsos_positivos')
        if not is_valid:
            logging.error(f"Error de validación: {error_msg}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": error_msg,
                    "required_features": {
                        "base_features": get_base_features('falsos_positivos'),
                        "all_features": get_feature_group('falsos_positivos')
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
            prediction = falsos_positivos_service.predict(processed_data)
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
    Retorna las características requeridas por el modelo de falsos positivos.
    
    Returns:
        Lista de características base y derivadas necesarias
    """
    return {
        "base_features": get_base_features('falsos_positivos'),
        "derived_features": [f for f in get_feature_group('falsos_positivos') 
                           if f not in get_base_features('falsos_positivos')],
        "example_input": PredictionRequest.Config.schema_extra["example"]["data"],
        "feature_descriptions": {
            "koi_fpflag_nt": "Flag indicando si no se observa tránsito",
            "koi_fpflag_ss": "Flag indicando si hay un eclipse estelar",
            "koi_fpflag_co": "Flag indicando si hay desplazamiento del centroide",
            "koi_fpflag_ec": "Flag indicando si hay contaminación por concordancia de efemérides"
        }
    }


@router.get("/health")
async def health_check():
    """
    Verifica que el servicio de falsos positivos esté funcionando.
    
    Comprueba:
    1. Carga del modelo
    2. Acceso a características
    3. Preprocesamiento
    """
    try:
        # Verificar carga del modelo
        falsos_positivos_service.load_model()
        
        # Verificar acceso a características
        features = get_feature_group('falsos_positivos')
        if not features:
            raise ValueError("No se pudieron obtener las características del modelo")
            
        return {
            "status": "healthy",
            "service": "falsos_positivos",
            "features_count": len(features)
        }
        
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Servicio no disponible: {str(e)}"
        )
