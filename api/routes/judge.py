# api/routes/judge.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from api.utils.preprocessing import preprocess_input, validate_input
from api.utils.feature_groups import get_base_features, get_feature_group
from api.services import judge_service

router = APIRouter(prefix="/judge", tags=["Judge"])


class PredictionRequest(BaseModel):
    """Modelo de datos para la solicitud de predicción."""
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    # Características de Fotometría
                    "koi_duration": 2.5,
                    "koi_duration_err1": 0.1,
                    "koi_duration_err2": 0.1,
                    "koi_depth": 100,
                    "koi_depth_err1": 5,
                    "koi_depth_err2": 5,
                    "koi_impact": 0.5,
                    "koi_model_snr": 50,
                    
                    # Características Orbitales
                    "koi_period": 365.25,
                    "koi_period_err1": 0.5,
                    "koi_period_err2": 0.5,
                    "koi_time0bk": 120.5,
                    "koi_time0bk_err1": 0.1,
                    "koi_time0bk_err2": 0.1,
                    
                    # Características Estelares
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
                    "koi_kepmag": 12.0,
                    
                    # Flags de Falsos Positivos
                    "koi_fpflag_nt": 0,
                    "koi_fpflag_ss": 0,
                    "koi_fpflag_co": 0,
                    "koi_fpflag_ec": 0
                }
            }
        }


@router.post("/predict")
async def predict_judge(request: PredictionRequest):
    """
    Endpoint para realizar predicciones usando el Juez Final (Regresión Logística).
    
    El juez es el modelo de más alto nivel que:
    1. Recopila predicciones de los 4 modelos especialistas
    2. Analiza y pondera sus predicciones
    3. Toma la decisión final sobre el candidato
    
    Especialistas:
    - Fotometría: Analiza la señal del tránsito
    - Orbital: Evalúa las características orbitales
    - Estelar: Valida propiedades de la estrella anfitriona
    - Falsos Positivos: Busca señales de falsos positivos conocidos
    
    Args:
        request: Objeto con los datos del candidato (requiere todas las características)
        
    Returns:
        Predicción completa incluyendo:
        - Decisión final del juez (score, predicción, confianza)
        - Predicciones y scores de cada especialista
        - Análisis de consenso entre especialistas
        
    Raises:
        HTTPException (400): Si faltan características o hay errores de validación
        HTTPException (500): Si hay errores en el procesamiento o predicción
    """
    try:
        # 1. Validar características para todos los modelos
        required_features = {
            model: get_base_features(model)
            for model in ['fotometria', 'orbital', 'estelar', 'falsos_positivos']
        }
        
        # Validar características para cada modelo especialista
        for model_type, features in required_features.items():
            is_valid, error_msg = validate_input(request.data, model_type)
            if not is_valid:
                logging.error(f"Error de validación en {model_type}: {error_msg}")
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": f"Error en características de {model_type}",
                        "message": error_msg,
                        "required_features": features
                    }
                )

        # 2. Preprocesar datos para todos los modelos
        try:
            processed_data = preprocess_input(request.data)
            logging.info("Datos preprocesados exitosamente para todos los modelos")
        except Exception as e:
            logging.error(f"Error en preprocesamiento: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error al preprocesar datos: {str(e)}"
            )

        # 3. Obtener predicción del juez (incluye predicciones de especialistas)
        try:
            prediction = judge_service.predict(processed_data)
            logging.info("Predicción del juez completada exitosamente")
            
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
    Retorna las características requeridas por cada modelo.
    
    Returns:
        Estructura completa de todas las características necesarias
        organizadas por especialista.
    """
    specialists = ['fotometria', 'orbital', 'estelar', 'falsos_positivos']
    
    features = {
        model: {
            "base_features": get_base_features(model),
            "derived_features": [
                f for f in get_feature_group(model) 
                if f not in get_base_features(model)
            ]
        }
        for model in specialists
    }
    
    return {
        "features_by_model": features,
        "example_input": PredictionRequest.Config.schema_extra["example"]["data"],
        "description": {
            "fotometria": "Analiza la señal del tránsito (duración, profundidad, etc.)",
            "orbital": "Evalúa las características orbitales del candidato",
            "estelar": "Valida las propiedades de la estrella anfitriona",
            "falsos_positivos": "Detecta señales conocidas de falsos positivos",
            "judge": "Integra y pondera los resultados de los especialistas"
        }
    }


@router.get("/health")
async def health_check():
    """
    Verifica que todos los servicios necesarios estén funcionando.
    
    Comprueba:
    1. Carga de todos los modelos (especialistas + juez)
    2. Acceso a características
    3. Estado del preprocesamiento
    """
    try:
        # Verificar carga del modelo juez
        judge_service.load_model()
        
        # Verificar acceso a características de todos los modelos
        specialists = ['fotometria', 'orbital', 'estelar', 'falsos_positivos']
        features_count = {
            model: len(get_feature_group(model))
            for model in specialists
        }
        
        if not all(features_count.values()):
            raise ValueError("No se pudieron obtener las características de todos los modelos")
            
        return {
            "status": "healthy",
            "service": "judge",
            "description": "Juez Final - Regresión Logística basada en scores de especialistas",
            "features_by_model": features_count
        }
        
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Servicio no disponible: {str(e)}"
        )
