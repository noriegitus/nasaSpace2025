# api/services/judge_service.py

import joblib
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

# Agregar el directorio raíz al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import logging
from model.architecture.m_judge import JudgeModel
from api.services import fotometria_service, orbital_service, estelar_service, falsos_positivos_service

# Configuración
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights", "judge_model.joblib")

# Cargar modelo
model = None


def load_model():
    """Carga el modelo del juez (Regresión Logística)."""
    global model
    if model is None:
        sklearn_model = joblib.load(WEIGHTS_PATH)
        model = JudgeModel()
        model.load_state_dict(sklearn_model)
        model.eval()
    return model


def validate_input_data(data: pd.DataFrame) -> None:
    """
    Valida que los datos de entrada contengan todas las características necesarias.
    
    Args:
        data: DataFrame con los datos de entrada
        
    Raises:
        ValueError: Si faltan características requeridas para algún modelo
    """
    from api.utils.preprocessing import validate_features_for_model
    
    logging.info("Validando datos de entrada para todos los modelos especialistas")
    
    # Validar características para cada modelo especialista
    for model_type in ['fotometria', 'orbital', 'estelar', 'falsos_positivos']:
        is_valid, error_msg = validate_features_for_model(data, model_type)
        if not is_valid:
            error = f"Error en validación de {model_type}: {error_msg}"
            logging.error(error)
            raise ValueError(error)
    
    logging.info("Validación de características completada exitosamente")


def collect_specialist_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene las predicciones de todos los modelos especialistas.
    
    Args:
        data: DataFrame con los datos preprocesados
        
    Returns:
        DataFrame con los scores de los especialistas
        
    Raises:
        ValueError: Si hay errores en las predicciones de los especialistas
    """
    try:
        # Obtener predicciones de cada especialista
        logging.info("Obteniendo predicciones de los especialistas")
        pred_fotometria = fotometria_service.predict(data)
        pred_orbital = orbital_service.predict(data)
        pred_estelar = estelar_service.predict(data)
        pred_falsos_positivos = falsos_positivos_service.predict(data)
        
        # Crear DataFrame con los scores
        specialist_scores = pd.DataFrame({
            'score_fotometria': [pred_fotometria['score']],
            'score_orbital': [pred_orbital['score']],
            'score_estelar': [pred_estelar['score']],
            'score_falsos_positivos': [pred_falsos_positivos['score']]
        })
        
        logging.info("Predicciones de especialistas obtenidas exitosamente")
        return specialist_scores, {
            'scores': {
                'fotometria': pred_fotometria['score'],
                'orbital': pred_orbital['score'],
                'estelar': pred_estelar['score'],
                'falsos_positivos': pred_falsos_positivos['score']
            },
            'predictions': {
                'fotometria': pred_fotometria['prediccion'],
                'orbital': pred_orbital['prediccion'],
                'estelar': pred_estelar['prediccion'],
                'falsos_positivos': pred_falsos_positivos['prediccion']
            }
        }
        
    except Exception as e:
        error = f"Error al obtener predicciones de especialistas: {str(e)}"
        logging.error(error)
        raise ValueError(error)


def predict(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza una predicción usando el juez final.
    
    El juez toma las predicciones de los 4 especialistas y hace la decisión final.
    
    Args:
        data: DataFrame preprocesado con las características
        
    Returns:
        Diccionario con la predicción del juez y los scores de los especialistas
        
    Raises:
        ValueError: Si hay errores en la validación, predicciones o decisión final
    """
    # 1. Validar datos para todos los modelos
    validate_input_data(data)
    
    # 2. Obtener predicciones de especialistas
    specialist_scores, specialist_results = collect_specialist_predictions(data)
    
    # 3. Cargar modelo del juez y hacer predicción
    try:
        judge_model = load_model()
        logging.info("Realizando predicción con el juez")
        
        # Obtener probabilidades y predicción
        probas = judge_model.predict_proba(specialist_scores.values)
        score = probas[0][1]  # Probabilidad de clase 1 (CONFIRMED)
        prediction = judge_model.predict(specialist_scores.values)[0]
        prediccion_text = "CONFIRMED" if prediction == 1 else "FALSE POSITIVE"
        
        logging.info(f"Predicción del juez exitosa: {prediccion_text} (score: {score})")
        
    except Exception as e:
        error = f"Error en predicción del juez: {str(e)}"
        logging.error(error)
        raise ValueError(error)
    
    # 4. Preparar resultado final
    return {
        "modelo": "judge",
        "score": round(float(score), 4),
        "prediccion": prediccion_text,
        "confianza": round(abs(score - 0.5) * 2, 4),
        "specialist_scores": specialist_results['scores'],
        "specialist_predictions": specialist_results['predictions']
    }
