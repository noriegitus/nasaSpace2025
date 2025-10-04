# api/services/ensemble_service.py

import pandas as pd
from typing import Dict, Any, List
import numpy as np

from api.services import fotometria_service
from api.services import orbital_service
from api.services import estelar_service
from api.services import falsos_positivos_service


def predict_ensemble(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza una predicción usando todos los modelos especialistas y combina los resultados.
    
    Args:
        data: DataFrame preprocesado con las características
        
    Returns:
        Diccionario con las predicciones de todos los modelos y la predicción final
    """
    # Obtener predicciones de cada modelo
    predictions = {
        'fotometria': fotometria_service.predict(data),
        'orbital': orbital_service.predict(data),
        'estelar': estelar_service.predict(data),
        'falsos_positivos': falsos_positivos_service.predict(data)
    }
    
    # Calcular score promedio ponderado
    # El modelo de falsos positivos tiene peso invertido
    scores = [
        predictions['fotometria']['score'],
        predictions['orbital']['score'],
        predictions['estelar']['score'],
        1 - predictions['falsos_positivos']['score']  # Invertir
    ]
    
    score_promedio = np.mean(scores)
    
    # Determinar predicción final
    prediccion_final = "CONFIRMED" if score_promedio > 0.5 else "FALSE POSITIVE"
    confianza_final = abs(score_promedio - 0.5) * 2
    
    # Contar votos
    votos_confirmed = sum(1 for pred in predictions.values() 
                         if pred['prediccion'] == 'CONFIRMED')
    votos_false_positive = 4 - votos_confirmed
    
    return {
        "prediccion_final": prediccion_final,
        "score_promedio": round(score_promedio, 4),
        "confianza_final": round(confianza_final, 4),
        "votos": {
            "CONFIRMED": votos_confirmed,
            "FALSE_POSITIVE": votos_false_positive
        },
        "modelos_individuales": predictions
    }
