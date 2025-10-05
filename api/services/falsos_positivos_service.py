# api/services/falsos_positivos_service.py

import torch
import os
import sys
import pandas as pd
from typing import Dict, Any

# Agregar el directorio raíz al path para importar los modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import logging
from model.architecture.m_falsospositivos import FalsosPositivosNet
from api.utils.feature_groups import get_feature_group

# Configuración
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights", "falsos_positivos_net.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = None


def load_model():
    """Carga el modelo de detección de falsos positivos."""
    global model
    if model is None:
        features = get_feature_group('falsos_positivos')
        model = FalsosPositivosNet(input_features=len(features))
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    return model


def validate_input_data(data: pd.DataFrame) -> None:
    """
    Valida que los datos de entrada contengan todas las características necesarias.
    
    Args:
        data: DataFrame con los datos de entrada
        
    Raises:
        ValueError: Si faltan características requeridas
    """
    from api.utils.preprocessing import validate_features_for_model
    
    logging.info(f"Validando datos de entrada: {data.columns.tolist()}")
    
    # Validar características base y derivadas
    is_valid, error_msg = validate_features_for_model(data, 'falsos_positivos')
    if not is_valid:
        logging.error(f"Error de validación: {error_msg}")
        raise ValueError(error_msg)


def prepare_features(data: pd.DataFrame) -> torch.Tensor:
    """
    Prepara las características para el modelo de falsos positivos.
    
    Args:
        data: DataFrame con los datos preprocesados
        
    Returns:
        Tensor de PyTorch con las características ordenadas
        
    Raises:
        ValueError: Si hay errores al preparar los datos
    """
    features = get_feature_group('falsos_positivos')
    logging.info(f"Preparando características: {features}")
    
    try:
        # Seleccionar y ordenar características
        X = data[features].values
        logging.info(f"Datos preparados shape: {X.shape}")
        
        # Convertir a tensor
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        logging.info(f"Tensor creado shape: {X_tensor.shape}")
        
        return X_tensor
        
    except Exception as e:
        logging.error(f"Error al preparar características: {str(e)}")
        raise ValueError(f"Error al preparar características: {str(e)}")


def predict(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza una predicción usando el modelo de detección de falsos positivos.
    
    Args:
        data: DataFrame preprocesado con las características
        
    Returns:
        Diccionario con la predicción, score y confianza
        
    Raises:
        ValueError: Si hay errores en la validación o predicción
    """
    # 1. Validar datos
    validate_input_data(data)
    
    # 2. Cargar modelo
    model = load_model()
    
    # 3. Preparar características
    X_tensor = prepare_features(data)
    
    # 4. Realizar predicción
    try:
        with torch.no_grad():
            output = model(X_tensor)
            score = output.cpu().item()  # Ya tiene sigmoid en la arquitectura
            logging.info(f"Predicción exitosa, score: {score}")
            
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        raise ValueError(f"Error al realizar predicción: {str(e)}")
    
    # 5. Preparar resultado
    return {
        "modelo": "falsos_positivos",
        "score": round(score, 4),
        "prediccion": "FALSE POSITIVE" if score > 0.5 else "CONFIRMED",
        "confianza": round(abs(score - 0.5) * 2, 4)  # Normalizado de 0 a 1
    }
