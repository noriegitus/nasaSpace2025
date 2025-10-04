# api/services/fotometria_service.py

import torch
import os
import sys
import pandas as pd
from typing import Dict, Any

# Agregar el directorio raíz al path para importar los modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from model.architecture.m_fotometria import FotometriaNet
from api.utils.feature_groups import get_feature_group

# Configuración
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights", "fotometria_net.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = None


def load_model():
    """Carga el modelo de fotometría."""
    global model
    if model is None:
        features = get_feature_group('fotometria')
        model = FotometriaNet(input_features=len(features))
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    return model


def predict(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza una predicción usando el modelo de fotometría.
    
    Args:
        data: DataFrame preprocesado con las características
        
    Returns:
        Diccionario con la predicción y el score
    """
    model = load_model()
    features = get_feature_group('fotometria')
    
    # Seleccionar solo las características relevantes
    X = data[features].values
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Realizar predicción
    with torch.no_grad():
        output = model(X_tensor)
        score = torch.sigmoid(output).cpu().item()
    
    return {
        "modelo": "fotometria",
        "score": round(score, 4),
        "prediccion": "CONFIRMED" if score > 0.5 else "FALSE POSITIVE",
        "confianza": round(abs(score - 0.5) * 2, 4)  # Normalizado de 0 a 1
    }
