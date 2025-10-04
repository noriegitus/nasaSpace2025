# model/train/generate_judge_data.py

import pandas as pd
import numpy as np
import torch
import os
import sys

# --- Añadir la ruta del proyecto al path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# --- Importar TODAS las arquitecturas y configuraciones ---
from model.train.train_specialists import SPECIALIST_CONFIG

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_BASE_PATH, "train_set")
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights")
JUDGE_DATA_PATH = os.path.join(PROCESSED_BASE_PATH, "judge_set")
os.makedirs(JUDGE_DATA_PATH, exist_ok=True)

# --------------------------------------------------------------------------
# 2. FUNCIÓN AUXILIAR PARA OBTENER SCORES
# --------------------------------------------------------------------------
def get_specialist_scores(specialist_name, full_data):
    """Carga un especialista, procesa los datos y devuelve sus scores."""
    print(f"Procesando con el especialista: '{specialist_name}'...")
    
    # Obtener configuración del especialista
    config = SPECIALIST_CONFIG[specialist_name]
    model_class = config['model_class']
    feature_columns = config['feature_columns']
    model_filename = config['output_filename']
    
    # Preparar datos de entrada para este especialista
    specialist_data = full_data[feature_columns].values
    specialist_tensor = torch.tensor(specialist_data, dtype=torch.float32)
    
    # Cargar modelo y pesos
    input_features = len(feature_columns)
    model = model_class(input_features=input_features)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH, model_filename)))
    model.eval()
    
    # Obtener scores
    with torch.no_grad():
        logits = model(specialist_tensor)
        scores = torch.sigmoid(logits).numpy().flatten()
        
    return scores

# --------------------------------------------------------------------------
# 3. BUCLE PRINCIPAL
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando la generación de datos para el Juez...")
    
    # Cargar el dataset de entrenamiento completo
    X_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.csv"))
    y_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.csv"))

    # Crear un nuevo DataFrame para los scores que alimentarán al Juez
    X_judge = pd.DataFrame()

    # Iterar sobre cada especialista para obtener sus scores
    for name in SPECIALIST_CONFIG.keys():
        scores = get_specialist_scores(name, X_train_full)
        X_judge[f'score_{name}'] = scores
        
    # Las etiquetas (y) siguen siendo las mismas
    y_judge = y_train_full
    
    # Guardar el nuevo dataset para el Juez
    X_judge_path = os.path.join(JUDGE_DATA_PATH, "X_judge.csv")
    y_judge_path = os.path.join(JUDGE_DATA_PATH, "y_judge.csv")
    
    X_judge.to_csv(X_judge_path, index=False)
    y_judge.to_csv(y_judge_path, index=False)
    
    print("\n--- Vista previa de los datos del Juez (X_judge.csv): ---")
    print(X_judge.head())
    
    print(f"\n✅ ¡Datos para el Juez generados y guardados en '{JUDGE_DATA_PATH}'!")