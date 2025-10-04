# model/train/train_judge.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y RUTAS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")
JUDGE_DATA_PATH = os.path.join(PROCESSED_BASE_PATH, "judge_set")
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights")
os.makedirs(WEIGHTS_PATH, exist_ok=True)

# --------------------------------------------------------------------------
# 2. ENTRENAMIENTO DEL JUEZ
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando el entrenamiento del Juez (Regresión Logística)...")
    
    # Cargar los datos generados en el paso anterior
    X_judge_path = os.path.join(JUDGE_DATA_PATH, "X_judge.csv")
    y_judge_path = os.path.join(JUDGE_DATA_PATH, "y_judge.csv")
    
    X_judge = pd.read_csv(X_judge_path)
    y_judge = pd.read_csv(y_judge_path).values.ravel() # .ravel() para convertirlo a un array 1D

    # Instanciar y entrenar el modelo de Regresión Logística
    # Usamos class_weight='balanced' para manejar el desbalance de clases
    print("\nEntrenando el modelo...")
    judge_model = LogisticRegression(random_state=42, class_weight='balanced')
    judge_model.fit(X_judge, y_judge)
    
    print("¡Entrenamiento completado!")

    # Evaluar el modelo con los mismos datos de entrenamiento para ver su rendimiento
    print("\n--- Reporte de Clasificación (sobre datos de entrenamiento) ---")
    predictions = judge_model.predict(X_judge)
    print(classification_report(y_judge, predictions, target_names=['Falso Positivo', 'Confirmado']))

    # Guardar el modelo entrenado
    model_save_path = os.path.join(WEIGHTS_PATH, "judge_model.joblib")
    joblib.dump(judge_model, model_save_path)
    
    print(f"\n✅ Modelo del Juez guardado en: '{model_save_path}'")