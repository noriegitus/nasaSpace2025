# scripts/run_full_prediction.py (versión corregida y lista para usar)

import pandas as pd
import torch
import joblib
import os
import sys

# --- Añadir la ruta del proyecto al path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# --- Importar arquitecturas y configuración ---
from model.train.train_specialists import SPECIALIST_CONFIG

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS
# --------------------------------------------------------------------------
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")
PREDICTION_DATA_PATH = os.path.join(PROCESSED_BASE_PATH, "prediction_set")
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "weights")
FINAL_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "predictions")
os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)

# --------------------------------------------------------------------------
# 2. FUNCIÓN PARA CARGAR TODOS LOS MODELOS
# --------------------------------------------------------------------------
def load_all_models():
    """Carga los 4 especialistas de PyTorch y el Juez de Scikit-learn."""
    print("Cargando modelos entrenados...")
    
    # Cargar especialistas
    specialist_models = {}
    for name, config in SPECIALIST_CONFIG.items():
        model_class = config['model_class']
        model_filename = config['output_filename']
        input_features = len(config['feature_columns'])
        
        model = model_class(input_features=input_features)
        model_path = os.path.join(WEIGHTS_PATH, model_filename)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        specialist_models[name] = model
    
    # Cargar Juez
    judge_model_path = os.path.join(WEIGHTS_PATH, "judge_model.joblib")
    judge_model = joblib.load(judge_model_path)
    
    print("✅ Todos los modelos han sido cargados.")
    return specialist_models, judge_model

# --------------------------------------------------------------------------
# 3. SCRIPT PRINCIPAL DE PREDICCIÓN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- INICIANDO PIPELINE DE PREDICCIÓN COMPLETO ---")
    
    # Cargar los datos de los CANDIDATOS que ya fueron procesados
    X_candidate_path = os.path.join(PREDICTION_DATA_PATH, "X_predict.csv")
    print(f"Cargando datos de candidatos desde: {X_candidate_path}")
    X_candidate_full = pd.read_csv(X_candidate_path)
    
    # --- CAMBIO 1: Separar IDs de las features justo después de cargar ---
    kepids_para_reporte = X_candidate_full['kepid']
    X_candidate_features = X_candidate_full.drop(columns=['kepid'])
    
    # Cargar todos los modelos
    specialist_models, judge_model = load_all_models()
    
    # --- Paso 1: Obtener scores de los especialistas ---
    print("\nObteniendo scores de los especialistas...")
    specialist_scores = pd.DataFrame()
    for name, model in specialist_models.items():
        config = SPECIALIST_CONFIG[name]
        feature_columns = config['feature_columns']
        
        # --- CAMBIO 2: Usar el DataFrame que solo contiene features ---
        candidate_data_subset = X_candidate_features[feature_columns].values
        candidate_tensor = torch.tensor(candidate_data_subset, dtype=torch.float32)
        
        # Obtener scores
        with torch.no_grad():
            logits = model(candidate_tensor)
            scores = torch.sigmoid(logits).numpy().flatten()
        
        specialist_scores[f'score_{name}'] = scores

    print("✅ Scores de especialistas calculados.")
    print(specialist_scores.head())

    # --- Paso 2: Obtener el veredicto del Juez ---
    print("\nConsultando al Juez para el veredicto final...")
    
    # El Juez usa los scores de los especialistas como sus features
    final_predictions = judge_model.predict(specialist_scores)
    final_probabilities = judge_model.predict_proba(specialist_scores)
    
    # La probabilidad de ser "Confirmado" (clase 1)
    confidence_scores = final_probabilities[:, 1]
    
    print("✅ Veredicto final emitido.")

    # --- Paso 3: Consolidar y guardar los resultados ---
    print("\nGenerando reporte final de predicciones...")
    
    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame({
        # --- CAMBIO 3: Usar la variable de kepids guardada al principio ---
        'kepid': kepids_para_reporte,
        **specialist_scores, # Añade todas las columnas de scores
        'confianza_planeta': confidence_scores,
        'veredicto_final_code': final_predictions
    })
    
    # Mapear el código del veredicto a texto legible
    results_df['veredicto_final'] = results_df['veredicto_final_code'].map({
        1: 'Planeta Potencial',
        0: 'Falso Positivo Probable'
    })
    
    # Guardar el archivo CSV final
    output_filepath = os.path.join(FINAL_OUTPUT_PATH, "final_predictions.csv")
    results_df.to_csv(output_filepath, index=False)
    
    print("\n--- VISTA PREVIA DE LOS RESULTADOS FINALES ---")
    print(results_df.head())
    
    print(f"\n🎉 ¡Proceso completado! Los resultados se han guardado en '{output_filepath}'")