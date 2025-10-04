# api/utils/preprocessing.py

import os
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# --- CAMBIO CLAVE: Construcción de rutas robusta ---
# 1. Obtener la ruta del directorio actual (api/utils)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Subir dos niveles para llegar a la raíz del proyecto (nasaSpace2025/)
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 3. Construir la ruta a la carpeta donde se guardaron los artefactos
ARTIFACTS_PATH = os.path.join(BASE_DIR, "data", "processed")
# --- FIN DEL CAMBIO ---

# Cargar los artefactos de preprocesamiento una sola vez cuando el módulo se importa
try:
    IMPUTER = joblib.load(os.path.join(ARTIFACTS_PATH, "imputer.gz"))
    SCALER = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.gz"))
except FileNotFoundError as e:
    print(f"❌ Error al cargar artefactos: {e}")
    print("Asegúrate de haber ejecutado el script 'scripts/preprocess.py' primero.")
    # Salir o manejar el error como prefieras si los archivos son cruciales para iniciar
    raise e

def preprocess_input(input_data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Aplica el preprocesamiento (imputación y escalado) a los datos de entrada.
    
    Args:
        input_data: DataFrame con los datos crudos.
        feature_columns: Lista de columnas a las que se aplicará el preprocesamiento.
    
    Returns:
        DataFrame con los datos procesados.
    """
    # 1. Asegurar que solo se procesen las columnas esperadas
    data_to_process = input_data[feature_columns]
    
    # 2. Aplicar imputación
    imputed_data = IMPUTER.transform(data_to_process)
    
    # 3. Aplicar escalado
    scaled_data = SCALER.transform(imputed_data)
    
    # 4. Convertir de vuelta a un DataFrame con los nombres de columna correctos
    processed_df = pd.DataFrame(scaled_data, columns=feature_columns, index=data_to_process.index)
    
    return processed_df

def validate_input(input_data: BaseModel, expected_features: List[str]):
    """
    Valida que los datos de entrada contengan todas las características necesarias.
    """
    input_dict = input_data.dict()
    missing_features = [feat for feat in expected_features if feat not in input_dict]
    if missing_features:
        raise ValueError(f"Faltan las siguientes características en la entrada: {missing_features}")