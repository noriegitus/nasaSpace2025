# api/utils/preprocessing.py

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_PATH = os.path.join(BASE_DIR, "data", "processed", "artifacts")

# Cargar artefactos de preprocesamiento
IMPUTER = joblib.load(os.path.join(ARTIFACTS_PATH, "imputer.gz"))
SCALER = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.gz"))


def generar_cols_incertidumbre(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Genera columnas de incertidumbre promedio y máxima para las columnas especificadas.
    
    Args:
        df: DataFrame con los datos
        cols: Lista de nombres de columnas base
        
    Returns:
        DataFrame con las columnas de incertidumbre añadidas
    """
    for col in cols:
        err1_col = f"{col}_err1"
        err2_col = f"{col}_err2"
        
        if err1_col in df.columns and err2_col in df.columns:
            df[f"{col}_incertidumbre_promedio"] = (
                df[err1_col].abs() + df[err2_col].abs()
            ) / 2
            df[f"{col}_incertidumbre_maxima"] = df[[err1_col, err2_col]].abs().max(axis=1)
            
            # Eliminar las columnas de error originales
            df = df.drop(columns=[err1_col, err2_col], errors="ignore")
    
    return df


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada para predicción.
    
    Args:
        data: Diccionario con los datos de entrada
        
    Returns:
        DataFrame preprocesado listo para predicción
    """
    # Convertir el diccionario a DataFrame
    df = pd.DataFrame([data])
    
    # Columnas con incertidumbre
    cols_con_incertidumbre = [
        "koi_period",
        "koi_time0bk",
        "koi_duration",
        "koi_depth",
        "koi_ror",
        "koi_srad",
        "koi_steff",
        "koi_slogg",
        "koi_prad",
        "koi_insol"
    ]
    
    # Generar columnas de incertidumbre
    df = generar_cols_incertidumbre(df, cols_con_incertidumbre)
    
    # Aplicar imputación
    df_imputed = pd.DataFrame(
        IMPUTER.transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Aplicar escalado
    df_scaled = pd.DataFrame(
        SCALER.transform(df_imputed),
        columns=df_imputed.columns,
        index=df_imputed.index
    )
    
    return df_scaled


def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Valida que los datos de entrada contengan los campos requeridos.
    
    Args:
        data: Diccionario con los datos de entrada
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    required_fields = [
        "koi_period", "koi_time0bk", "koi_duration", "koi_depth",
        "koi_ror", "koi_srad", "koi_steff", "koi_slogg",
        "koi_prad", "koi_insol"
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Faltan campos requeridos: {', '.join(missing_fields)}"
    
    return True, ""
