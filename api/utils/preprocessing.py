# api/utils/preprocessing.py

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Tuple

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

# Cargar artefactos de preprocesamiento
IMPUTER = joblib.load(os.path.join(PROCESSED_PATH, "imputer.gz"))
SCALER = joblib.load(os.path.join(PROCESSED_PATH, "scaler.gz"))

# Intentar cargar columnas esperadas desde X_train.csv para validaciones más claras
EXPECTED_COLUMNS = None
try:
    train_cols_path = os.path.join(PROCESSED_PATH, "train_set", "X_train.csv")
    if os.path.exists(train_cols_path):
        EXPECTED_COLUMNS = list(pd.read_csv(train_cols_path, nrows=0).columns)
except Exception:
    EXPECTED_COLUMNS = None


def generar_cols_incertidumbre(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Genera columnas de incertidumbre (sigma, snr, rel_unc) para las columnas especificadas.
    
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
            # Crear sigma (incertidumbre máxima)
            df[f"{col}_sigma"] = df[[err1_col, err2_col]].abs().max(axis=1)
            
            # Crear SNR (señal a ruido)
            epsilon = 1e-8
            df[f"{col}_snr"] = df[col] / (df[f"{col}_sigma"] + epsilon)
            
            # Crear incertidumbre relativa
            df[f"{col}_rel_unc"] = df[f"{col}_sigma"] / (df[col].abs() + epsilon)
            
            # Eliminar las columnas de error originales
            df = df.drop(columns=[err1_col, err2_col], errors="ignore")
    
    return df


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada para predicción.
    Garantiza la generación de todas las características necesarias para cada modelo.
    
    Args:
        data: Diccionario con los datos de entrada
        
    Returns:
        DataFrame preprocesado listo para predicción
    """
    # Convertir el diccionario a DataFrame
    df = pd.DataFrame([data])
    
    # Solo mantener columnas que están en los datos de entrenamiento
    df = df[df.columns.intersection(IMPUTER.feature_names_in_)]
    
    # Obtener todas las características que requieren incertidumbre
    from .feature_groups import UNCERTAINTY_FEATURES
    
    # Combinar todas las columnas que necesitan procesamiento de incertidumbre
    cols_con_incertidumbre = list(set(
        UNCERTAINTY_FEATURES['fotometria'] +
        UNCERTAINTY_FEATURES['orbital'] +
        UNCERTAINTY_FEATURES['estelar']
    ))
    
    # Generar columnas de incertidumbre
    df = generar_cols_incertidumbre(df, cols_con_incertidumbre)
    
    # Eliminar koi_ror y sus errores si existen (no se usan en entrenamiento)
    cols_to_drop = ['koi_ror', 'koi_ror_err1', 'koi_ror_err2']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Reordenar columnas para que coincidan con el orden del transformador
    train_columns = list(IMPUTER.feature_names_in_)
    available_columns = [col for col in train_columns if col in df.columns]
    df = df[available_columns]
    
    # Rellenar columnas faltantes con 0
    missing_columns = set(train_columns) - set(available_columns)
    for col in missing_columns:
        df[col] = 0
        
    # Asegurar el mismo orden que en entrenamiento
    df = df[train_columns]
    
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


def validate_features_for_model(df: pd.DataFrame, model_type: str) -> Tuple[bool, str]:
    """
    Valida que estén presentes todas las características necesarias para un modelo específico.
    
    Args:
        df: DataFrame con los datos procesados
        model_type: Tipo de modelo ('fotometria', 'orbital', 'estelar', 'falsos_positivos', 'judge')
    
    Returns:
        (True, "") si las características están completas, (False, error_msg) si no
    """
    from .feature_groups import get_feature_group, get_base_features
    import logging
    
    # Validar características base
    base_features = get_base_features(model_type)
    missing_base = [f for f in base_features if f not in df.columns]
    if missing_base:
        return False, f"Faltan características base requeridas para el modelo {model_type}: {missing_base}"
    
    # Validar todas las características requeridas
    required_features = get_feature_group(model_type)
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        return False, f"Faltan características derivadas requeridas para el modelo {model_type}: {missing_features}"
    
    logging.info(f"Características encontradas para {model_type}: {[f for f in df.columns if f in required_features]}")
    return True, ""

def validate_input(data: Dict[str, Any], model_type: str = None) -> Tuple[bool, str]:
    """
    Valida los datos de entrada básicos y las características necesarias según el modelo.
    
    Args:
        data: Diccionario con los datos de entrada
        model_type: Tipo de modelo para validar características específicas
    
    Returns:
        (True, "") si los datos son válidos, (False, error_msg) si no
    """
    if not data:
        return False, "No se proporcionaron datos"
    
    if "data" not in data:
        return False, "El campo 'data' es requerido"
        
    input_data = data["data"]
        
    # Importar los features requeridos
    from .feature_groups import (
        BASE_FEATURES,
        UNCERTAINTY_FEATURES,
        FOTOMETRIA_FEATURES,
        ORBITAL_FEATURES,
        ESTELAR_FEATURES,
        FALSOS_POSITIVOS_FEATURES
    )
    
    # Si no se especifica modelo, solo validar estructura básica
    if not model_type:
        return True, ""

    # Obtener los features base requeridos para este modelo
    model_base_features = BASE_FEATURES.get(model_type, [])
    model_uncertainty_features = UNCERTAINTY_FEATURES.get(model_type, [])
    
    # 1. Validar features base del modelo específico
    missing_base = []
    invalid_type = []
    
    for feature in model_base_features:
        if feature not in input_data:
            missing_base.append(feature)
        elif not isinstance(input_data[feature], (int, float)):
            invalid_type.append(feature)
    
    if missing_base:
        return False, f"Faltan features base requeridos por {model_type}: {missing_base}"
    if invalid_type:
        return False, f"Features con tipo inválido para {model_type} (deben ser numéricos): {invalid_type}"
    
    # 2. Validar incertidumbres requeridas por este modelo
    missing_uncertainties = []
    invalid_uncertainties = []
    
    for feature in model_uncertainty_features:
        err1_key = f"{feature}_err1"
        err2_key = f"{feature}_err2"
        
        # Verificar presencia
        if err1_key not in input_data or err2_key not in input_data:
            missing_uncertainties.append(f"{feature} (err1 y err2)")
            continue
        
        # Verificar tipos
        if not isinstance(input_data[err1_key], (int, float)) or not isinstance(input_data[err2_key], (int, float)):
            invalid_uncertainties.append(f"{feature} (valores no numéricos)")
            continue
        
        # Verificar signos
        if input_data[err1_key] < 0:
            invalid_uncertainties.append(f"{feature} (err1 debe ser positivo)")
        if input_data[err2_key] > 0:
            invalid_uncertainties.append(f"{feature} (err2 debe ser negativo)")
    
    if missing_uncertainties:
        return False, f"Faltan incertidumbres requeridas por {model_type}: {missing_uncertainties}"
    if invalid_uncertainties:
        return False, f"Incertidumbres inválidas para {model_type}: {invalid_uncertainties}"
    
    # 3. Validar features específicos del modelo
    if model_type == "ensemble" or model_type == "judge":
        # Para ensemble y judge necesitamos validar los features de todos los modelos
        for model_name, features in {
            'fotometria': FOTOMETRIA_FEATURES,
            'orbital': ORBITAL_FEATURES,
            'estelar': ESTELAR_FEATURES,
            'falsos_positivos': FALSOS_POSITIVOS_FEATURES
        }.items():
            missing_required = []
            for feature in features:
                if not feature.endswith(('_sigma', '_snr', '_rel_unc')) and feature not in input_data:
                    missing_required.append(feature)
            if missing_required:
                return False, f"Features de {model_name} requeridos por {model_type}: {missing_required}"
    else:
        # Para modelos individuales, solo validar sus propios features
        required_features = {
            'fotometria': FOTOMETRIA_FEATURES,
            'orbital': ORBITAL_FEATURES,
            'estelar': ESTELAR_FEATURES,
            'falsos_positivos': FALSOS_POSITIVOS_FEATURES
        }.get(model_type, [])
        
        missing_required = []
        for feature in required_features:
            if not feature.endswith(('_sigma', '_snr', '_rel_unc')) and feature not in input_data:
                missing_required.append(feature)
        
        if missing_required:
            return False, f"Features requeridos por {model_type}: {missing_required}"
    
    return True, ""  # Todo está válido
