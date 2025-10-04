# nasaSpace2025/scripts/preprocess.py

import pandas as pd
import numpy as np
import os
import joblib  # Para guardar el scaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Kepler.csv")
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")

# Subcarpetas de salida
TRAIN_PATH = os.path.join(PROCESSED_BASE_PATH, "train_set")
PREDICT_PATH = os.path.join(PROCESSED_BASE_PATH, "prediction_set")
ARTIFACTS_PATH = os.path.join(PROCESSED_BASE_PATH, "artifacts")

# Crear carpetas si no existen
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(PREDICT_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# --- Carga de Datos ---
print("Cargando datos crudos...")
df = pd.read_csv(RAW_DATA_PATH, comment='#')

# --- Limpieza Inicial ---
print("Realizando limpieza inicial...")
columnas_a_eliminar = [
    'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
    'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname'
]
df = df.drop(columns=[c for c in columnas_a_eliminar if c in df.columns])

# --- Separación Lógica de Datos (Entrenamiento vs Predicción) ---
print("Separando datos en conjuntos de entrenamiento y predicción...")
df_entrenamiento = df[df['koi_disposition'] != 'CANDIDATE'].copy()
df_prediccion = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# --- Preparar Conjunto de Entrenamiento ---
print("Preparando conjunto de entrenamiento...")
mapeo_objetivo = {'CONFIRMED': 1, 'FALSE POSITIVE': 0}
y_train = df_entrenamiento['koi_disposition'].map(mapeo_objetivo)
X_train = df_entrenamiento.drop(columns=['koi_disposition'])

# --- Preprocesamiento (solo ajustado en entrenamiento) ---
print("Imputando valores nulos en entrenamiento...")
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)

print("Escalando características en entrenamiento...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# --- Preprocesar Conjunto de Predicción ---
print("Preparando conjunto de predicción...")
X_predict = df_prediccion.drop(columns=['koi_disposition'])

X_predict_imputed = imputer.transform(X_predict)
X_predict_scaled = scaler.transform(X_predict_imputed)
X_predict = pd.DataFrame(X_predict_scaled, columns=X_predict.columns)

# --- Guardar los Archivos Procesados ---
print("Guardando archivos procesados...")

# Train
X_train.to_csv(os.path.join(TRAIN_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(TRAIN_PATH, "y_train.csv"), index=False)

# Prediction
X_predict.to_csv(os.path.join(PREDICT_PATH, "X_predict.csv"), index=False)

# Artifacts (imputer + scaler)
joblib.dump(imputer, os.path.join(ARTIFACTS_PATH, "imputer.gz"))
joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.gz"))

print("\n✅ Preprocesamiento completado!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_predict shape: {X_predict.shape}")
