# nasaSpace2025/scripts/preprocess.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Kepler.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# Crear carpeta base
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- Carga de Datos ---
print("Cargando datos crudos...")
df = pd.read_csv(RAW_DATA_PATH, comment='#', skiprows=53)

# --- Limpieza Inicial ---
print("Realizando limpieza inicial...")
columnas_a_eliminar = [
    'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
    'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname'
]
df = df.drop(columns=columnas_a_eliminar, errors="ignore")

# --- Separación lógica ---
print("Separando datos en conjuntos de entrenamiento y predicción...")
df_entrenamiento = df[df['koi_disposition'] != 'CANDIDATE'].copy()
df_prediccion = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# --- Variables con incertidumbre ---
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

def generar_cols_incertidumbre(df, cols):
    for col in cols:
        err1 = col + "_err1"
        err2 = col + "_err2"
        if err1 in df.columns and err2 in df.columns:
            sigma_col = col + "_sigma"
            snr_col = col + "_snr"
            rel_unc_col = col + "_rel_unc"

            # sigma: magnitud del error (tomando simetría)
            df[sigma_col] = (df[err1].abs() + df[err2].abs()) / 2

            # snr: valor absoluto entre incertidumbre
            df[snr_col] = df[col] / df[sigma_col].replace(0, np.nan)

            # rel_unc: incertidumbre relativa
            df[rel_unc_col] = df[sigma_col] / df[col].replace(0, np.nan)
    return df

print("Generando columnas de incertidumbre...")
df_entrenamiento = generar_cols_incertidumbre(df_entrenamiento, cols_con_incertidumbre)
df_prediccion = generar_cols_incertidumbre(df_prediccion, cols_con_incertidumbre)

# --- Preparar entrenamiento ---
print("Preparando conjunto de entrenamiento...")
mapeo_objetivo = {'CONFIRMED': 1, 'FALSE POSITIVE': 0}
y_train = df_entrenamiento['koi_disposition'].map(mapeo_objetivo)
X_train = df_entrenamiento.drop(columns=['koi_disposition'])

# --- Preprocesamiento ---
imputer = KNNImputer(n_neighbors=5)
print("Ajustando KNN Imputer en entrenamiento...")
X_train_imputed = imputer.fit_transform(X_train)
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)

scaler = StandardScaler()
print("Ajustando StandardScaler en entrenamiento...")
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# --- Preprocesar predicción ---
print("Preparando conjunto de predicción...")
X_predict = df_prediccion.drop(columns=['koi_disposition'])
X_predict_imputed = imputer.transform(X_predict)
X_predict_scaled = scaler.transform(X_predict_imputed)
X_predict = pd.DataFrame(X_predict_scaled, columns=X_predict.columns)

# --- Guardar en subcarpetas ---
TRAIN_SET_PATH = os.path.join(PROCESSED_DATA_PATH, "train_set")
PREDICTION_SET_PATH = os.path.join(PROCESSED_DATA_PATH, "prediction_set")
os.makedirs(TRAIN_SET_PATH, exist_ok=True)
os.makedirs(PREDICTION_SET_PATH, exist_ok=True)

print("Guardando archivos procesados...")
X_train.to_csv(os.path.join(TRAIN_SET_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(TRAIN_SET_PATH, "y_train.csv"), index=False)
X_predict.to_csv(os.path.join(PREDICTION_SET_PATH, "X_predict.csv"), index=False)

# Guardar scaler e imputer
joblib.dump(scaler, os.path.join(PROCESSED_DATA_PATH, "scaler.gz"))
joblib.dump(imputer, os.path.join(PROCESSED_DATA_PATH, "imputer.gz"))

print("\n✅ Preprocesamiento completado!")
print(f"Entrenamiento: {X_train.shape}")
print(f"Predicción: {X_predict.shape}")
