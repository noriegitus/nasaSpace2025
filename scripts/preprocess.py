# nasaSpace2025/scripts/preprocess.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- 1. Configuración de Rutas ---
# Esta ruta asume que el script está en la carpeta 'scripts'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Kepler.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- 2. Carga y Limpieza Inicial ---
print("Cargando datos crudos...")
# Tu archivo original tenía un nombre diferente, asegúrate que 'Kepler.csv' sea el correcto
df = pd.read_csv(RAW_DATA_PATH, comment='#')

print("Realizando limpieza inicial...")
# Eliminamos columnas que no son features, pero CONSERVAMOS 'kepid' y 'koi_disposition' por ahora
columnas_a_eliminar = [
    'kepoi_name', 'kepler_name', 'koi_pdisposition',
    'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname'
]
df = df.drop(columns=columnas_a_eliminar, errors="ignore")

# --- 3. Separación Lógica ---
print("Separando datos en conjuntos de entrenamiento y predicción...")
df_entrenamiento = df[df['koi_disposition'] != 'CANDIDATE'].copy()
df_prediccion = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# --- 4. Preparación de Datos (Separar IDs, Features y Target) ---
print("Separando IDs, features y targets...")

# Para el conjunto de entrenamiento
kepid_train = df_entrenamiento['kepid'].copy()
y_train = df_entrenamiento['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
X_train = df_entrenamiento.drop(columns=['kepid', 'koi_disposition'])

# Para el conjunto de predicción
kepid_predict = df_prediccion['kepid'].copy()
X_predict = df_prediccion.drop(columns=['kepid', 'koi_disposition'])

# --- 5. Ingeniería de Características (Incertidumbres) ---
cols_con_incertidumbre = [
    "koi_period", "koi_time0bk", "koi_duration", "koi_depth", "koi_ror",
    "koi_srad", "koi_steff", "koi_slogg", "koi_prad", "koi_insol"
]

# --- MODIFICADO: La función ahora elimina las columnas de error originales ---
def generar_cols_incertidumbre(df, cols):
    for col in cols:
        err1 = f"{col}_err1"
        err2 = f"{col}_err2"
        if err1 in df.columns and err2 in df.columns:
            # Crear sigma, snr, y rel_unc
            df[f"{col}_sigma"] = df[[err1, err2]].abs().max(axis=1)
            # Añadimos un valor pequeño (epsilon) para evitar división por cero
            epsilon = 1e-8
            df[f"{col}_snr"] = df[col] / (df[f"{col}_sigma"] + epsilon)
            df[f"{col}_rel_unc"] = df[f"{col}_sigma"] / (df[col].abs() + epsilon)
            # Eliminar las columnas de error originales que ya no necesitamos
            df.drop(columns=[err1, err2], inplace=True)
    return df

print("Generando columnas de incertidumbre...")
X_train = generar_cols_incertidumbre(X_train, cols_con_incertidumbre)
X_predict = generar_cols_incertidumbre(X_predict, cols_con_incertidumbre)

# Asegurar que ambos DataFrames tengan las mismas columnas en el mismo orden
X_predict = X_predict[X_train.columns]

# --- 6. Preprocesamiento (Imputación y Escalado) ---
imputer = KNNImputer(n_neighbors=5)
print("Ajustando y aplicando KNN Imputer...")
X_train_imputed = imputer.fit_transform(X_train)
X_predict_imputed = imputer.transform(X_predict)

scaler = StandardScaler()
print("Ajustando y aplicando StandardScaler...")
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_predict_scaled = scaler.transform(X_predict_imputed)

# Convertir de vuelta a DataFrames
X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_predict_processed = pd.DataFrame(X_predict_scaled, columns=X_predict.columns)

# --- 7. Re-unir IDs ---
print("Re-uniendo IDs a los datos procesados...")
X_train_processed.insert(0, 'kepid', kepid_train.reset_index(drop=True))
X_predict_processed.insert(0, 'kepid', kepid_predict.reset_index(drop=True))

# --- 8. Guardar Archivos ---
TRAIN_SET_PATH = os.path.join(PROCESSED_DATA_PATH, "train_set")
PREDICTION_SET_PATH = os.path.join(PROCESSED_DATA_PATH, "prediction_set")
os.makedirs(TRAIN_SET_PATH, exist_ok=True)
os.makedirs(PREDICTION_SET_PATH, exist_ok=True)

print("Guardando archivos procesados...")
X_train_processed.to_csv(os.path.join(TRAIN_SET_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(TRAIN_SET_PATH, "y_train.csv"), index=False)
X_predict_processed.to_csv(os.path.join(PREDICTION_SET_PATH, "X_predict.csv"), index=False)

# Guardar scaler e imputer
joblib.dump(scaler, os.path.join(PROCESSED_DATA_PATH, "scaler.gz"))
joblib.dump(imputer, os.path.join(PROCESSED_DATA_PATH, "imputer.gz"))

print("\n✅ Preprocesamiento completado!")
print(f"Archivos de entrenamiento guardados: {X_train_processed.shape}, {y_train.shape}")
print(f"Archivos de predicción guardados: {X_predict_processed.shape}")