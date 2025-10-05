import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- 1. Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Kepler.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- 2. Carga y Limpieza Inicial ---
print("Cargando datos crudos...")
df = pd.read_csv(RAW_DATA_PATH, comment='#')

print("Realizando limpieza inicial...")
columnas_a_eliminar = [
    'kepler_name', 'koi_pdisposition',
    'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname'
]
df = df.drop(columns=columnas_a_eliminar, errors="ignore")

# --- 3. Separación Lógica ---
print("Separando datos en conjuntos de entrenamiento y predicción...")
df_entrenamiento = df[df['koi_disposition'] != 'CANDIDATE'].copy()
df_prediccion = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# --- 4. Preparación de Datos (Separar IDs, Features y Target) ---
print("Separando IDs, features y targets...")

kepid_train = df_entrenamiento['kepid'].copy()
y_train = df_entrenamiento['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
X_train = df_entrenamiento.drop(columns=['kepid', 'koi_disposition'])

kepid_predict = df_prediccion['kepid'].copy()
X_predict = df_prediccion.drop(columns=['kepid', 'koi_disposition'])

# --- 5. Ingeniería de Características (Incertidumbres) ---
cols_con_incertidumbre = [
    "koi_period", "koi_time0bk", "koi_duration", "koi_depth", "koi_ror",
    "koi_srad", "koi_steff", "koi_slogg", "koi_prad", "koi_insol"
]

def generar_cols_incertidumbre(df, cols):
    for col in cols:
        err1 = f"{col}_err1"
        err2 = f"{col}_err2"
        if err1 in df.columns and err2 in df.columns:
            df[f"{col}_sigma"] = df[[err1, err2]].abs().max(axis=1)
            epsilon = 1e-8
            df[f"{col}_snr"] = df[col] / (df[f"{col}_sigma"] + epsilon)
            df[f"{col}_rel_unc"] = df[f"{col}_sigma"] / (df[col].abs() + epsilon)
            df.drop(columns=[err1, err2], inplace=True)
    return df

print("Generando columnas de incertidumbre...")
X_train = generar_cols_incertidumbre(X_train, cols_con_incertidumbre)
X_predict = generar_cols_incertidumbre(X_predict, cols_con_incertidumbre)

X_predict = X_predict[X_train.columns]

# --- 6. Preprocesamiento (Imputación y Escalado) ---

# Seleccionar solo columnas numéricas para imputar
columnas_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()

imputer = KNNImputer(n_neighbors=5)
print("Ajustando y aplicando KNN Imputer...")

# Imputar solo en columnas numéricas
X_train_num_imputed = imputer.fit_transform(X_train[columnas_numericas])
X_predict_num_imputed = imputer.transform(X_predict[columnas_numericas])

# Reconstruir DataFrames imputados solo en columnas numéricas
X_train_imputed = X_train.copy()
X_train_imputed[columnas_numericas] = X_train_num_imputed

X_predict_imputed = X_predict.copy()
X_predict_imputed[columnas_numericas] = X_predict_num_imputed

scaler = StandardScaler()
print("Ajustando y aplicando StandardScaler...")
X_train_scaled = scaler.fit_transform(X_train_imputed[columnas_numericas])
X_predict_scaled = scaler.transform(X_predict_imputed[columnas_numericas])

X_train_processed = X_train_imputed.copy()
X_train_processed[columnas_numericas] = X_train_scaled

X_predict_processed = X_predict_imputed.copy()
X_predict_processed[columnas_numericas] = X_predict_scaled

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

joblib.dump(scaler, os.path.join(PROCESSED_DATA_PATH, "scaler.gz"))
joblib.dump(imputer, os.path.join(PROCESSED_DATA_PATH, "imputer.gz"))

print("\n✅ Preprocesamiento completado!")
print(f"Archivos de entrenamiento guardados: {X_train_processed.shape}, {y_train.shape}")
print(f"Archivos de predicción guardados: {X_predict_processed.shape}")
