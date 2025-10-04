# nasaSpace2025/scripts/train.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

# --- Añadir la ruta del proyecto al path para poder importar el modelo ---
# Esto asegura que Python pueda encontrar la carpeta 'model'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Importamos la clase del modelo desde su nueva ubicación
from model.m_fotometria import FotometriaNet

# --- Configuración de Rutas (ACTUALIZADO) ---
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_BASE_PATH, "train_set")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "weights")
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# --- Carga de Datos Procesados desde las carpetas correctas ---
print("Cargando datos preprocesados desde 'data/processed/train_set'...")
X_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.csv"))
y_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.csv"))

# --- Definir Columnas para el Modelo de Fotometría ---
# (Se mantiene igual, seleccionas las características para este modelo específico)
columnas_fotometria = [
    'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
    'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
    'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
    'koi_model_snr'
]
# Seleccionamos solo las columnas necesarias del DataFrame completo
X_train_fotometria = X_train_full[columnas_fotometria].values
y_train_full = y_train_full.values.flatten()

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fotometria, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# --- Crear Dataset y DataLoader de PyTorch ---
# (Esta sección no necesita cambios)
class KeplerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)

train_loader = DataLoader(KeplerDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(KeplerDataset(X_val, y_val), batch_size=32)

# --- Bucle de Entrenamiento ---
# (Esta sección no necesita cambios)
input_size = X_train.shape[1]
model = FotometriaNet(input_features=input_size)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 100

print("\n--- Iniciando Entrenamiento del Modelo de Fotometría ---")
for epoch in range(epochs):
    model.train()
    for features, labels in train_loader:
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Época [{epoch+1}/{epochs}], Pérdida: {loss.item():.4f}, Precisión en Validación: {accuracy:.2f}%')

# --- Guardar el Modelo Entrenado en la carpeta correcta ---
print("\nEntrenamiento finalizado. Guardando modelo...")
model_save_path = os.path.join(MODEL_OUTPUT_PATH, "fotometria_net.pth")
torch.save(model.state_dict(), model_save_path)
print(f"✅ Modelo guardado en: {model_save_path}")  