# nasaSpace2025/model/train/train_specialists.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os
import sys

# --- Añadir la ruta del proyecto al path para poder importar modelos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# --- Importar TODAS las arquitecturas de los especialistas ---
# (Asegúrate que los nombres de archivo y clase son correctos)
from model.architecture.m_fotometria import FotometriaNet
from model.architecture.m_orbital import OrbitalNet
from model.architecture.m_estrella import PropiedadesEstelaresNet
from model.architecture.m_falsospositivos import FalsosPositivosNet

# --- Configuración de Rutas ---
PROCESSED_BASE_PATH = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_BASE_PATH, "train_set")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "weights")
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# --- CONFIGURACIÓN DE ESPECIALISTAS ---
SPECIALIST_CONFIG = {
    "fotometria": {
        "model_class": FotometriaNet,
        "output_filename": "fotometria_net.pth",
        "feature_columns": [
            'koi_duration', 'koi_duration_sigma', 'koi_duration_snr', 'koi_duration_rel_unc',
            'koi_depth', 'koi_depth_sigma', 'koi_depth_snr', 'koi_depth_rel_unc',
            'koi_impact',
            'koi_model_snr'
        ]
    },
    "orbital": {
        "model_class": OrbitalNet,
        "output_filename": "orbital_net.pth",
        "feature_columns": [
            'koi_period', 'koi_period_sigma', 'koi_period_snr', 'koi_period_rel_unc',
            'koi_time0bk', 'koi_time0bk_sigma', 'koi_time0bk_snr', 'koi_time0bk_rel_unc'
        ]
    },
    "estelar": {
        "model_class": PropiedadesEstelaresNet,
        "output_filename": "estelar_net.pth",
        "feature_columns": [
            'koi_srad', 'koi_srad_sigma', 'koi_srad_snr', 'koi_srad_rel_unc',
            'koi_steff', 'koi_steff_sigma', 'koi_steff_snr', 'koi_steff_rel_unc',
            'koi_slogg', 'koi_slogg_sigma', 'koi_slogg_snr', 'koi_slogg_rel_unc',
            'koi_prad', 'koi_prad_sigma', 'koi_prad_snr', 'koi_prad_rel_unc',
            'koi_insol', 'koi_insol_sigma', 'koi_insol_snr', 'koi_insol_rel_unc',
            'koi_teq',
            'koi_kepmag'
        ]
    },
    "falsos_positivos": {
        "model_class": FalsosPositivosNet,
        "output_filename": "falsos_positivos_net.pth",
        "feature_columns": [
            'koi_fpflag_nt',
            'koi_fpflag_ss',
            'koi_fpflag_co',
            'koi_fpflag_ec'
        ]
    }
}

# --- Dataset de PyTorch (Sin cambios) ---
class KeplerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)

# --- Función de Entrenamiento Reutilizable (Sin cambios) ---
def train_specialist(name, config, X_full, y_full):
    print(f"\n{'='*50}")
    print(f"🚀 Iniciando entrenamiento para el especialista: {name.upper()}")
    print(f"{'='*50}")

    print(f"Seleccionando {len(config['feature_columns'])} características...")
    X_specialist = X_full[config['feature_columns']].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_specialist, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    train_loader = DataLoader(KeplerDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(KeplerDataset(X_val, y_val), batch_size=32)

    input_size = X_train.shape[1]
    model = config['model_class'](input_features=input_size)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 100

    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = torch.sigmoid(model(features))
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f'Época [{epoch+1}/{epochs}], Pérdida: {loss.item():.4f}, Precisión en Validación: {accuracy:.2f}%')

    print("\nEntrenamiento finalizado. Guardando modelo...")
    model_save_path = os.path.join(MODEL_OUTPUT_PATH, config['output_filename'])
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modelo '{name}' guardado en: {model_save_path}")

# --------------------------------------------------------------------------
# --- BUCLE PRINCIPAL CON MENÚ INTERACTIVO ---
# --------------------------------------------------------------------------
if __name__ == "__main__":
    specialist_names = list(SPECIALIST_CONFIG.keys())
    
    while True:
        print("\n--- MENÚ DE ENTRENAMIENTO DE ESPECIALISTAS ---")
        for i, name in enumerate(specialist_names):
            print(f"  {i+1}. Entrenar especialista '{name}'")
        
        print(f"  {len(specialist_names) + 1}. Entrenar TODOS los especialistas")
        print(f"  {len(specialist_names) + 2}. Salir")
        
        try:
            choice = int(input("\nSelecciona una opción: "))
            
            if choice > 0 and choice <= len(specialist_names):
                # --- Opción: Entrenar un especialista específico ---
                chosen_name = specialist_names[choice - 1]
                print(f"\nHas elegido entrenar a '{chosen_name}'.")
                
                print("Cargando datos preprocesados...")
                X_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.csv"))
                y_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.csv")).values.flatten()
                
                train_specialist(chosen_name, SPECIALIST_CONFIG[chosen_name], X_train_full, y_train_full)
                break

            elif choice == len(specialist_names) + 1:
                # --- Opción: Entrenar todos ---
                print("\nHas elegido entrenar a TODOS los especialistas.")
                
                print("Cargando datos preprocesados...")
                X_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.csv"))
                y_train_full = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.csv")).values.flatten()
                
                for name, config in SPECIALIST_CONFIG.items():
                    train_specialist(name, config, X_train_full, y_train_full)
                
                print("\n🎉 ¡Sistema de especialistas completo! Todos los modelos han sido entrenados. 🎉")
                break
                
            elif choice == len(specialist_names) + 2:
                # --- Opción: Salir ---
                print("Saliendo del programa.")
                break
            else:
                print("❌ Opción no válida. Por favor, intenta de nuevo.")

        except ValueError:
            print("❌ Entrada no válida. Por favor, introduce un número.")