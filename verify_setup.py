# verify_setup.py
"""
Script para verificar que todos los componentes necesarios esten en su lugar.
"""

import os
import sys

def check_file_exists(path, name):
    """Verifica si un archivo existe."""
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[ERROR]"
    print(f"{status} {name}: {path}")
    return exists

def check_directory_exists(path, name):
    """Verifica si un directorio existe."""
    exists = os.path.isdir(path)
    status = "[OK]" if exists else "[ERROR]"
    print(f"{status} {name}: {path}")
    return exists

def main():
    print("=" * 70)
    print("NASA Space Apps 2025 - Verificacion de Configuracion")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    print("\n[DIRECTORIOS] Verificando Estructura...")
    dirs = [
        (os.path.join(base_dir, "api"), "API Directory"),
        (os.path.join(base_dir, "api", "routes"), "Routes Directory"),
        (os.path.join(base_dir, "api", "services"), "Services Directory"),
        (os.path.join(base_dir, "api", "utils"), "Utils Directory"),
        (os.path.join(base_dir, "model", "architecture"), "Model Architecture"),
        (os.path.join(base_dir, "outputs", "weights"), "Weights Directory"),
        (os.path.join(base_dir, "data", "processed", "artifacts"), "Artifacts Directory"),
    ]
    
    for path, name in dirs:
        if not check_directory_exists(path, name):
            all_ok = False
    
    print("\n[API] Verificando Archivos de API...")
    api_files = [
        (os.path.join(base_dir, "api", "main.py"), "Main API File"),
        (os.path.join(base_dir, "api", "routes", "fotometria.py"), "Fotometria Route"),
        (os.path.join(base_dir, "api", "routes", "orbital.py"), "Orbital Route"),
        (os.path.join(base_dir, "api", "routes", "estelar.py"), "Estelar Route"),
        (os.path.join(base_dir, "api", "routes", "falsos_positivos.py"), "Falsos Positivos Route"),
        (os.path.join(base_dir, "api", "routes", "ensemble.py"), "Ensemble Route"),
    ]
    
    for path, name in api_files:
        if not check_file_exists(path, name):
            all_ok = False
    
    print("\n[MODELOS] Verificando Arquitecturas...")
    model_files = [
        (os.path.join(base_dir, "model", "architecture", "m_fotometria.py"), "Fotometria Model"),
        (os.path.join(base_dir, "model", "architecture", "m_orbital.py"), "Orbital Model"),
        (os.path.join(base_dir, "model", "architecture", "m_estrella.py"), "Estelar Model"),
        (os.path.join(base_dir, "model", "architecture", "m_falsospositivos.py"), "Falsos Positivos Model"),
    ]
    
    for path, name in model_files:
        if not check_file_exists(path, name):
            all_ok = False
    
    print("\n[PESOS] Verificando Pesos de Modelos...")
    weight_files = [
        (os.path.join(base_dir, "outputs", "weights", "fotometria_net.pth"), "Fotometria Weights"),
        (os.path.join(base_dir, "outputs", "weights", "orbital_net.pth"), "Orbital Weights"),
        (os.path.join(base_dir, "outputs", "weights", "estelar_net.pth"), "Estelar Weights"),
        (os.path.join(base_dir, "outputs", "weights", "falsos_positivos_net.pth"), "Falsos Positivos Weights"),
    ]
    
    for path, name in weight_files:
        if not check_file_exists(path, name):
            all_ok = False
    
    print("\n[ARTEFACTOS] Verificando Preprocesamiento...")
    artifact_files = [
        (os.path.join(base_dir, "data", "processed", "artifacts", "imputer.gz"), "Imputer"),
        (os.path.join(base_dir, "data", "processed", "artifacts", "scaler.gz"), "Scaler"),
    ]
    
    for path, name in artifact_files:
        if not check_file_exists(path, name):
            all_ok = False
    
    print("\n[DEPENDENCIAS] Verificando Paquetes de Python...")
    try:
        import fastapi
        print("[OK] FastAPI instalado")
    except ImportError:
        print("[ERROR] FastAPI no instalado")
        all_ok = False
    
    try:
        import uvicorn
        print("[OK] Uvicorn instalado")
    except ImportError:
        print("[ERROR] Uvicorn no instalado")
        all_ok = False
    
    try:
        import torch
        print("[OK] PyTorch instalado")
        print(f"      Version: {torch.__version__}")
        print(f"      CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("[ERROR] PyTorch no instalado")
        all_ok = False
    
    try:
        import pandas
        print("[OK] Pandas instalado")
    except ImportError:
        print("[ERROR] Pandas no instalado")
        all_ok = False
    
    try:
        import sklearn
        print("[OK] Scikit-learn instalado")
    except ImportError:
        print("[ERROR] Scikit-learn no instalado")
        all_ok = False
    
    try:
        import joblib
        print("[OK] Joblib instalado")
    except ImportError:
        print("[ERROR] Joblib no instalado")
        all_ok = False
    
    print("\n" + "=" * 70)
    if all_ok:
        print("[EXITO] Todo esta listo! Puedes ejecutar el API con:")
        print("        cd api")
        print("        python main.py")
        print("")
        print("Luego visita: http://localhost:8000/docs")
    else:
        print("[ERROR] Hay componentes faltantes. Revisa los errores arriba.")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()
