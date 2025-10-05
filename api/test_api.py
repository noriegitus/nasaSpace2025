# api/test_api.py
"""
Script de ejemplo para probar el API de detección de exoplanetas.
Incluye pruebas para los 4 especialistas, el ensemble y el juez final.
"""

import os
import sys
import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

from utils.feature_groups import (
    BASE_FEATURES,
    UNCERTAINTY_FEATURES,
    FOTOMETRIA_FEATURES,
    ORBITAL_FEATURES, 
    ESTELAR_FEATURES,
    FALSOS_POSITIVOS_FEATURES,
    JUDGE_FEATURES
)

# URL base del API
BASE_URL = "http://localhost:8000"

# ================================================================
# 🔧 Datos de prueba compatibles con TODOS los especialistas
# ================================================================
example_data = {
    "data": {
        # --- FALSOS POSITIVOS ---
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0,

        # --- FOTOMETRÍA ---
        "koi_duration": 2.9575,
        "koi_duration_err1": 0.0819,
        "koi_duration_err2": -0.0819,

        "koi_depth": 615.8,
        "koi_depth_err1": 19.5,
        "koi_depth_err2": -19.5,

        "koi_impact": 0.146,
        "koi_impact_err1": 0.318,
        "koi_impact_err2": -0.146,
        
        "koi_model_snr": 35.8,

        # --- ORBITAL ---
        "koi_period": 9.48803557,
        "koi_period_err1": 0.00002775,
        "koi_period_err2": -0.00002775,

        "koi_time0bk": 170.53875,
        "koi_time0bk_err1": 0.00216,
        "koi_time0bk_err2": -0.00216,

        # --- ESTELAR ---
        "koi_srad": 0.927,
        "koi_srad_err1": 0.105,
        "koi_srad_err2": -0.061,

        "koi_steff": 5455.0,
        "koi_steff_err1": 81.0,
        "koi_steff_err2": -81.0,

        "koi_slogg": 4.467,
        "koi_slogg_err1": 0.064,
        "koi_slogg_err2": -0.096,

        "koi_prad": 2.26,
        "koi_prad_err1": 0.26,
        "koi_prad_err2": -0.15,

        "koi_insol": 93.59,
        "koi_insol_err1": 29.45,
        "koi_insol_err2": -16.65,

        "koi_teq": 793.0,
        "koi_kepmag": 15.347,

        # --- METADATOS EXTRA (no usados directamente por los modelos) ---
        "ra": 291.93423,
        "dec": 48.141651,
        "koi_score": 1.0
    }
}


def validate_features(data: Dict[str, Any], model_type: str) -> List[str]:
    """
    Valida que los datos de entrada contengan todas las características requeridas.
    
    Args:
        data: Diccionario con los datos de entrada
        model_type: Tipo de modelo a validar ('fotometria', 'orbital', etc.)
    
    Returns:
        Lista de errores encontrados. Si está vacía, los datos son válidos.
    """
    errors = []
    
    # Validar estructura básica
    if not isinstance(data, dict):
        return ["Los datos deben ser un diccionario"]
    if "data" not in data:
        return ["El diccionario debe tener una clave 'data'"]
    if not isinstance(data["data"], dict):
        return ["data debe ser un diccionario"]
        
    input_data = data["data"]
    
    # Validar features base
    if model_type in BASE_FEATURES:
        for feature in BASE_FEATURES[model_type]:
            if feature not in input_data:
                errors.append(f"Falta la característica base '{feature}'")
            elif not isinstance(input_data[feature], (int, float)):
                errors.append(f"'{feature}' debe ser numérico")
                
    # Validar features con incertidumbre
    if model_type in UNCERTAINTY_FEATURES:
        for feature in UNCERTAINTY_FEATURES[model_type]:
            # Validar err1 y err2
            for suffix in ["_err1", "_err2"]:
                error_feat = f"{feature}{suffix}"
                if error_feat not in input_data:
                    errors.append(f"Falta la incertidumbre '{error_feat}'")
                elif not isinstance(input_data[error_feat], (int, float)):
                    errors.append(f"'{error_feat}' debe ser numérico")
                    
            # Validar lógica de incertidumbre
            if feature in input_data:
                err1 = input_data.get(f"{feature}_err1")
                err2 = input_data.get(f"{feature}_err2")
                if err1 is not None and err2 is not None:
                    if err1 < 0:
                        errors.append(f"'{feature}_err1' debe ser positivo")
                    if err2 > 0:
                        errors.append(f"'{feature}_err2' debe ser negativo")
                        
    return errors

def log_with_timestamp(msg: str, file=None):
    """Agrega timestamp a los mensajes de log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    if file:
        file.write(log_msg + "\n")
        file.flush()

def wait_for_server(max_attempts=30, delay=1):
    """Espera a que el servidor esté disponible."""
    print(f"\nEsperando a que el servidor este disponible...")
    for i in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                print(f"[OK] Servidor disponible!")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            print(f"Intento {i+1}/{max_attempts}...", end='\r')
            time.sleep(delay)
    return False


def test_health():
    """Prueba el endpoint de salud general."""
    print("\n" + "="*70)
    print("HEALTH CHECK GENERAL")
    print("="*70)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_individual_model(model_name, url_path):
    """Prueba un modelo individual."""
    log_file = f"test_{url_path}.log"
    with open(log_file, "w") as f:
        def log(msg):
            log_with_timestamp(msg, f)
            
        log("\n" + "="*70)
        log(f"MODELO: {model_name.upper()}")
        log("="*70)
        
        # Validar datos de entrada
        errors = validate_features(example_data, url_path)
        if errors:
            log("\n[ERROR] Datos de entrada inválidos:")
            for error in errors:
                log(f"- {error}")
            return
        
        # Health check
        log(f"\n[1] Health Check de {model_name}...")
        health_url = f"{BASE_URL}/{url_path}/health"
        try:
            health_response = requests.get(health_url, timeout=30)
            log(f"Status: {health_response.status_code}")
            log(json.dumps(health_response.json(), indent=2))
        except Exception as e:
            log(f"Error en health check: {str(e)}")
            return
    
        # Predicción
        log(f"\n[2] Prediccion con {model_name}...")
        predict_url = f"{BASE_URL}/{url_path}/predict"
        try:
            log(f"Enviando datos a {predict_url}...")
            log(f"Datos enviados: {json.dumps(example_data, indent=2)}")
            predict_response = requests.post(predict_url, json=example_data, timeout=60)
            log(f"Status: {predict_response.status_code}")
            try:
                result = predict_response.json()
                log(f"Respuesta completa: {json.dumps(result, indent=2)}")
                
                # Validar formato de respuesta
                if result.get("status") != "success":
                    log("[ERROR] La respuesta no indica éxito")
                if "result" not in result:
                    log("[ERROR] Falta el campo 'result' en la respuesta")
                if "prediccion" not in result.get("result", {}):
                    log("[ERROR] Falta el campo 'prediccion' en el resultado")
                if "score" not in result.get("result", {}):
                    log("[ERROR] Falta el campo 'score' en el resultado")
                    
            except:
                log(f"Respuesta raw: {predict_response.text}")
            if predict_response.status_code == 200:
                log("¡Predicción exitosa!")
            else:
                log(f"Error en la predicción: {predict_response.text}")
        except Exception as e:
            log(f"Excepción durante la predicción: {str(e)}")
            log(f"Tipo de error: {type(e)}")
            import traceback
            log(traceback.format_exc())


def test_ensemble():
    """Prueba el modelo ensemble."""
    print("\n" + "="*70)
    print("MODELO ENSEMBLE (PROMEDIO DE MODELOS)")
    print("="*70)
    
    # Health check
    print("\n[1] Health Check del Ensemble...")
    health_url = f"{BASE_URL}/ensemble/health"
    try:
        health_response = requests.get(health_url, timeout=10)
        print(f"Status: {health_response.status_code}")
        print(json.dumps(health_response.json(), indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    # Predicción
    print("\n[2] Prediccion con Ensemble...")
    predict_url = f"{BASE_URL}/ensemble/predict"
    try:
        predict_response = requests.post(predict_url, json=example_data, timeout=30)
        print(f"Status: {predict_response.status_code}")
        if predict_response.status_code == 200:
            result = predict_response.json()
            print(json.dumps(result, indent=2))
            
            # Resumen
            if 'result' in result:
                r = result['result']
                print("\n" + "="*70)
                print("RESUMEN - ENSEMBLE")
                print("="*70)
                print(f"Prediccion Final: {r.get('prediccion_final', 'N/A')}")
                print(f"Score Promedio: {r.get('score_promedio', 'N/A')}")
                print(f"Confianza: {r.get('confianza_final', 'N/A')}")
        else:
            print(f"Error: {predict_response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_judge():
    """Prueba el juez final."""
    print("\n" + "="*70)
    print("JUEZ FINAL (REGRESION LOGISTICA)")
    print("="*70)
    
    # Health check
    print("\n[1] Health Check del Juez...")
    health_url = f"{BASE_URL}/judge/health"
    try:
        health_response = requests.get(health_url, timeout=10)
        print(f"Status: {health_response.status_code}")
        print(json.dumps(health_response.json(), indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    # Predicción
    print("\n[2] Prediccion con el Juez...")
    predict_url = f"{BASE_URL}/judge/predict"
    try:
        predict_response = requests.post(predict_url, json=example_data, timeout=30)
        print(f"Status: {predict_response.status_code}")
        if predict_response.status_code == 200:
            result = predict_response.json()
            print(json.dumps(result, indent=2))
            
            # Resumen
            if 'result' in result:
                r = result['result']
                print("\n" + "="*70)
                print("RESUMEN - JUEZ FINAL")
                print("="*70)
                print(f"Prediccion: {r.get('prediccion', 'N/A')}")
                print(f"Score: {r.get('score', 'N/A')}")
                print(f"Confianza: {r.get('confianza', 'N/A')}")
                print(f"\nScores de Especialistas:")
                if 'specialist_scores' in r:
                    for name, score in r['specialist_scores'].items():
                        print(f"  - {name}: {score}")
        else:
            print(f"Error: {predict_response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


def generate_invalid_data():
    """
    Genera datos inválidos para pruebas.
    """
    invalid_cases = []
    
    # Caso 1: Falta una característica base
    case1 = json.loads(json.dumps(example_data))
    del case1["data"]["koi_duration"]
    invalid_cases.append(("Falta koi_duration", case1))
    
    # Caso 2: Valor no numérico
    case2 = json.loads(json.dumps(example_data))
    case2["data"]["koi_depth"] = "no_numerico"
    invalid_cases.append(("koi_depth no numérico", case2))
    
    # Caso 3: Falta incertidumbre
    case3 = json.loads(json.dumps(example_data))
    del case3["data"]["koi_period_err1"]
    invalid_cases.append(("Falta koi_period_err1", case3))
    
    # Caso 4: Incertidumbre inválida (err1 negativo)
    case4 = json.loads(json.dumps(example_data))
    case4["data"]["koi_period_err1"] = -1.0
    invalid_cases.append(("koi_period_err1 negativo", case4))
    
    return invalid_cases

def test_invalid_data():
    """
    Prueba el manejo de datos inválidos.
    """
    print("\n" + "="*70)
    print("PRUEBAS DE DATOS INVÁLIDOS")
    print("="*70)
    
    invalid_cases = generate_invalid_data()
    
    for case_name, invalid_data in invalid_cases:
        print(f"\nProbando: {case_name}")
        
        # Validar con cada modelo
        for model_type in ["fotometria", "orbital", "estelar", "falsos_positivos"]:
            print(f"\n[{model_type}]")
            errors = validate_features(invalid_data, model_type)
            for error in errors:
                print(f"- {error}")
            
            # Probar endpoint
            url = f"{BASE_URL}/{model_type}/predict"
            try:
                response = requests.post(url, json=invalid_data, timeout=10)
                print(f"Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error esperado: {response.text}")
            except Exception as e:
                print(f"Error: {str(e)}")

def test_all_models():
    """
    Prueba todos los modelos secuencialmente.
    """
    print("=" * 70)
    print("NASA SPACE APPS 2025 - API TEST")
    print("Testing ALL MODELS")
    print("=" * 70)
    
    if not wait_for_server():
        print("\n[ERROR] No se pudo conectar al servidor.")
        return
        
    # Health check general
    test_health()
    
    # Probar cada especialista
    for model in ["fotometria", "orbital", "estelar", "falsos_positivos"]:
        name, path = {
            "fotometria": ("Fotometría", "fotometria"),
            "orbital": ("Orbital", "orbital"), 
            "estelar": ("Estelar", "estelar"),
            "falsos_positivos": ("Falsos Positivos", "falsos-positivos")
        }[model]
        test_individual_model(name, path)
        
    # Probar ensemble y juez
    test_ensemble()
    test_judge()
    
    print("\n" + "="*70)
    print("PRUEBAS COMPLETAS")
    print("="*70)
    
    print("\nArchivos de log generados:")
    print("- test_fotometria.log")
    print("- test_orbital.log")
    print("- test_estelar.log")
    print("- test_falsos-positivos.log")

def test_specific_model(model_name):
    """
    Prueba un modelo específico solamente.
    """
    print("=" * 70)
    print("NASA SPACE APPS 2025 - API TEST")
    print(f"Testing {model_name} only")
    print("=" * 70)
    
    if not wait_for_server():
        print("\n[ERROR] No se pudo conectar al servidor.")
        return

    model_map = {
        "fotometria": ("Fotometría", "fotometria"),
        "orbital": ("Orbital", "orbital"),
        "estelar": ("Estelar", "estelar"),
        "falsos_positivos": ("Falsos Positivos", "falsos-positivos"),
        "ensemble": None,
        "judge": None
    }

    if model_name == "ensemble":
        test_ensemble()
    elif model_name == "judge":
        test_judge()
    elif model_name in model_map:
        name, path = model_map[model_name]
        test_individual_model(name, path)
    else:
        print(f"Modelo {model_name} no reconocido")


if __name__ == "__main__":
    print("=" * 70)
    print("NASA SPACE APPS 2025 - PRUEBAS DEL API")
    print("=" * 70)

    # Comprobar argumentos
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        if model_name == "--help" or model_name == "-h":
            print("\nUso:")
            print("  python test_api.py             # Probar todos los modelos")
            print("  python test_api.py fotometria  # Probar solo el modelo de fotometría")
            print("  python test_api.py orbital     # Probar solo el modelo orbital")
            print("  python test_api.py estelar     # Probar solo el modelo estelar")
            print("  python test_api.py falsos-positivos  # Probar modelo de falsos positivos")
            print("  python test_api.py ensemble    # Probar el modelo ensemble")
            print("  python test_api.py judge       # Probar el juez final")
            sys.exit(0)
        
        print(f"\nProbando modelo: {model_name}")
        test_specific_model(model_name)
    else:
        print("\nProbando TODAS las arquitecturas:")
        print("1. Modelos Especialistas:")
        print("   - Fotometría")
        print("   - Orbital")
        print("   - Estelar")
        print("   - Falsos Positivos")
        print("2. Modelo Ensemble")
        print("3. Juez Final")
        print("\nIniciando pruebas...")
        
        # Probar todas las arquitecturas
        test_all_models()
        
        print("\nProbando casos de error...")
        # Probar casos inválidos
        test_invalid_data()
        
        print("\nRevisa los archivos de log para más detalles:")
