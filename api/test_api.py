# api/test_api.py
"""
Script adaptado para probar el API de detección de exoplanetas,
utilizando el formato de features procesadas que esperan los modelos.
"""

import requests
import json
import math

# URL base del API
BASE_URL = "http://localhost:8000"

# --------------------------------------------------------------------------
# --- DATOS DE EJEMPLO ADAPTADOS ---
# Estos datos ahora reflejan el formato "procesado" que tus modelos esperan.
# Se han calculado las features _sigma, _snr y _rel_unc a partir de los
# errores, y se han añadido las columnas que faltaban.
# --------------------------------------------------------------------------
example_data = {
    "data": {
        # --- Features de Fotometría ---
        "koi_duration": 2.6,
        "koi_duration_sigma": 0.1,
        "koi_duration_snr": 26.0,
        "koi_duration_rel_unc": 0.03846,
        "koi_depth": 615.8,
        "koi_depth_sigma": 15.2,
        "koi_depth_snr": 40.51,
        "koi_depth_rel_unc": 0.02468,
        "koi_impact": 0.15,          # Añadido (valor típico)
        "koi_model_snr": 35.5,

        # --- Features Orbitales ---
        "koi_period": 3.5226,
        "koi_period_sigma": 0.0001,
        "koi_period_snr": 35226.0,
        "koi_period_rel_unc": 0.000028,
        "koi_time0bk": 170.538,
        "koi_time0bk_sigma": 0.004,
        "koi_time0bk_snr": 42634.5,
        "koi_time0bk_rel_unc": 0.000023,

        # --- Features Estelares ---
        "koi_srad": 1.046,
        "koi_srad_sigma": 0.044,
        "koi_srad_snr": 23.77,
        "koi_srad_rel_unc": 0.04206,
        "koi_steff": 5853.0,
        "koi_steff_sigma": 81.0,
        "koi_steff_snr": 72.25,
        "koi_steff_rel_unc": 0.01384,
        "koi_slogg": 4.467,
        "koi_slogg_sigma": 0.064,
        "koi_slogg_snr": 69.79,
        "koi_slogg_rel_unc": 0.01432,
        "koi_prad": 2.26,
        "koi_prad_sigma": 0.1,
        "koi_prad_snr": 22.6,
        "koi_prad_rel_unc": 0.04424,
        "koi_insol": 205.0,
        "koi_insol_sigma": 18.0,
        "koi_insol_snr": 11.38,
        "koi_insol_rel_unc": 0.0878,
        "koi_teq": 1500,              # Añadido (valor típico)
        "koi_kepmag": 15.3,           # Añadido (valor típico)

        # --- Features de Falsos Positivos ---
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0
    }
}


def test_health():
    """Prueba el endpoint de salud general."""
    print("\n=== Probando Health Check General ===")
    response = requests.get(f"{BASE_URL}/health")
    response.raise_for_status() # Lanza un error si el status no es 2xx
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_individual_model(model_name):
    """Prueba un modelo individual."""
    print(f"\n=== Probando Modelo: {model_name} ===")
    
    # Health check
    health_url = f"{BASE_URL}/{model_name}/health"
    health_response = requests.get(health_url)
    health_response.raise_for_status()
    print(f"Health Status: {health_response.status_code}")
    print(json.dumps(health_response.json(), indent=2))
    
    # Predicción
    predict_url = f"{BASE_URL}/{model_name}/predict"
    predict_response = requests.post(predict_url, json=example_data)
    predict_response.raise_for_status()
    print(f"\nPredicción Status: {predict_response.status_code}")
    print(json.dumps(predict_response.json(), indent=2))


def test_ensemble():
    """Prueba el modelo ensemble."""
    print("\n=== Probando Modelo Ensemble ===")
    
    # Health check
    health_url = f"{BASE_URL}/ensemble/health"
    health_response = requests.get(health_url)
    health_response.raise_for_status()
    print(f"Health Status: {health_response.status_code}")
    print(json.dumps(health_response.json(), indent=2))
    
    # Predicción
    predict_url = f"{BASE_URL}/ensemble/predict"
    predict_response = requests.post(predict_url, json=example_data)
    predict_response.raise_for_status()
    print(f"\nPredicción Status: {predict_response.status_code}")
    print(json.dumps(predict_response.json(), indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("NASA Space Apps 2025 - API Test")
    print("=" * 60)
    
    try:
        # Probar health check general
        test_health()
        
        # Probar modelos individuales
        test_individual_model("fotometria")
        test_individual_model("orbital")
        test_individual_model("estelar")
        test_individual_model("falsos_positivos") # Asegúrate que el endpoint sea este
        
        # Probar ensemble
        test_ensemble()
        
        print("\n" + "=" * 60)
        print("✅ Pruebas completadas con éxito!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al servidor.")
        print(f"Asegúrate de que el servidor FastAPI esté corriendo en {BASE_URL}")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Error HTTP: {e.response.status_code} - {e.response.reason}")
        try:
            print("Detalles:", json.dumps(e.response.json(), indent=2))
        except json.JSONDecodeError:
            print("Respuesta del servidor:", e.response.text)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")