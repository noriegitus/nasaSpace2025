# api/test_api.py
"""
Script de ejemplo para probar el API de detección de exoplanetas.
"""

import requests
import json

# URL base del API
BASE_URL = "http://localhost:8000"

# Datos de ejemplo de un exoplaneta candidato
example_data = {
    "data": {
        "koi_period": 3.5226,
        "koi_period_err1": 0.0001,
        "koi_period_err2": -0.0001,
        "koi_time0bk": 170.538,
        "koi_time0bk_err1": 0.004,
        "koi_time0bk_err2": -0.004,
        "koi_duration": 2.6,
        "koi_duration_err1": 0.1,
        "koi_duration_err2": -0.1,
        "koi_depth": 615.8,
        "koi_depth_err1": 15.2,
        "koi_depth_err2": -15.2,
        "koi_ror": 0.0248,
        "koi_ror_err1": 0.0003,
        "koi_ror_err2": -0.0003,
        "koi_srad": 1.046,
        "koi_srad_err1": 0.044,
        "koi_srad_err2": -0.044,
        "koi_steff": 5853,
        "koi_steff_err1": 81,
        "koi_steff_err2": -81,
        "koi_slogg": 4.467,
        "koi_slogg_err1": 0.064,
        "koi_slogg_err2": -0.064,
        "koi_prad": 2.26,
        "koi_prad_err1": 0.1,
        "koi_prad_err2": -0.1,
        "koi_insol": 205,
        "koi_insol_err1": 18,
        "koi_insol_err2": -18,
        "koi_model_snr": 35.5,
        "koi_ingress": 0.2,
        "koi_eccen": 0,
        "koi_longp": 90,
        "ra": 291.93423,
        "dec": 48.141651,
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0,
        "koi_score": 0.969
    }
}


def test_health():
    """Prueba el endpoint de salud general."""
    print("\n=== Probando Health Check General ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_individual_model(model_name):
    """Prueba un modelo individual."""
    print(f"\n=== Probando Modelo: {model_name} ===")
    
    # Health check
    health_url = f"{BASE_URL}/{model_name}/health"
    health_response = requests.get(health_url)
    print(f"Health Status: {health_response.status_code}")
    print(json.dumps(health_response.json(), indent=2))
    
    # Predicción
    predict_url = f"{BASE_URL}/{model_name}/predict"
    predict_response = requests.post(predict_url, json=example_data)
    print(f"\nPredicción Status: {predict_response.status_code}")
    print(json.dumps(predict_response.json(), indent=2))


def test_ensemble():
    """Prueba el modelo ensemble."""
    print("\n=== Probando Modelo Ensemble ===")
    
    # Health check
    health_url = f"{BASE_URL}/ensemble/health"
    health_response = requests.get(health_url)
    print(f"Health Status: {health_response.status_code}")
    print(json.dumps(health_response.json(), indent=2))
    
    # Predicción
    predict_url = f"{BASE_URL}/ensemble/predict"
    predict_response = requests.post(predict_url, json=example_data)
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
        test_individual_model("falsos-positivos")
        
        # Probar ensemble
        test_ensemble()
        
        print("\n" + "=" * 60)
        print("Pruebas completadas!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al servidor.")
        print("Asegúrate de que el servidor esté corriendo en http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
