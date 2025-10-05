# Manual del API de Detección de Exoplanetas NASA Space Apps 2025

## Índice
1. [Introducción](#introducción)
2. [Estructura General](#estructura-general)
3. [Endpoints Disponibles](#endpoints-disponibles)
4. [Modelos Especialistas](#modelos-especialistas)
5. [Modelo Ensemble](#modelo-ensemble)
6. [Juez Final](#juez-final)
7. [Ejemplos de Uso](#ejemplos-de-uso)
8. [Manejo de Errores](#manejo-de-errores)

## Introducción

Este API proporciona servicios de detección y clasificación de exoplanetas utilizando un sistema de votación basado en múltiples modelos especializados y un juez final. El sistema está compuesto por:

- 4 modelos especialistas (fotometría, orbital, estelar, falsos positivos)
- 1 modelo ensemble que combina predicciones
- 1 juez final que toma la decisión definitiva

## Estructura General

Todos los endpoints siguen la misma estructura básica:

- **URL Base**: `http://localhost:8000`
- **Formato**: JSON
- **Métodos**: GET (health), POST (predict)
- **Headers**: Content-Type: application/json

## Endpoints Disponibles

### Health Check General
- **GET** `/health`
- **Respuesta**: Status del servidor y versión
```json
{
    "status": "ok",
    "version": "1.0.0"
}
```

## Modelos Especialistas

### 1. Modelo de Fotometría
- **Health Check**: GET `/fotometria/health`
- **Predicción**: POST `/fotometria/predict`

#### Features Requeridos
```json
{
    "data": {
        "koi_duration": 2.9575,        // Duración del tránsito
        "koi_duration_err1": 0.0819,   // Error positivo de duración
        "koi_duration_err2": -0.0819,  // Error negativo de duración
        "koi_depth": 615.8,            // Profundidad del tránsito
        "koi_depth_err1": 19.5,        // Error positivo de profundidad
        "koi_depth_err2": -19.5,       // Error negativo de profundidad
        "koi_impact": 0.146,           // Parámetro de impacto
        "koi_model_snr": 35.8          // Relación señal/ruido
    }
}
```

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion": 1,           // 1: Exoplaneta, 0: No exoplaneta
        "score": 0.95,            // Confianza de la predicción
        "confianza": "alta"       // Interpretación cualitativa
    }
}
```

### 2. Modelo Orbital
- **Health Check**: GET `/orbital/health`
- **Predicción**: POST `/orbital/predict`

#### Features Requeridos
```json
{
    "data": {
        "koi_period": 9.48803557,     // Período orbital
        "koi_period_err1": 0.00002775, // Error positivo del período
        "koi_period_err2": -0.00002775,// Error negativo del período
        "koi_time0bk": 170.53875,     // Tiempo de tránsito
        "koi_time0bk_err1": 0.00216,  // Error positivo del tiempo
        "koi_time0bk_err2": -0.00216  // Error negativo del tiempo
    }
}
```

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion": 1,
        "score": 0.88,
        "confianza": "media"
    }
}
```

### 3. Modelo Estelar
- **Health Check**: GET `/estelar/health`
- **Predicción**: POST `/estelar/predict`

#### Features Requeridos
```json
{
    "data": {
        "koi_srad": 0.927,           // Radio estelar
        "koi_srad_err1": 0.105,      // Error positivo del radio
        "koi_srad_err2": -0.061,     // Error negativo del radio
        "koi_steff": 5455.0,         // Temperatura efectiva
        "koi_steff_err1": 81.0,      // Error positivo de temperatura
        "koi_steff_err2": -81.0,     // Error negativo de temperatura
        "koi_slogg": 4.467,          // Gravedad superficial
        "koi_slogg_err1": 0.064,     // Error positivo de gravedad
        "koi_slogg_err2": -0.096,    // Error negativo de gravedad
        "koi_prad": 2.26,            // Radio planetario
        "koi_prad_err1": 0.26,       // Error positivo del radio planetario
        "koi_prad_err2": -0.15,      // Error negativo del radio planetario
        "koi_insol": 93.59,          // Insolación
        "koi_insol_err1": 29.45,     // Error positivo de insolación
        "koi_insol_err2": -16.65,    // Error negativo de insolación
        "koi_teq": 793.0,            // Temperatura de equilibrio
        "koi_kepmag": 15.347         // Magnitud Kepler
    }
}
```

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion": 1,
        "score": 0.92,
        "confianza": "alta"
    }
}
```

### 4. Modelo de Falsos Positivos
- **Health Check**: GET `/falsos-positivos/health`
- **Predicción**: POST `/falsos-positivos/predict`

#### Features Requeridos
```json
{
    "data": {
        "koi_fpflag_nt": 0,  // Flag de not transit-like
        "koi_fpflag_ss": 0,  // Flag de stellar eclipse
        "koi_fpflag_co": 0,  // Flag de centroid offset
        "koi_fpflag_ec": 0   // Flag de ephemeris match
    }
}
```

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion": 1,
        "score": 0.97,
        "confianza": "alta"
    }
}
```

## Modelo Ensemble
- **Health Check**: GET `/ensemble/health`
- **Predicción**: POST `/ensemble/predict`

El modelo ensemble acepta los mismos features que los modelos individuales y combina sus predicciones.

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion_final": 1,
        "score_promedio": 0.93,
        "confianza_final": "alta",
        "predicciones": {
            "fotometria": {"prediccion": 1, "score": 0.95},
            "orbital": {"prediccion": 1, "score": 0.88},
            "estelar": {"prediccion": 1, "score": 0.92},
            "falsos_positivos": {"prediccion": 1, "score": 0.97}
        }
    }
}
```

## Juez Final
- **Health Check**: GET `/judge/health`
- **Predicción**: POST `/judge/predict`

El juez final utiliza los scores de todos los especialistas para tomar una decisión final.

#### Respuesta Esperada
```json
{
    "status": "success",
    "result": {
        "prediccion": 1,
        "score": 0.94,
        "confianza": "alta",
        "specialist_scores": {
            "fotometria": 0.95,
            "orbital": 0.88,
            "estelar": 0.92,
            "falsos_positivos": 0.97
        }
    }
}
```

## Manejo de Errores

Los errores siguen un formato consistente:

```json
{
    "status": "error",
    "error": {
        "type": "ValidationError",
        "message": "Falta la característica requerida: koi_duration"
    }
}
```

### Códigos de Estado HTTP
- 200: Éxito
- 400: Error de validación
- 500: Error interno del servidor

### Tipos de Error Comunes
1. ValidationError: Datos de entrada inválidos
2. MissingFeatureError: Falta una característica requerida
3. InvalidValueError: Valor fuera de rango o tipo incorrecto

## Ejemplos de Uso

### Python
```python
import requests

# URL base
BASE_URL = "http://localhost:8000"

# Datos de ejemplo
data = {
    "data": {
        "koi_duration": 2.9575,
        "koi_duration_err1": 0.0819,
        "koi_duration_err2": -0.0819,
        # ... (resto de features)
    }
}

# Llamada al API
response = requests.post(f"{BASE_URL}/fotometria/predict", json=data)
result = response.json()

print(f"Predicción: {result['result']['prediccion']}")
print(f"Score: {result['result']['score']}")
print(f"Confianza: {result['result']['confianza']}")
```

### cURL
```bash
curl -X POST http://localhost:8000/fotometria/predict \
     -H "Content-Type: application/json" \
     -d @example_input.json
```

Para probar todos los endpoints y validar su funcionamiento, puedes usar el script `test_api.py` incluido:

```bash
# Probar todos los modelos
python test_api.py

# Probar un modelo específico
python test_api.py fotometria
```