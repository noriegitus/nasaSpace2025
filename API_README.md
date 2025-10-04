# NASA Space Apps 2025 - Exoplanet Detection API

API backend para detección de exoplanetas candidatos usando modelos especialistas de deep learning.

## Estructura del Proyecto

```
nasaSpace2025/
├── api/
│   ├── main.py                 # Punto de entrada FastAPI
│   ├── routes/                 # Rutas por módulo
│   │   ├── fotometria.py
│   │   ├── orbital.py
│   │   ├── estelar.py
│   │   ├── falsos_positivos.py
│   │   └── ensemble.py
│   ├── services/               # Lógica de carga de modelos y predicción
│   │   ├── fotometria_service.py
│   │   ├── orbital_service.py
│   │   ├── estelar_service.py
│   │   ├── falsos_positivos_service.py
│   │   └── ensemble_service.py
│   └── utils/                  # Utilidades comunes
│       ├── preprocessing.py
│       └── feature_groups.py
├── model/                      # Arquitecturas de modelos
├── outputs/                    # Pesos entrenados
└── data/                       # Datos y artefactos
```

## Instalación

1. Activar el ambiente conda:
```bash
conda activate nasa2025
```

2. Instalar dependencias adicionales del API:
```bash
pip install -r requirements-api.txt
```

## Ejecución

### Modo Desarrollo
```bash
cd api
python main.py
```

### Modo Producción
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Modelos Individuales

- **POST** `/fotometria/predict` - Predicción con modelo de fotometría
- **POST** `/orbital/predict` - Predicción con modelo orbital
- **POST** `/estelar/predict` - Predicción con modelo de propiedades estelares
- **POST** `/falsos-positivos/predict` - Predicción con modelo de detección de falsos positivos

### Modelo Ensemble

- **POST** `/ensemble/predict` - Predicción combinada de todos los modelos

### Health Checks

- **GET** `/health` - Estado general del API
- **GET** `/fotometria/health` - Estado del servicio de fotometría
- **GET** `/orbital/health` - Estado del servicio orbital
- **GET** `/estelar/health` - Estado del servicio estelar
- **GET** `/falsos-positivos/health` - Estado del servicio de falsos positivos
- **GET** `/ensemble/health` - Estado del servicio ensemble

## Formato de Solicitud

```json
{
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
```

## Formato de Respuesta

### Modelo Individual
```json
{
  "status": "success",
  "result": {
    "modelo": "fotometria",
    "score": 0.8532,
    "prediccion": "CONFIRMED",
    "confianza": 0.7064
  }
}
```

### Modelo Ensemble
```json
{
  "status": "success",
  "result": {
    "prediccion_final": "CONFIRMED",
    "score_promedio": 0.8234,
    "confianza_final": 0.6468,
    "votos": {
      "CONFIRMED": 3,
      "FALSE_POSITIVE": 1
    },
    "modelos_individuales": {
      "fotometria": {
        "modelo": "fotometria",
        "score": 0.8532,
        "prediccion": "CONFIRMED",
        "confianza": 0.7064
      },
      "orbital": {
        "modelo": "orbital",
        "score": 0.7821,
        "prediccion": "CONFIRMED",
        "confianza": 0.5642
      },
      "estelar": {
        "modelo": "estelar",
        "score": 0.9102,
        "prediccion": "CONFIRMED",
        "confianza": 0.8204
      },
      "falsos_positivos": {
        "modelo": "falsos_positivos",
        "score": 0.2481,
        "prediccion": "CONFIRMED",
        "confianza": 0.5038
      }
    }
  }
}
```

## Documentación Interactiva

Una vez ejecutado el servidor, accede a:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Arquitectura de los Modelos

### 1. Modelo de Fotometría
Analiza las características de fotometría del tránsito (profundidad, duración, SNR).

### 2. Modelo Orbital
Evalúa la periodicidad y regularidad de la órbita.

### 3. Modelo de Propiedades Estelares
Determina si el sistema es físicamente plausible basado en las características de la estrella anfitriona.

### 4. Modelo de Detección de Falsos Positivos
Identifica firmas conocidas que indican falsos positivos de origen instrumental.

### 5. Ensemble
Combina las predicciones de los 4 modelos especialistas para generar una predicción final más robusta.
