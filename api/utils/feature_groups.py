# api/utils/feature_groups.py

"""
Define los grupos de características para cada modelo especialista.
"""

FOTOMETRIA_FEATURES = [
    "koi_depth",
    "koi_depth_incertidumbre_promedio",
    "koi_depth_incertidumbre_maxima",
    "koi_duration",
    "koi_duration_incertidumbre_promedio",
    "koi_duration_incertidumbre_maxima",
    "koi_ingress",
    "koi_model_snr"
]

ORBITAL_FEATURES = [
    "koi_period",
    "koi_period_incertidumbre_promedio",
    "koi_period_incertidumbre_maxima",
    "koi_time0bk",
    "koi_time0bk_incertidumbre_promedio",
    "koi_time0bk_incertidumbre_maxima",
    "koi_eccen",
    "koi_longp"
]

ESTELAR_FEATURES = [
    "koi_srad",
    "koi_srad_incertidumbre_promedio",
    "koi_srad_incertidumbre_maxima",
    "koi_steff",
    "koi_steff_incertidumbre_promedio",
    "koi_steff_incertidumbre_maxima",
    "koi_slogg",
    "koi_slogg_incertidumbre_promedio",
    "koi_slogg_incertidumbre_maxima",
    "ra",
    "dec"
]

FALSOS_POSITIVOS_FEATURES = [
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_score"
]


def get_feature_group(model_type: str) -> list:
    """
    Retorna el grupo de características para un tipo de modelo específico.
    
    Args:
        model_type: Tipo de modelo ('fotometria', 'orbital', 'estelar', 'falsos_positivos')
        
    Returns:
        Lista de nombres de características
    """
    feature_map = {
        'fotometria': FOTOMETRIA_FEATURES,
        'orbital': ORBITAL_FEATURES,
        'estelar': ESTELAR_FEATURES,
        'falsos_positivos': FALSOS_POSITIVOS_FEATURES
    }
    
    return feature_map.get(model_type, [])
