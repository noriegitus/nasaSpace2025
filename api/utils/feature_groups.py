# api/utils/feature_groups.py

"""
Define los grupos de características para cada modelo especialista.
"""

FOTOMETRIA_FEATURES = [
            'koi_duration', 'koi_duration_sigma', 'koi_duration_snr', 'koi_duration_rel_unc',
            'koi_depth', 'koi_depth_sigma', 'koi_depth_snr', 'koi_depth_rel_unc',
            'koi_impact',
            'koi_model_snr'
        ]

ORBITAL_FEATURES = [
            'koi_period', 'koi_period_sigma', 'koi_period_snr', 'koi_period_rel_unc',
            'koi_time0bk', 'koi_time0bk_sigma', 'koi_time0bk_snr', 'koi_time0bk_rel_unc'
        ]

ESTELAR_FEATURES = [
            'koi_srad', 'koi_srad_sigma', 'koi_srad_snr', 'koi_srad_rel_unc',
            'koi_steff', 'koi_steff_sigma', 'koi_steff_snr', 'koi_steff_rel_unc',
            'koi_slogg', 'koi_slogg_sigma', 'koi_slogg_snr', 'koi_slogg_rel_unc',
            'koi_prad', 'koi_prad_sigma', 'koi_prad_snr', 'koi_prad_rel_unc',
            'koi_insol', 'koi_insol_sigma', 'koi_insol_snr', 'koi_insol_rel_unc',
            'koi_teq',
            'koi_kepmag'
        ]

FALSOS_POSITIVOS_FEATURES = [
            'koi_fpflag_nt',
            'koi_fpflag_ss',
            'koi_fpflag_co',
            'koi_fpflag_ec'
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
