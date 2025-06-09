"""
Feature Engineering Module for Multimodal Alzheimer Monitoring
==============================================================

Módulo de ingeniería de características para el proyecto de 
Monitorización Multimodal de Alzheimer.

Autor: Abraham Tartalos
Fecha: Mayo 2025
Fase: 3 - Feature Engineering y Selección

Estructura del módulo:
- fe_demographics.py: Features demográficas y socioeconómicas
- fe_genetics.py: Features genéticas y APOE
- fe_mri.py: Features de neuroimagen estructural (MRI)
- fe_pet.py: Features de neuroimagen funcional (PET)
- fe_biomarkers.py: Features de biomarcadores en fluidos
- fe_clinical.py: Features clínicas y cognitivas
- fe_synthetic_activity_sleep.py: Features de actividad y sueño
- feature_engineering_pipeline.py: Pipeline maestro
"""

__version__ = "1.0.0"
__author__ = "Abraham Tartalos"
__description__ = "Feature engineering multimodal para detección de Alzheimer"
#__email__ = "alzheimer.monitoring@research.com"

# Importar clases principales
from .fe_demographics import DemographicsFeatureEngineering
from .fe_genetics import GeneticsFeatureEngineering
from .fe_mri import NeuroImagingFeatureEngineer  # CAMBIO: fe_neuroimaging -> fe_mri
from .fe_pet import PETFeatureEngineer  # NUEVO: agregado fe_pet
from .fe_biomarkers import BiomarkersFeatureEngineering
from .fe_clinical import ClinicalFeatureEngineering
from .fe_synthetic_activity_sleep import ActivitySleepFeatureEngineering
from .feature_engineering_pipeline import FeatureEngineeringPipeline

# Lista de todas las clases exportadas
__all__ = [
    'DemographicsFeatureEngineering',
    'GeneticsFeatureEngineering', 
    'NeuroImagingFeatureEngineer',  # CAMBIO: mantener nombre original de la clase
    'PETFeatureEngineer',  # NUEVO: agregada clase PET
    'BiomarkersFeatureEngineering',
    'ClinicalFeatureEngineering',
    'ActivitySleepFeatureEngineering',
    'FeatureEngineeringPipeline'
]

# Configuración por defecto para el módulo
DEFAULT_CONFIG = {
    'missing_threshold': 0.8,
    'correlation_threshold': 0.95,
    'temporal_window_days': 365,
    'feature_selection_methods': ['univariate', 'recursive', 'clinical_relevance'],
    'risk_score_components': {
        'cognitive': 0.25,  # CAMBIO: reducido de 0.3
        'biomarker': 0.25,
        'mri': 0.15,  # CAMBIO: neuroimaging dividido en MRI y PET
        'pet': 0.10,  # NUEVO: componente PET
        'genetic': 0.15,
        'activity_sleep': 0.1
    },
    'outlier_detection': {
        'method': 'iqr',
        'threshold': 3.0
    },
    'normalization': {
        'method': 'robust_scaler',
        'handle_missing': 'median'
    }
}

# Metadatos de modalidades
MODALITY_INFO = {
    'demographics': {
        'description': 'Características demográficas y socioeconómicas',
        'expected_features': ['age', 'gender', 'education', 'socioeconomic_status'],
        'clinical_relevance': 'high'
    },
    'genetics': {
        'description': 'Información genética y polimorfismos',
        'expected_features': ['APOE4_status', 'genetic_risk_variants'],
        'clinical_relevance': 'high'
    },
    'mri': {  # CAMBIO: neuroimaging -> mri
        'description': 'Medidas de neuroimagen estructural (MRI)',
        'expected_features': ['brain_volume', 'cortical_thickness', 'hippocampal_volume'],
        'clinical_relevance': 'very_high'
    },
    'pet': {  # NUEVO: modalidad PET separada
        'description': 'Medidas de neuroimagen funcional (PET)',
        'expected_features': ['amyloid_uptake', 'tau_uptake', 'glucose_metabolism'],
        'clinical_relevance': 'very_high'
    },
    'biomarkers': {
        'description': 'Biomarcadores en fluidos corporales',
        'expected_features': ['abeta42', 'tau', 'ptau181'],
        'clinical_relevance': 'very_high'
    },
    'clinical': {
        'description': 'Evaluaciones clínicas y cognitivas',
        'expected_features': ['mmse', 'cdr', 'cognitive_assessments'],
        'clinical_relevance': 'very_high'
    },
    'activity_sleep': {
        'description': 'Patrones de actividad física y sueño',
        'expected_features': ['sleep_quality', 'physical_activity', 'circadian_rhythm'],
        'clinical_relevance': 'medium'
    }
}

# Funciones de utilidad
def get_modality_info(modality: str) -> dict:
    """
    Obtener información sobre una modalidad específica
    
    Args:
        modality: Nombre de la modalidad
        
    Returns:
        Diccionario con información de la modalidad
    """
    return MODALITY_INFO.get(modality, {})

def get_all_modalities() -> list:
    """
    Obtener lista de todas las modalidades disponibles
    
    Returns:
        Lista con nombres de modalidades
    """
    return list(MODALITY_INFO.keys())

def get_neuroimaging_modalities() -> list:
    """
    NUEVA FUNCIÓN: Obtener modalidades de neuroimagen específicamente
    
    Returns:
        Lista con modalidades de neuroimagen ['mri', 'pet']
    """
    return ['mri', 'pet']

def validate_config(config: dict) -> bool:
    """
    Validar configuración del pipeline
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        True si la configuración es válida
    """
    required_keys = ['missing_threshold', 'correlation_threshold', 'risk_score_components']
    
    for key in required_keys:
        if key not in config:
            return False
            
    # Validar rangos de valores
    if not (0 <= config['missing_threshold'] <= 1):
        return False
        
    if not (0 <= config['correlation_threshold'] <= 1):
        return False
        
    # Validar componentes del score de riesgo
    if 'risk_score_components' in config:
        components = config['risk_score_components']
        if not isinstance(components, dict):
            return False
            
        total_weight = sum(components.values())
        if not (0.95 <= total_weight <= 1.05):  # Tolerancia para suma = 1
            return False
            
    return True

def create_feature_summary(df, modality: str) -> dict:
    """
    Crear resumen de features para una modalidad
    
    Args:
        df: DataFrame con features
        modality: Nombre de la modalidad
        
    Returns:
        Diccionario con resumen estadístico
    """
    if df is None or df.empty:
        return {'error': 'DataFrame vacío o None'}
        
    summary = {
        'modality': modality,
        'total_features': len(df.columns),
        'total_records': len(df),
        'missing_percentage': df.isnull().mean().mean() * 100,
        'numeric_features': len(df.select_dtypes(include=['number']).columns),
        'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    return summary

def create_neuroimaging_combined_summary(mri_df=None, pet_df=None) -> dict:
    """
    NUEVA FUNCIÓN: Crear resumen combinado de modalidades de neuroimagen
    
    Args:
        mri_df: DataFrame con features de MRI
        pet_df: DataFrame con features de PET
        
    Returns:
        Diccionario con resumen combinado
    """
    summary = {
        'modalities_available': [],
        'total_features': 0,
        'complementary_coverage': 0.0
    }
    
    if mri_df is not None and not mri_df.empty:
        summary['modalities_available'].append('mri')
        summary['total_features'] += len(mri_df.columns)
        summary['mri_features'] = len(mri_df.columns)
    
    if pet_df is not None and not pet_df.empty:
        summary['modalities_available'].append('pet')
        summary['total_features'] += len(pet_df.columns)
        summary['pet_features'] = len(pet_df.columns)
    
    # Calcular cobertura complementaria si ambas modalidades están disponibles
    if len(summary['modalities_available']) == 2:
        # Esto sería calculado con datos reales
        summary['complementary_coverage'] = 0.85  # Placeholder
        summary['neuroimaging_completeness'] = 'high'
    elif len(summary['modalities_available']) == 1:
        summary['complementary_coverage'] = 0.60
        summary['neuroimaging_completeness'] = 'partial'
    else:
        summary['complementary_coverage'] = 0.0
        summary['neuroimaging_completeness'] = 'none'
    
    return summary

# Logging configuration para el módulo
import logging

def setup_module_logging(log_level: str = 'INFO'):
    """
    Configurar logging para el módulo de feature engineering
    
    Args:
        log_level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_engineering.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configurar loggers específicos para cada modalidad
    loggers = [
        'fe_demographics',
        'fe_genetics', 
        'fe_mri',  # CAMBIO: fe_neuroimaging -> fe_mri
        'fe_pet',  # NUEVO: logger para PET
        'fe_biomarkers',
        'fe_clinical',
        'fe_synthetic_activity_sleep',
        'feature_engineering_pipeline'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

# Constantes útiles
CLINICAL_THRESHOLDS = {
    'mmse_mild_impairment': 24,
    'mmse_moderate_impairment': 18,
    'mmse_severe_impairment': 10,
    'cdr_normal': 0,
    'cdr_questionable': 0.5,
    'cdr_mild': 1,
    'age_elderly': 65,
    'education_low': 8,
    'education_high': 16
}

BIOMARKER_REFERENCE_VALUES = {
    'abeta42_cutoff': 192,  # pg/mL
    'tau_elevated': 300,    # pg/mL
    'ptau181_elevated': 23  # pg/mL
}

# NUEVAS CONSTANTES: Referencias de neuroimagen
MRI_REFERENCE_VALUES = {
    'hippocampal_volume_atrophy': 0.15,  # Porcentaje de atrofia
    'cortical_thickness_thinning': 0.10,  # mm de adelgazamiento
    'ventricular_enlargement': 1.5  # Factor de agrandamiento
}

PET_REFERENCE_VALUES = {
    'amyloid_suvr_positive': 1.11,  # SUVR cutoff para positividad amiloide
    'tau_suvr_elevated': 1.25,     # SUVR cutoff para tau elevado
    'fdg_hypometabolism': 0.85     # Ratio de hipometabolismo FDG
}

# NUEVA FUNCIÓN UTILITARIA: Validación de modalidades de neuroimagen
def validate_neuroimaging_compatibility(mri_features: list = None, pet_features: list = None) -> dict:
    """
    Validar compatibilidad entre features de MRI y PET
    
    Args:
        mri_features: Lista de features de MRI disponibles
        pet_features: Lista de features de PET disponibles
        
    Returns:
        Diccionario con información de compatibilidad
    """
    compatibility = {
        'compatible': True,
        'warnings': [],
        'recommendations': []
    }
    
    if mri_features is None and pet_features is None:
        compatibility['compatible'] = False
        compatibility['warnings'].append('No hay features de neuroimagen disponibles')
        return compatibility
    
    if mri_features is not None and len(mri_features) < 3:
        compatibility['warnings'].append('Pocas features de MRI disponibles (<3)')
        compatibility['recommendations'].append('Considerar agregar más medidas estructurales')
    
    if pet_features is not None and len(pet_features) < 2:
        compatibility['warnings'].append('Pocas features de PET disponibles (<2)')
        compatibility['recommendations'].append('Considerar agregar más trazadores PET')
    
    # Verificar complementariedad
    if mri_features and pet_features:
        compatibility['recommendations'].append('Excelente: ambas modalidades disponibles para análisis multimodal')
    elif mri_features and not pet_features:
        compatibility['recommendations'].append('Considerar agregar PET para análisis funcional complementario')
    elif pet_features and not mri_features:
        compatibility['recommendations'].append('Considerar agregar MRI para contexto estructural')
    
    return compatibility

# Mensaje de inicialización actualizado
print(f"📦 Feature Engineering Module v{__version__} cargado exitosamente")
print(f"🔬 Modalidades disponibles: {', '.join(get_all_modalities())}")
print(f"🧠 Neuroimagen: {', '.join(get_neuroimaging_modalities())} (separadas para análisis especializado)")
print(f"⚙️  Configuración por defecto aplicada")