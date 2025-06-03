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
- fe_neuroimaging.py: Features de neuroimagen (MRI/PET)
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
from .fe_neuroimaging import NeuroimagingFeatureEngineering
from .fe_biomarkers import BiomarkersFeatureEngineering
from .fe_clinical import ClinicalFeatureEngineering
from .fe_synthetic_activity_sleep import ActivitySleepFeatureEngineering
from .feature_engineering_pipeline import FeatureEngineeringPipeline

# Lista de todas las clases exportadas
__all__ = [
    'DemographicsFeatureEngineering',
    'GeneticsFeatureEngineering', 
    'NeuroimagingFeatureEngineering',
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
        'cognitive': 0.3,
        'biomarker': 0.25,
        'neuroimaging': 0.2,
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
    'neuroimaging': {
        'description': 'Medidas de neuroimagen estructural y funcional',
        'expected_features': ['brain_volume', 'cortical_thickness', 'pet_uptake'],
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
        'fe_neuroimaging',
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

# Mensaje de inicialización
print(f"📦 Feature Engineering Module v{__version__} cargado exitosamente")
print(f"🔬 Modalidades disponibles: {', '.join(get_all_modalities())}")
print(f"⚙️  Configuración por defecto aplicada")