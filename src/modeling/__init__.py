"""
Módulo de Modelado para Monitorización Multimodal de Alzheimer
============================================================

Este módulo contiene pipelines y utilidades para el desarrollo de modelos
de predicción de riesgo de Alzheimer usando datos multimodales.

Componentes principales:
- regression_pipeline: Pipeline para predicción de score de riesgo continuo
- classification_pipeline: Pipeline para clasificación de categorías de riesgo
- temporal_modeling: Modelos para análisis de series temporales
- risk_stratification: Algoritmos de estratificación de riesgo
- ensemble_methods: Métodos de ensemble y combinación de modelos
- model_utils: Utilidades comunes y funciones auxiliares

Autor: Abraham Tartalos
Fecha: 2025
"""

from .regression_pipeline import RegressionPipeline
from .classification_pipeline import ClassificationPipeline
from .temporal_modeling import TemporalModeling
from .risk_stratification import RiskStratification
from .ensemble_methods import EnsembleMethods
from .model_utils import (
    load_processed_data,
    prepare_features,
    evaluate_model,
    log_model_metrics,
    create_model_comparison,
    save_model_artifacts
)

__version__ = "1.0.0"
__author__ = "Abraham Tartalos"

__all__ = [
    'RegressionPipeline',
    'ClassificationPipeline', 
    'TemporalModeling',
    'RiskStratification',
    'EnsembleMethods',
    'load_processed_data',
    'prepare_features',
    'evaluate_model',
    'log_model_metrics',
    'create_model_comparison',
    'save_model_artifacts'
]