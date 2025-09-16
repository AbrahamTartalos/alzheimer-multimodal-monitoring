"""
Dashboard Interactivo de Monitorización y Prevención de Alzheimer
Proyecto: Monitorización y Predicción Multimodal de Alzheimer - Fase 6
Autor: Abraham Tartalos
Fecha: 2025

Dashboard accesible para usuarios médicos y no médicos con explicaciones claras,
visualizaciones intuitivas y recomendaciones accionables.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
# Nuevos imports para integración
from pathlib import Path
import warnings
import re
import os
warnings.filterwarnings('ignore')

# Configuración de rutas
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features"
REPORTS_PATH = PROJECT_ROOT / "reports" / "evaluation"

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# MLflow opcional (controlado por la variable de entorno USE_MLFLOW)
# -------------------------
MLFLOW_ENABLED = os.environ.get("USE_MLFLOW", "true").lower() not in ("0", "false", "no")
mlflow = None
mlflow_available = False

if MLFLOW_ENABLED:
    try:
        import mlflow
        import mlflow.pyfunc
        mlflow_available = True
        logger.info("MLflow importado correctamente.")
    except Exception as e:
        # Si mlflow no está instalado, lo registramos y continuamos sin MLflow.
        logger.warning(f"mlflow no disponible en el entorno: {e}. Procediendo sin MLflow.")
else:
    logger.info("USE_MLFLOW=false -> MLflow deshabilitado por variable de entorno.")


# ==========================================
# CONFIGURACIÓN Y DATOS SIMULADOS
# ==========================================

class AlzheimerDashboardConfig:
    """Configuración centralizada del dashboard con nueva paleta médica profesional"""
    
    def __init__(self):
        # NUEVA PALETA MÉDICA PROFESIONAL
        self.colors = {
            # Colores principales
            'primary': '#2563eb',           # Azul médico
            'secondary': '#0f766e',         # Teal oscuro
            'text': '#374151',              # Gris carbón
            
            # Colores funcionales de riesgo
            'low_risk': '#059669',          # Verde esmeralda (antes #10B981)
            'moderate_risk': '#d97706',     # Ámbar (antes #F59E0B)
            'high_risk': '#dc2626',         # Rojo coral (antes #EF4444)
            
            # Colores especiales
            'neuro': '#7c3aed',             # Púrpura suave (datos neurológicos)
            'info': '#2563eb',              # Azul médico (mismo que primary)
            
            # Fondos
            'background': '#f8fafc',        # Gris muy claro (antes #F9FAFB)
            'card': '#ffffff',              # Blanco
            'data_bg': '#eff6ff',           # Azul muy claro (áreas importantes)
            
            # Utilidades
            'border': '#e2e8f0',            # Bordes suaves
            'muted': '#64748b',             # Texto secundario
            'disabled': '#cbd5e1'           # Elementos deshabilitados
        }
        
        # Mantener thresholds existentes
        self.thresholds = {
            'low_risk': 0.3,
            'moderate_risk': 0.7,
            'high_risk': 1.0
        }
        
        # Color maps específicos para gráficos
        self.color_maps = {
            'risk_categories': {
                'Bajo': self.colors['low_risk'],
                'Moderado': self.colors['moderate_risk'],
                'Alto': self.colors['high_risk']
            },
            'feature_groups': {
                'biomarcadores': self.colors['high_risk'],     # Rojo coral para biomarcadores críticos
                'cognitivo': self.colors['moderate_risk'],     # Ámbar para evaluaciones cognitivas
                'demografico': self.colors['info'],            # Azul médico para demográficos
                'neuroimagen': self.colors['neuro'],           # Púrpura para neuroimagen
                'lifestyle': self.colors['low_risk']           # Verde para estilo de vida
            },
            'gradients': {
                'primary': f"linear-gradient(135deg, {self.colors['primary']}, #1d4ed8)",
                'secondary': f"linear-gradient(135deg, {self.colors['secondary']}, #0d9488)",
                'risk': f"linear-gradient(90deg, {self.colors['low_risk']}, {self.colors['moderate_risk']}, {self.colors['high_risk']})",
                'neuro': f"linear-gradient(135deg, {self.colors['neuro']}, #6366f1)"
            }
        }
        
        # Configuración de transparencias para overlays - CORREGIDO
        self.alpha_colors = {
            'primary_10': f"rgba(37, 99, 235, 0.1)",      # 10% opacity
            'primary_20': f"rgba(37, 99, 235, 0.2)",      # 20% opacity
            'low_risk_10': f"rgba(5, 150, 105, 0.1)",
            'moderate_risk_10': f"rgba(217, 119, 6, 0.1)",
            'high_risk_10': f"rgba(220, 38, 38, 0.1)",
            'neuro_10': f"rgba(124, 58, 237, 0.1)"
        }
        
        # Mantener configuración existente
        self.feature_groups = {
            'biomarcadores': ['tau_protein', 'abeta_42', 'ptau_181', 'nfl_protein'],
            'neuroimagen': ['hippocampus_volume', 'entorhinal_thickness', 'whole_brain_volume'],
            'cognitivo': ['mmse_score', 'cdr_score', 'adas_cog_score'],
            'lifestyle': ['physical_activity', 'social_engagement', 'sleep_quality'],
            'demografico': ['age', 'education_years', 'apoe4_carriers']
        }
        
        self.feature_explanations = {
            'tau_protein': 'Proteína Tau: Biomarcador clave del Alzheimer. Valores altos indican daño neuronal.',
            'abeta_42': 'Beta-amiloide 42: Proteína que forma placas en el cerebro. Niveles bajos son preocupantes.',
            'mmse_score': 'Mini-Mental State Exam: Evaluación cognitiva básica (0-30 puntos).',
            'age': 'Edad: Factor de riesgo no modificable. El riesgo aumenta después de los 65 años.',
            'education_years': 'Años de Educación: Mayor educación puede ser protectora contra el Alzheimer.',
            'physical_activity': 'Actividad Física: Ejercicio regular reduce el riesgo de deterioro cognitivo.',
            'hippocampus_volume': 'Volumen del Hipocampo: Área cerebral crucial para la memoria.',
            'apoe4_carriers': 'Gen APOE4: Variante genética que aumenta el riesgo de Alzheimer.'
        }

        # Rutas de archivos reales
        self.config_file = REPORTS_PATH / "dashboard_complete_config.json"
        self.winners_file = REPORTS_PATH / "model_winners_summary.csv"
        self.feature_importance_file = REPORTS_PATH / "feature_importance_detailed_summary.csv"
        self.validation_file = REPORTS_PATH / "subgroup_validation_results.csv"
        self.clinical_metrics_file = REPORTS_PATH / "clinical_impact_metrics.json"
        
        # Datos reales
        self.real_data_path = DATA_PATH
        
        # MLflow configuración
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.experiments_mapping = {
            'regression': 'alzheimer_multimodal_monitoring',
            'classification': 'Default',  # Aquí está el ensemble_classification
            'temporal': 'alzheimer_temporal_analysis',
            'stratification': 'alzheimer_risk_stratification'
        }
        
        # Cargar configuración real si existe
        self.load_real_config()

    def load_real_config(self):
        """Carga configuración real desde archivos de la etapa 5"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    real_config = json.load(f)
                    # Actualizar configuración con datos reales
                    if 'feature_groups' in real_config:
                        self.feature_groups = real_config['feature_groups']
                    if 'thresholds' in real_config:
                        self.thresholds.update(real_config['thresholds'])
                    logger.info("Configuración real cargada exitosamente")
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_file}")
        except Exception as e:
            logger.error(f"Error cargando configuración real: {e}")


def hex_to_rgba(hex_color, alpha=0.1):
    """Convierte color hexadecimal a rgba con transparencia"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(128, 128, 128, {alpha})"  # fallback

# Instancia global de configuración
config = AlzheimerDashboardConfig()

def verify_mlflow_setup():
    """Verificar configuración MLflow"""
    logger.info("=== VERIFICACIÓN MLFLOW ===")

    # Si mlflow no está disponible, salir temprano
    if not mlflow_available:
        logger.info("MLflow deshabilitado o no disponible; se omite verificación de mlruns.")
        return
    
    # Verificar directorio mlruns
    mlruns_path = Path("./mlruns")
    if mlruns_path.exists():
        logger.info(f"✅ Directorio mlruns encontrado: {mlruns_path.absolute()}")
        
        # Listar experimentos disponibles
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            experiments = mlflow.search_experiments()
            logger.info(f"✅ Experimentos encontrados: {len(experiments)}")
            for exp in experiments:
                logger.info(f"   - {exp.name} (ID: {exp.experiment_id})")
        except Exception as e:
            logger.error(f"❌ Error listando experimentos: {e}")
    else:
        logger.error(f"❌ Directorio mlruns no encontrado: {mlruns_path.absolute()}")
        logger.info("💡 Ejecuta el dashboard desde la carpeta raíz del proyecto")

# Llamar verificación
verify_mlflow_setup()

# ==========================================
# GENERACIÓN DE DATOS SIMULADOS
# ==========================================

def generate_sample_data():
    """Genera datos de muestra para el dashboard"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generar características base
    data = {
        'patient_id': [f'PAT_{i:04d}' for i in range(n_samples)],
        'age': np.random.normal(72, 8, n_samples).clip(50, 90),
        'education_years': np.random.normal(14, 4, n_samples).clip(8, 20),
        'mmse_score': np.random.normal(26, 4, n_samples).clip(0, 30),
        'cdr_score': np.random.exponential(0.5, n_samples).clip(0, 3),
        'adas_cog_score': np.random.exponential(8, n_samples).clip(0, 70),
        'tau_protein': np.random.lognormal(4.5, 0.8, n_samples),
        'abeta_42': np.random.normal(800, 200, n_samples).clip(200, 1500),
        'ptau_181': np.random.lognormal(3.2, 0.6, n_samples),
        'nfl_protein': np.random.lognormal(3.8, 0.7, n_samples),
        'hippocampus_volume': np.random.normal(7500, 1200, n_samples).clip(4000, 10000),
        'entorhinal_thickness': np.random.normal(3.2, 0.6, n_samples).clip(1.5, 5.0),
        'whole_brain_volume': np.random.normal(1200000, 150000, n_samples).clip(900000, 1500000),
        'physical_activity': np.random.beta(2, 3, n_samples) * 10,
        'social_engagement': np.random.beta(3, 2, n_samples) * 10,
        'sleep_quality': np.random.normal(7, 1.5, n_samples).clip(3, 10),
        'apoe4_carriers': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Generar probabilidades de riesgo basadas en las características
    risk_score = (
        (df['age'] - 50) / 40 * 0.2 +
        (30 - df['mmse_score']) / 30 * 0.25 +
        df['cdr_score'] / 3 * 0.2 +
        np.log(df['tau_protein'] / df['tau_protein'].median()) * 0.15 +
        (df['abeta_42'].median() - df['abeta_42']) / df['abeta_42'].std() * 0.1 +
        df['apoe4_carriers'] / 2 * 0.1
    )
    
    df['risk_probability'] = 1 / (1 + np.exp(-risk_score.clip(-5, 5)))
    
    # Categorizar riesgo
    df['risk_category'] = pd.cut(
        df['risk_probability'],
        bins=[0, config.thresholds['low_risk'], config.thresholds['moderate_risk'], 1],
        labels=['Bajo', 'Moderado', 'Alto']
    )
    
    return df

def generate_feature_importance():
    """Genera importancia de características para el modelo"""
    features = [
        'tau_protein', 'mmse_score', 'age', 'cdr_score', 'abeta_42',
        'hippocampus_volume', 'adas_cog_score', 'apoe4_carriers',
        'entorhinal_thickness', 'education_years', 'physical_activity',
        'ptau_181', 'nfl_protein', 'social_engagement', 'sleep_quality'
    ]
    
    # Importancias basadas en literatura médica
    importances = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005]
    
    return pd.DataFrame({
        'feature': features,
        'importance': importances,
        'group': [
            'biomarcadores', 'cognitivo', 'demografico', 'cognitivo', 'biomarcadores',
            'neuroimagen', 'cognitivo', 'demografico', 'neuroimagen', 'demografico',
            'lifestyle', 'biomarcadores', 'biomarcadores', 'lifestyle', 'lifestyle'
        ]
    }).sort_values('importance', ascending=False)


def load_real_data():
    """Carga datos reales desde archivos de la etapa 5"""
    try:
        # Intentar cargar modelo ganador
        winners_df = None
        if config.winners_file.exists():
            winners_df = pd.read_csv(config.winners_file)
            logger.info(f"Modelos ganadores cargados: {len(winners_df)} modelos")
        
        # Intentar cargar datos procesados reales
        # Ruta específica de tu archivo
        specific_data_path = PROJECT_ROOT / "data" / "processed" / "features" / "alzheimer_features_selected_20250621.csv"
        
        if specific_data_path.exists():
            real_data = pd.read_csv(specific_data_path)
            logger.info(f"Datos reales cargados desde archivo específico: {len(real_data)} pacientes")
            return real_data, winners_df
        else:
            # Fallback: buscar otros archivos CSV en processed
            real_data_files = list(config.real_data_path.glob("**/*.csv"))  # Búsqueda recursiva
            if real_data_files:
                logger.info(f"Archivos encontrados: {[f.name for f in real_data_files]}")
                # Buscar archivo con patrón específico
                main_data_file = next((f for f in real_data_files 
                                     if 'alzheimer_features' in f.name or 'processed' in f.name or 'final' in f.name), 
                                     None)
                if main_data_file:
                    real_data = pd.read_csv(main_data_file)
                    logger.info(f"Datos reales cargados desde: {main_data_file.name} - {len(real_data)} pacientes")
                    return real_data, winners_df
        
        logger.warning("No se encontraron datos reales, usando datos simulados")
        return None, None
        
    except Exception as e:
        logger.error(f"Error cargando datos reales: {e}")
        return None, None

def load_real_feature_importance():
    """Carga importancia real de características"""
    try:
        if config.feature_importance_file.exists():
            feature_importance = pd.read_csv(config.feature_importance_file)
            logger.info(f"Feature importance real cargada: {len(feature_importance)} características")
            logger.info(f"Columnas disponibles: {feature_importance.columns.tolist()}")
            
            # Mapear las columnas reales a las que espera el código
            if 'avg_importance' in feature_importance.columns:
                feature_importance['importance'] = feature_importance['avg_importance']
            elif 'perm_importance_avg' in feature_importance.columns:
                feature_importance['importance'] = feature_importance['perm_importance_avg']
            else:
                # Si no hay ninguna columna de importancia reconocida, crear una simulada
                logger.warning("No se encontró columna de importancia reconocida, usando valores simulados")
                return generate_feature_importance()
            
            # Mapear la columna category a group si no existe group
            if 'group' not in feature_importance.columns and 'category' in feature_importance.columns:
                feature_importance['group'] = feature_importance['category']
            elif 'group' not in feature_importance.columns:
                # Asignar grupos por defecto basados en el nombre de la característica
                feature_importance['group'] = feature_importance['feature'].apply(assign_feature_group)
            
            # Asegurar que tenemos las columnas necesarias
            required_columns = ['feature', 'importance', 'group']
            for col in required_columns:
                if col not in feature_importance.columns:
                    logger.error(f"Columna requerida '{col}' no encontrada")
                    return generate_feature_importance()
            
            # Ordenar por importancia
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            logger.warning("Archivo de feature importance no encontrado, usando simulado")
            return generate_feature_importance()
    except Exception as e:
        logger.error(f"Error cargando feature importance real: {e}")
        return generate_feature_importance()

def assign_feature_group(feature_name):
    """Asigna grupo a una característica basado en su nombre"""
    feature_name_lower = feature_name.lower()
    
    if any(word in feature_name_lower for word in ['tau', 'abeta', 'ptau', 'nfl', 'protein']):
        return 'biomarcadores'
    elif any(word in feature_name_lower for word in ['mmse', 'cdr', 'adas', 'cognitive', 'memory']):
        return 'cognitivo'
    elif any(word in feature_name_lower for word in ['age', 'education', 'apoe', 'sex', 'gender']):
        return 'demografico'
    elif any(word in feature_name_lower for word in ['hippocampus', 'volume', 'thickness', 'brain']):
        return 'neuroimagen'
    elif any(word in feature_name_lower for word in ['activity', 'physical', 'sleep', 'social']):
        return 'lifestyle'
    else:
        return 'demografico'  # Por defecto




# :
def generate_monitoring_data():
    """Genera datos de monitoreo para el dashboard"""
    dates = pd.date_range(start='2024-01-01', end='2025-08-31', freq='W')
    monitoring_data = pd.DataFrame({
        'date': dates,
        'total_patients': np.cumsum(np.random.poisson(10, len(dates))) + 100,
        'high_risk': np.cumsum(np.random.poisson(2, len(dates))) + 20,
        'moderate_risk': np.cumsum(np.random.poisson(3, len(dates))) + 30,
        'low_risk': np.cumsum(np.random.poisson(5, len(dates))) + 50,
        'model_accuracy': 0.87 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001
    })
    monitoring_data['model_accuracy'] = monitoring_data['model_accuracy'].clip(0.82, 0.92)
    return monitoring_data

def load_mlflow_models(winners_df=None):
    """Carga modelos desde MLflow usando los experimentos correctos"""
    models = {}
    
    if not mlflow_available:
        logger.info("MLflow no disponible - omitiendo carga de modelos desde MLflow.")
        return {}

    try:
        # Configurar MLflow con la ruta correcta
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        logger.info(f"Conectando a MLflow en: {config.mlflow_tracking_uri}")
        
        # Mapeo de experimentos (igual que en tus notebooks)
        experiments_mapping = {
            'regression': 'alzheimer_multimodal_monitoring',
            'classification': 'Default',  # Aquí está el ensemble_classification
            'temporal': 'alzheimer_temporal_analysis',
            'stratification': 'alzheimer_risk_stratification'
        }
        
        # Cargar modelos de cada experimento
        for task, exp_name in experiments_mapping.items():
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    logger.info(f"✅ {task.upper()}: Experimento '{exp_name}' encontrado")
                    
                    # Buscar runs finalizados
                    all_runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="status = 'FINISHED'",
                        order_by=["start_time DESC"],
                        max_results=10
                    )
                    
                    if not all_runs.empty:
                        if task == 'classification':
                            # Buscar específicamente ensemble_classification
                            ensemble_runs = all_runs[
                                all_runs.get('tags.mlflow.runName', pd.Series()) == 'ensemble_classification'
                            ]
                            if not ensemble_runs.empty:
                                runs = ensemble_runs
                                logger.info(f"🎯 Encontrado ensemble_classification")
                            else:
                                runs = all_runs.head(3)
                                logger.info(f"📊 Usando {len(runs)} runs de clasificación")
                        else:
                            runs = all_runs.head(3)
                            logger.info(f"📊 {task}: {len(runs)} runs encontrados")
                        
                        # Cargar el mejor modelo de cada tipo
                        for idx, run in runs.iterrows():
                            try:
                                run_id = run.run_id
                                model_name = f"{task}_{idx}" if pd.isna(run.get('tags.mlflow.runName', None)) else f"{task}_{run.get('tags.mlflow.runName', idx)}"
                                
                                # Intentar cargar el modelo
                                model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
                                
                                # Extraer métricas
                                metrics = {}
                                for col in run.index:
                                    if col.startswith('metrics.'):
                                        metric_name = col.replace('metrics.', '')
                                        metrics[metric_name] = run[col]
                                
                                models[model_name] = {
                                    'model': model,
                                    'metrics': metrics,
                                    'run_id': run_id,
                                    'task': task,
                                    'experiment': exp_name
                                }
                                
                                logger.info(f"✅ Modelo cargado: {model_name}")
                                break  # Solo cargar el mejor de cada tipo
                                
                            except Exception as e:
                                logger.warning(f"⚠️ Error cargando run {run_id}: {e}")
                                continue
                    else:
                        logger.warning(f"⚠️ {task}: Sin runs finalizados en '{exp_name}'")
                else:
                    logger.warning(f"❌ {task.upper()}: Experimento '{exp_name}' no encontrado")
                    
            except Exception as e:
                logger.error(f"❌ Error procesando {task}: {e}")
        
        logger.info(f"✅ Total modelos MLflow cargados: {len(models)}")
        return models
        
    except Exception as e:
        logger.error(f"❌ Error conectando con MLflow: {e}")
        return {}


def load_pickle_model():
    """Carga modelo pickle como fallback"""
    try:
        import pickle
        
        pickle_model_path = PROJECT_ROOT / "models" / "pretrained" / "best_xgboost_model.pkl"
        
        if pickle_model_path.exists():
            with open(pickle_model_path, 'rb') as f:
                pickle_model = pickle.load(f)
            
            logger.info(f"✅ Modelo pickle cargado: {pickle_model_path}")
            
            return {
                'pickle_xgboost': {
                    'model': pickle_model,
                    'metrics': {
                        'accuracy': 0.87,  # Métricas por defecto
                        'precision': 0.85,
                        'recall': 0.89,
                        'f1_score': 0.87
                    },
                    'run_id': 'pickle_model',
                    'task': 'classification',
                    'experiment': 'pickle_fallback'
                }
            }
        else:
            logger.warning(f"❌ Modelo pickle no encontrado: {pickle_model_path}")
            return {}
            
    except ModuleNotFoundError as me:
        logger.error(f"❌ Error cargando modelo pickle (falta librería): {me}")
        if "xgboost" in str(me).lower():
            logger.error("El pickle requiere 'xgboost'. Añade 'xgboost' a requirements.txt y redeploy.")
        return {}
    except Exception as e:
        logger.error(f"❌ Error cargando modelo pickle: {e}")
        return {}


def get_mlflow_runs_info():
    """Obtiene información de los runs disponibles en MLflow"""
    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        # Obtener experimentos
        experiments = mlflow.search_experiments()
        logger.info(f"Experimentos encontrados: {len(experiments)}")
        
        runs_info = []
        for experiment in experiments:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            logger.info(f"Experimento {experiment.name}: {len(runs)} runs")
            
            for idx, run in runs.iterrows():
                runs_info.append({
                    'run_id': run['run_id'],
                    'experiment_name': experiment.name,
                    'status': run['status'],
                    'metrics': {key: value for key, value in run.items() if key.startswith('metrics.')},
                    'start_time': run['start_time']
                })
        
        logger.info(f"Total runs encontrados: {len(runs_info)}")
        return runs_info
        
    except Exception as e:
        logger.error(f"Error obteniendo runs de MLflow: {e}")
        return []

def debug_files_status():
    """Debug: verificar estado de archivos"""
    logger.info("=== DEBUG: Estado de archivos ===")
    files_to_check = [
        config.config_file,
        config.winners_file,
        config.feature_importance_file,
        config.validation_file,
        config.clinical_metrics_file
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            logger.info(f"✅ Encontrado: {file_path}")
            if file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"   - Filas: {len(df)}, Columnas: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"   - Error leyendo CSV: {e}")
        else:
            logger.warning(f"❌ No encontrado: {file_path}")
    
    # Verificar directorio MLflow 
    mlflow_dir = PROJECT_ROOT / "mlruns"  # Cambiar aquí
    if mlflow_dir.exists():
        logger.info(f"✅ Directorio MLflow encontrado: {mlflow_dir}")
        # Listar experimentos
        experiments = list(mlflow_dir.glob("*/"))
        logger.info(f"   - Experimentos encontrados: {[e.name for e in experiments]}")
    else:
        logger.warning(f"❌ Directorio MLflow no encontrado: {mlflow_dir}")

    # Verificar modelo pickle
    pickle_model_path = PROJECT_ROOT / "models" / "pretrained" / "best_xgboost_model.pkl"
    if pickle_model_path.exists():
        logger.info(f"✅ Modelo pickle encontrado: {pickle_model_path}")
    else:
        logger.warning(f"❌ Modelo pickle no encontrado: {pickle_model_path}")
        # Listar archivos en el directorio models
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            logger.info(f"📁 Archivos en {models_dir}:")
            for item in models_dir.rglob("*"):
                if item.is_file():
                    logger.info(f"   - {item.relative_to(models_dir)}")

def get_model_feature_names(model):
    """
    Intenta obtener una lista de feature names esperadas por el modelo.
    Maneja varios tipos: sklearn (feature_names_in_), xgboost Booster, wrappers con get_booster, y atributos comunes.
    Devuelve list[str] o None si no se pudo obtener.
    """
    try:
        # sklearn >= 1.0
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)

        # XGBoost sklearn wrapper (XGBClassifier/XGBRegressor) -> booster().feature_names
        if hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                if booster is not None and hasattr(booster, "feature_names") and booster.feature_names is not None:
                    return list(booster.feature_names)
            except Exception:
                pass

        # direct attribute (some saved boosters)
        if hasattr(model, "_Booster"):
            booster = getattr(model, "_Booster")
            if booster is not None and hasattr(booster, "feature_names") and booster.feature_names is not None:
                return list(booster.feature_names)

        # generic attribute
        if hasattr(model, "feature_names"):
            fn = getattr(model, "feature_names")
            if isinstance(fn, (list, tuple, np.ndarray)):
                return list(fn)

        # mlflow pyfunc models sometimes expose a "metadata" or "metadata" attribute with signature
        if hasattr(model, "metadata"):
            metadata = getattr(model, "metadata")
            try:
                # best-effort: mlflow pyfunc pyfunc-loaded model may have .metadata.get("signature")
                sig = metadata.get("signature", None) if isinstance(metadata, dict) else None
                if sig and "inputs" in sig:
                    return [inp["name"] for inp in sig["inputs"]]
            except Exception:
                pass

    except Exception:
        pass

    return None


def _normalize_name(s):
    if s is None:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# mapa de alias (best-effort) — extiéndelo según tus nombres reales
_ALIAS_MAP = {
    'age': ['age', 'edad', 'age_years', 'age_at_visit_estimated'],
    'education_years': ['education_years', 'educ', 'years_of_education'],
    'mmse_score': ['mmse_score', 'mmse', 'mmse_total'],
    'cdr_score': ['cdr_score', 'cdrsb', 'cdr'],
    'tau_protein': ['tau_protein', 'ptau', 'ptau181', 'tau'],
    'abeta_42': ['abeta_42', 'abeta42', 'ab42', 'abeta'],
    'apoe4_carriers': ['apoe4_carriers', 'apoe4', 'apoe_e4', 'apoe_e4_carrier']
}

def _build_alias_reverse_map():
    rev = {}
    for canonical, variants in _ALIAS_MAP.items():
        for v in variants:
            rev[_normalize_name(v)] = canonical
    return rev
_ALIAS_REVERSE = _build_alias_reverse_map()

def predict_with_real_model(patient_data, model_name=None):
    """Predicción robusta que intenta mapear nombres de input a los nombres esperados por el modelo."""
    try:
        if model_name is None and loaded_models:
            model_name = list(loaded_models.keys())[0]
            logger.info(f"Usando primer modelo disponible: {model_name}")

        logger.info(f"Intentando predicción con modelo: {model_name}")
        logger.info(f"Modelos disponibles: {list(loaded_models.keys())}")

        if not loaded_models or model_name not in loaded_models:
            logger.warning(f"⚠️ Modelo {model_name} no encontrado, usando cálculo simulado")
            return None, None

        model_info = loaded_models[model_name]
        model = model_info['model']

        # input como DataFrame único
        input_df = pd.DataFrame([patient_data])

        # obtener features esperadas por el modelo
        expected = get_model_feature_names(model)
        if expected:
            logger.info(f"Modelo {model_name} declara {len(expected)} features.")
            expected = list(expected)
            # Normalización: construir maps normalizados
            expected_norm_map = { _normalize_name(c): c for c in expected }
            input_keys = list(input_df.columns)
            input_norm_map = { _normalize_name(k): k for k in input_keys }

            # Mapeo resultado: expected_col -> source_key (si se encuentra)
            mapping = {}
            for enorm, ecol in expected_norm_map.items():
                # 1) Coincidencia exacta normalizada en input
                if enorm in input_norm_map:
                    mapping[ecol] = input_norm_map[enorm]
                    continue
                # 2) alias reverse map (try to map canonical names)
                if enorm in _ALIAS_REVERSE:
                    canonical = _ALIAS_REVERSE[enorm]
                    # si user provided canonical (normalized)
                    if _normalize_name(canonical) in input_norm_map:
                        mapping[ecol] = input_norm_map[_normalize_name(canonical)]
                        continue
                # 3) intentar buscar en input por tokens (contains)
                found = None
                for inorm, ik in input_norm_map.items():
                    if inorm in enorm or enorm in inorm:
                        found = ik
                        break
                if found:
                    mapping[ecol] = found
                    continue
                # no se encontró — mapping quedará vacío y se rellenará luego
            # Logging del mapeo
            matched = len(mapping)
            logger.info(f"Feature mapping: {matched}/{len(expected)} features mapeadas desde entradas provistas.")
            if matched < max(1, int(0.6*len(expected))):
                # si se mapearon muy pocas columnas, emitimos advertencia
                logger.warning(f"Pocas features mapeadas ({matched}/{len(expected)}). Revisa nombres de campos enviados desde la UI.")
            # Crear columnas en input_df para expected — asignar valores mapeados
            for col in expected:
                if col in mapping:
                    src_key = mapping[col]
                    input_df[col] = input_df[src_key]
                else:
                    # si no existe en input, intentar medianas del dataset
                    if 'sample_data' in globals() and isinstance(sample_data, pd.DataFrame) and col in sample_data.columns:
                        input_df[col] = sample_data[col].median()
                    else:
                        input_df[col] = 0
            # Reordenar según expected
            input_df = input_df[expected]
        else:
            # Si no se conocen features, quedarnos con lo que dio el usuario (fallback previo)
            logger.info("No se pudieron obtener feature names del modelo; usando las columnas provistas por usuario.")
            # fallback pequeño ya existente
            if model_name == 'pickle_xgboost':
                expected_small = ['age', 'education_years', 'mmse_score', 'cdr_score', 'tau_protein', 'abeta_42', 'apoe4_carriers']
                for col in expected_small:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[[c for c in expected_small if c in input_df.columns]]

        # Intentar predecir
        try:
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_df)
                probability = prediction_proba[0][1] if prediction_proba.shape[1] > 1 else prediction_proba[0][0]
            elif hasattr(model, 'predict'):
                prediction = model.predict(input_df)
                probability = prediction[0] if isinstance(prediction, (list, np.ndarray)) else float(prediction)
            else:
                logger.error(f"Modelo {model_name} no tiene método predict/predict_proba")
                return None, None
        except Exception as inner_e:
            logger.warning(f"Error al predecir con {model_name}: {inner_e}. Intentando fallback con numpy array.")
            try:
                arr = input_df.values.astype(float)
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(arr)[0][1]
                else:
                    probability = model.predict(arr)[0]
            except Exception as inner2:
                logger.error(f"Fallback falló para {model_name}: {inner2}")
                return None, None

        probability = float(np.clip(probability, 0, 1))
        logger.info(f"✅ Predicción exitosa con {model_name}: {probability:.3f}")
        # adicional: log corto de qué columnas del input variaron (opcional)
        return probability, model_info.get('metrics', {})

    except Exception as e:
        logger.error(f"❌ Error en predicción con {model_name}: {e}")
        return None, None

# Cargar datos (reales o simulados)
real_data, winners_df = load_real_data()
if real_data is not None:
    sample_data = real_data
    logger.info("Usando datos reales")
else:
    sample_data = generate_sample_data()
    logger.info("Usando datos simulados")

# Cargar feature importance (real o simulada)
feature_importance = load_real_feature_importance()

# Cargar datos de monitoreo
monitoring_data = generate_monitoring_data()

# Cargar modelos desde MLflow (sin depender de winners_df)
loaded_models = {}
loaded_models = load_mlflow_models()

# Si no hay modelos MLflow, cargar modelo pickle
if not loaded_models:
    logger.info("No hay modelos MLFlow, cargando  modelo pickle...")
    pickle_models = load_pickle_model()
    loaded_models.update(pickle_models)

# Mensaje si no se cargó ningún modelo
if not loaded_models:
    logger.warning("No se pudieron cargar modelos MLflow ni pickle, usando solo cálculos simulados.")

# Llamar debug al inicializar
debug_files_status()
# Debug final
logger.info(f"Estado final - Modelos cargados: {len(loaded_models)}")
if loaded_models:
    logger.info(f"Nombres de modelos: {list(loaded_models.keys())}")
# ==========================================
# FUNCIONES DE UTILIDAD
# ==========================================

def get_risk_color(probability):
    """Obtiene el color según la probabilidad de riesgo con nueva paleta"""
    if probability < config.thresholds['low_risk']:
        return config.colors['low_risk']      # Verde esmeralda
    elif probability < config.thresholds['moderate_risk']:
        return config.colors['moderate_risk'] # Ámbar
    else:
        return config.colors['high_risk']     # Rojo coral

def get_risk_category_style(category):
    """Obtiene estilos completos para categorías de riesgo"""
    styles = {
        'Bajo': {
            'color': config.colors['low_risk'],
            'backgroundColor': config.alpha_colors['low_risk_10'],
            'borderColor': config.colors['low_risk']
        },
        'Moderado': {
            'color': config.colors['moderate_risk'],
            'backgroundColor': config.alpha_colors['moderate_risk_10'],
            'borderColor': config.colors['moderate_risk']
        },
        'Alto': {
            'color': config.colors['high_risk'],
            'backgroundColor': config.alpha_colors['high_risk_10'],
            'borderColor': config.colors['high_risk']
        }
    }
    return styles.get(category, styles['Bajo'])

def get_feature_group_color(group):
    """Obtiene color específico para grupo de características"""
    return config.color_maps['feature_groups'].get(group, config.colors['primary'])



def get_risk_category(probability):
    """Obtiene la categoría de riesgo"""
    if probability < config.thresholds['low_risk']:
        return 'Bajo'
    elif probability < config.thresholds['moderate_risk']:
        return 'Moderado'
    else:
        return 'Alto'

def generate_explanation(patient_data, risk_prob):
    """Genera explicación personalizada del riesgo"""
    explanations = []
    
    # Factores principales
    if patient_data.get('age', 65) > 75:
        explanations.append("🔴 Edad avanzada (>75 años) aumenta significativamente el riesgo")
    elif patient_data.get('age', 65) > 65:
        explanations.append("🟡 Edad moderada (65-75 años) contribuye al riesgo")
    
    if patient_data.get('mmse_score', 30) < 24:
        explanations.append("🔴 Puntuación MMSE baja indica deterioro cognitivo significativo")
    elif patient_data.get('mmse_score', 30) < 27:
        explanations.append("🟡 Puntuación MMSE moderada sugiere leve deterioro cognitivo")
    
    if patient_data.get('tau_protein', 50) > 80:
        explanations.append("🔴 Niveles altos de proteína Tau indican daño neuronal")
    
    if patient_data.get('abeta_42', 800) < 600:
        explanations.append("🔴 Niveles bajos de Aβ42 sugieren formación de placas")
    
    if patient_data.get('apoe4_carriers', 0) >= 1:
        explanations.append("🔴 Presencia de gen APOE4 aumenta el riesgo genético")
    
    # Factores protectores
    if patient_data.get('education_years', 12) > 16:
        explanations.append("🟢 Alto nivel educativo puede ser protector")
    
    if patient_data.get('physical_activity', 5) > 7:
        explanations.append("🟢 Alta actividad física es protectora")
    
    return explanations

def generate_recommendations(patient_data, risk_prob):
    """Genera recomendaciones personalizadas"""
    recommendations = []
    
    if risk_prob > config.thresholds['moderate_risk']:
        recommendations.extend([
            "🏥 Consulta neurológica especializada urgente",
            "🧪 Evaluación de biomarcadores adicionales",
            "🧠 Resonancia magnética cerebral detallada"
        ])
    
    if patient_data.get('physical_activity', 5) < 5:
        recommendations.append("🚶‍♀️ Aumentar actividad física: caminatas diarias de 30 min")
    
    if patient_data.get('social_engagement', 5) < 5:
        recommendations.append("👥 Incrementar actividad social y estimulación cognitiva")
    
    if patient_data.get('sleep_quality', 7) < 6:
        recommendations.append("😴 Mejorar higiene del sueño: 7-8 horas diarias")
    
    recommendations.extend([
        "🥗 Dieta mediterránea rica en antioxidantes",
        "📚 Actividades cognitivamente estimulantes",
        "🎵 Terapia musical y actividades creativas",
        "👨‍⚕️ Seguimiento médico regular cada 6 meses"
    ])
    
    return recommendations


# Configuración para producción
if os.environ.get('RENDER'):
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# ==========================================
# INICIALIZACIÓN DE LA APP DASH
# ==========================================

app = dash.Dash(__name__, external_stylesheets=[
    'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

# EXPONER EL SERVIDOR PARA GUNICORN
server = app.server

# Permitir callbacks en componentes que se cargan dinámicamente
app.config.suppress_callback_exceptions = True

app.title = "Dashboard Alzheimer - Monitorización Inteligente"

# ==========================================
# LAYOUT PRINCIPAL
# ==========================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-brain mr-3 text-blue-500"),
                "Dashboard de Monitorización de Alzheimer"
            ], className="text-3xl font-bold text-gray-800"),
            html.P(
                "Sistema inteligente de evaluación y prevención - Accesible para todos los usuarios",
                className="text-gray-600 mt-2"
            )
        ], className="flex-1"),
        html.Div([
            html.Div([
                html.I(className="fas fa-clock mr-2"),            # icono de reloj
                html.Span(id="client-time", children="Cargando...")  # aquí irá la hora cliente
            ], className="text-sm text-gray-600"),
            # añade este Interval junto al resto de componentes de layout
            dcc.Interval(id="interval-time", interval=1000, n_intervals=0),  # actualiza cada 1000 ms
            html.Div([
                html.I(className="fas fa-users mr-2"), # Icono de usuarios
                f"≈2000 pacientes monitoreados" # Número de pacientes
            ], className="text-sm text-gray-600 mt-1")
        ])
    ], className="bg-white shadow-lg rounded-lg p-6 mb-6 flex items-center justify-between"),
    
    # Tabs Navigation
    dcc.Tabs(id="main-tabs", value="risk-evaluation", children=[
        dcc.Tab(label="🎯 Evaluación de Riesgo", value="risk-evaluation",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="📊 Análisis de Factores", value="risk-factors",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="👥 Casos Educativos", value="case-examples",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="📈 Monitoreo", value="monitoring",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="🧠 Centro de Aprendizaje", value="learning-center",
                className="tab-style", selected_className="tab-selected")
    ], className="mb-6"),
    
    # Content Area
    html.Div(id="tabs-content", className="min-h-screen")
    
], className="bg-gray-50 min-h-screen p-4")

# ==========================================
# CALLBACKS PRINCIPALES
# ==========================================


app.clientside_callback(
    """
    function(n_intervals) {
        const now = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const dd = pad(now.getDate());
        const mm = pad(now.getMonth() + 1);
        const yyyy = now.getFullYear();
        const hh = pad(now.getHours());
        const mi = pad(now.getMinutes());
        const ss = pad(now.getSeconds());
        return `${dd}/${mm}/${yyyy} ${hh}:${mi}:${ss}`;
    }
    """,
    Output("client-time", "children"),
    Input("interval-time", "n_intervals")
)

@app.callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value")
)
def render_tab_content(active_tab):
    """Renderiza el contenido de cada tab"""
    if active_tab == "risk-evaluation":
        return render_risk_evaluation_tab()
    elif active_tab == "risk-factors":
        return render_risk_factors_tab()
    elif active_tab == "case-examples":
        return render_case_examples_tab()
    elif active_tab == "monitoring":
        return render_monitoring_tab()
    elif active_tab == "learning-center":
        return render_learning_center_tab()

def render_risk_evaluation_tab():
    """Tab 1: Evaluación de Riesgo Individual"""
    return html.Div([
        # Información introductoria
        html.Div([
            html.H2("🎯 Evaluación Personalizada de Riesgo", className="text-2xl font-bold text-gray-800 mb-4"),
            html.P([
                "Introduce los datos del paciente para obtener una evaluación completa del riesgo de Alzheimer. ",
                html.Strong("Todos los valores son opcionales"), " - el sistema calculará el riesgo con la información disponible."
            ], className="text-gray-600 mb-6")
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Panel de entrada de datos
            html.Div([
                html.H3("📝 Datos del Paciente", className="text-xl font-bold text-gray-800 mb-4"),
                
                # Datos demográficos
                html.Div([
                    html.H4("👤 Información Demográfica", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("Edad (años)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-age", type="number", value=70, min=50, max=100,
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Años de Educación", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-education", type="number", value=14, min=0, max=25,
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Gen APOE4 (copias)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Dropdown(id="input-apoe4", options=[
                                {"label": "0 copias (sin riesgo genético)", "value": 0},
                                {"label": "1 copia (riesgo moderado)", "value": 1},
                                {"label": "2 copias (alto riesgo genético)", "value": 2}
                            ], value=0, className="mb-4")
                        ])
                    ], className="grid grid-cols-1 gap-4")
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Evaluaciones cognitivas
                html.Div([
                    html.H4("🧠 Evaluaciones Cognitivas", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("MMSE Score (0-30)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Slider(id="input-mmse", min=0, max=30, value=26, marks={i: str(i) for i in range(0, 31, 5)},
                                     tooltip={"placement": "bottom", "always_visible": True})
                        ], className="mb-6"),
                        html.Div([
                            html.Label("CDR Score (0-3)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Slider(id="input-cdr", min=0, max=3, step=0.5, value=0.5,
                                     marks={i: str(i) for i in np.arange(0, 3.5, 0.5)},
                                     tooltip={"placement": "bottom", "always_visible": True})
                        ], className="mb-6")
                    ])
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Biomarcadores
                html.Div([
                    html.H4("🧪 Biomarcadores (Opcionales)", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("Proteína Tau (pg/mL)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-tau", type="number", value=85, min=0, max=200,
                                    placeholder="Valores normales: 40-80",
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Beta-amiloide 42 (pg/mL)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-abeta", type="number", value=750, min=200, max=1500,
                                    placeholder="Valores normales: 800-1200",
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4")
                    ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Botón de evaluación
                html.Button(
                    [html.I(className="fas fa-calculator mr-2"), "Evaluar Riesgo"],
                    id="evaluate-button",
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200",
                    n_clicks=0
                )
                
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Panel de resultados
            html.Div([
                html.H3("📊 Resultado de la Evaluación", className="text-xl font-bold text-gray-800 mb-4"),
                html.Div(id="risk-results-content")
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 lg:grid-cols-2 gap-6")
    ])

@app.callback(
    Output("risk-results-content", "children"),
    [Input("evaluate-button", "n_clicks")],
    [State("input-age", "value"),
     State("input-education", "value"),
     State("input-apoe4", "value"),
     State("input-mmse", "value"),
     State("input-cdr", "value"),
     State("input-tau", "value"),
     State("input-abeta", "value")]
)
def update_risk_evaluation(n_clicks, age, education, apoe4, mmse, cdr, tau, abeta):
    """Actualiza la evaluación de riesgo"""
    if n_clicks == 0:
        return html.Div([
            html.I(className="fas fa-info-circle text-blue-500 text-6xl mb-4"),
            html.P("Haz clic en 'Evaluar Riesgo' para obtener los resultados", 
                   className="text-gray-600 text-center")
        ], className="text-center py-8")
    
    # Crear diccionario de datos del paciente
    patient_data = {
        'age': age or 70,
        'education_years': education or 14,
        'apoe4_carriers': apoe4 or 0,
        'mmse_score': mmse or 26,
        'cdr_score': cdr or 0.5,
        'tau_protein': tau or 85,
        'abeta_42': abeta or 750,
        'physical_activity': 5,  # Valores por defecto
        'social_engagement': 5
    }
    
   # Intentar predicción con modelo real
    real_prediction, model_metrics = predict_with_real_model(patient_data)

    if real_prediction is not None:
        risk_probability = real_prediction
        logger.info(f"Predicción real: {risk_probability:.3f}")
    else:
        # Fallback a cálculo simulado (solo aquí)
        risk_score = (
            (patient_data['age'] - 50) / 40 * 0.2 +
            (30 - patient_data['mmse_score']) / 30 * 0.25 +
            patient_data['cdr_score'] / 3 * 0.2 +
            (patient_data['tau_protein'] - 60) / 100 * 0.15 +
            (800 - patient_data['abeta_42']) / 400 * 0.1 +
            patient_data['apoe4_carriers'] / 2 * 0.1
        )
        risk_probability = max(0, min(1, 1 / (1 + np.exp(-max(-5, min(5, risk_score))))))
        logger.info("Usando cálculo simulado")

    risk_category = get_risk_category(risk_probability)
    risk_color = get_risk_color(risk_probability)
    
    # Generar explicaciones y recomendaciones
    explanations = generate_explanation(patient_data, risk_probability)
    recommendations = generate_recommendations(patient_data, risk_probability)
    
    return html.Div([
        # Gauge principal
        html.Div([
            dcc.Graph(figure=create_risk_gauge(risk_probability))
        ], className="mb-6"),
        
        # Resultado principal
        html.Div([
            html.Div([
                html.H4(f"Riesgo: {risk_category}", 
                       className=f"text-2xl font-bold mb-2",
                       style={'color': risk_color}),
                html.P(f"Probabilidad: {risk_probability:.1%}", className="text-lg text-gray-600"),
                html.P(f"Nivel de confianza: 85%", className="text-sm text-gray-500")
            ], className="text-center p-4 border-2 rounded-lg", style={'border-color': risk_color})
        ], className="mb-6"),
        
        # Explicaciones
        html.Div([
            html.H4("🔍 Explicación Detallada", className="text-lg font-bold text-gray-800 mb-3"),
            html.Ul([
                html.Li(exp, className="mb-2 p-2 bg-gray-50 rounded") 
                for exp in explanations[:5]
            ], className="space-y-2")
        ], className="mb-6"),
        
        # Recomendaciones
        html.Div([
            html.H4("💡 Recomendaciones Personalizadas", className="text-lg font-bold text-gray-800 mb-3"),
            html.Ul([
                html.Li(rec, className="mb-2 p-2 bg-green-50 rounded border-l-4 border-green-400") 
                for rec in recommendations[:6]
            ], className="space-y-2")
        ])
    ])

def render_risk_factors_tab():
    """Tab 2: Análisis de Factores de Riesgo"""
    return html.Div([
        html.H2("📊 Análisis Completo de Factores de Riesgo", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Gráfico de importancia de características
        html.Div([
            html.H3("🎯 Importancia de los Factores", className="text-xl font-bold text-gray-800 mb-4"),
            dcc.Graph(
                figure=px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    color='group',
                    orientation='h',
                    title="Top 10 Factores Más Importantes para la Predicción",
                    labels={'importance': 'Importancia (%)', 'feature': 'Factor'},
                    color_discrete_map=config.color_maps['feature_groups'],
                    # SOLUCIÓN: Personalizar el template del hover
                    hover_data={
                        'importance': ':.1%',  # Mostrar como porcentaje con 1 decimal
                        'group': True          # Incluir el grupo
                    }
                ).update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=True,
                    # SOLUCIÓN: Configurar el estilo del hover
                    hoverlabel=dict(
                        bgcolor="white",           # Fondo blanco
                        font_size=14,             # Tamaño de fuente legible
                        font_family="Inter",      # Fuente consistente
                        font_color="black",       # Texto negro para contraste
                        bordercolor="gray",       # Borde gris
                        namelength=-1            # Mostrar nombres completos
                    )
                ).update_traces(
                    # SOLUCIÓN: Template personalizado para el hover
                    hovertemplate="<b>%{y}</b><br>" +
                                "Importancia: %{x:.1%}<br>" +
                                "Grupo: %{customdata[0]}<br>" +
                                "<extra></extra>",  # Elimina el box del nombre de la serie
                    customdata=feature_importance.head(10)[['group']].values
                )
            )
        ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
        
        # Análisis por grupos
        html.Div([
            html.H3("📋 Análisis por Grupos de Factores", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                # Biomarcadores
                html.Div([
                    html.H4("🧪 Biomarcadores", className="text-lg font-bold text-red-600 mb-2"),
                    html.P("Proteínas y moléculas que indican daño cerebral", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Proteína Tau: Indica daño en neuronas", className="mb-1"),
                        html.Li("• Beta-amiloide 42: Forma placas tóxicas", className="mb-1"),
                        html.Li("• pTau-181: Marcador específico de Alzheimer", className="mb-1"),
                        html.Li("• NFL: Indica daño en axones", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-red-50 border border-red-200 rounded-lg p-4"),
                
                # Cognitivo
                html.Div([
                    html.H4("🧠 Evaluaciones Cognitivas", className="text-lg font-bold text-yellow-600 mb-2"),
                    html.P("Pruebas que miden el funcionamiento mental", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• MMSE: Evaluación cognitiva básica (0-30)", className="mb-1"),
                        html.Li("• CDR: Escala de demencia clínica (0-3)", className="mb-1"),
                        html.Li("• ADAS-Cog: Evaluación detallada de Alzheimer", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-yellow-50 border border-yellow-200 rounded-lg p-4"),
                
                # Neuroimagen
                html.Div([
                    html.H4("🔍 Neuroimagen", className="text-lg font-bold text-green-600 mb-2"),
                    html.P("Medidas del cerebro por resonancia magnética", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Volumen del hipocampo: Área de la memoria", className="mb-1"),
                        html.Li("• Corteza entorrinal: Primera área afectada", className="mb-1"),
                        html.Li("• Volumen cerebral total: Atrofia general", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-green-50 border border-green-200 rounded-lg p-4"),
                
                # Estilo de vida
                html.Div([
                    html.H4("🏃‍♀️ Estilo de Vida", className="text-lg font-bold text-blue-600 mb-2"),
                    html.P("Factores modificables que puedes controlar", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Actividad física: Ejercicio regular", className="mb-1"),
                        html.Li("• Compromiso social: Interacción con otros", className="mb-1"),
                        html.Li("• Calidad del sueño: Descanso reparador", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-blue-50 border border-blue-200 rounded-lg p-4")
                
            ], className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6")
        ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
        
        # Factores modificables vs no modificables
        html.Div([
            html.H3("⚖️ Factores Modificables vs No Modificables", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("❌ No Modificables", className="text-lg font-bold text-red-600 mb-3"),
                    html.Div([
                        html.Div("Edad", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Genética (APOE4)", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Sexo", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Antecedentes familiares", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg")
                    ])
                ], className="bg-white border border-red-200 rounded-lg p-6"),
                
                html.Div([
                    html.H4("✅ Modificables", className="text-lg font-bold text-green-600 mb-3"),
                    html.Div([
                        html.Div("Actividad física", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Dieta mediterránea", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Estimulación cognitiva", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Control de factores cardiovasculares", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Calidad del sueño", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Vida social activa", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg")
                    ])
                ], className="bg-white border border-green-200 rounded-lg p-6")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-6")
        ], className="bg-gray-50 rounded-lg p-6")
    ])

def render_case_examples_tab():
    """Tab 3: Casos Educativos"""
    # Generar casos ejemplo
    case_examples = [
        {
            'id': 'caso_1',
            'name': 'María García, 68 años',
            'risk': 'Bajo',
            'probability': 0.15,
            'color': config.colors['low_risk'],
            'profile': {
                'age': 68,
                'education': 16,
                'mmse': 29,
                'tau': 45,
                'abeta': 950,
                'apoe4': 0,
                'activity': 8
            },
            'description': 'Profesora jubilada con alta educación y estilo de vida saludable',
            'factors': [
                'Alto nivel educativo (16 años)',
                'Excelente puntuación cognitiva (MMSE: 29/30)',
                'Biomarcadores dentro de rango normal',
                'Sin factor genético de riesgo',
                'Muy activa físicamente'
            ],
            'recommendations': [
                'Continuar con actividad física regular',
                'Mantener estimulación cognitiva',
                'Seguimiento anual de rutina'
            ]
        },
        {
            'id': 'caso_2',
            'name': 'Carlos Rodríguez, 74 años',
            'risk': 'Moderado',
            'probability': 0.55,
            'color': config.colors['moderate_risk'],
            'profile': {
                'age': 74,
                'education': 10,
                'mmse': 25,
                'tau': 95,
                'abeta': 620,
                'apoe4': 1,
                'activity': 4
            },
            'description': 'Jubilado con algunos factores de riesgo emergentes',
            'factors': [
                'Edad avanzada (74 años)',
                'Biomarcadores ligeramente alterados',
                'Una copia del gen APOE4',
                'Leve deterioro cognitivo (MMSE: 25/30)',
                'Actividad física limitada'
            ],
            'recommendations': [
                'Evaluación neurológica semestral',
                'Incrementar actividad física gradualmente',
                'Estimulación cognitiva estructurada',
                'Monitoreo de biomarcadores'
            ]
        },
        {
            'id': 'caso_3',
            'name': 'Ana Martínez, 79 años',
            'risk': 'Alto',
            'probability': 0.85,
            'color': config.colors['high_risk'],
            'profile': {
                'age': 79,
                'education': 6,
                'mmse': 21,
                'tau': 150,
                'abeta': 420,
                'apoe4': 2,
                'activity': 2
            },
            'description': 'Múltiples factores de riesgo requieren atención especializada',
            'factors': [
                'Edad muy avanzada (79 años)',
                'Deterioro cognitivo significativo (MMSE: 21/30)',
                'Biomarcadores muy alterados',
                'Dos copias del gen APOE4',
                'Muy baja actividad física'
            ],
            'recommendations': [
                'Consulta neurológica especializada urgente',
                'Evaluación integral de biomarcadores',
                'Plan de intervención multidisciplinario',
                'Soporte familiar y cuidador'
            ]
        }
    ]
    
    return html.Div([
        html.H2("👥 Casos Educativos Interactivos", className="text-2xl font-bold text-gray-800 mb-6"),
        
        html.Div([
            html.I(className="fas fa-info-circle text-blue-500 mr-2"),
            "Estos casos son ejemplos educativos basados en perfiles reales anonimizados. "
            "Cada caso muestra cómo diferentes factores influyen en el riesgo de Alzheimer."
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Caso 1: Riesgo Bajo
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[0]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[0]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[0]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[0]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[0]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[0]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': hex_to_rgba(config.colors['low_risk'], 0.3)},
                                        {'range': [30, 70], 'color': hex_to_rgba(config.colors['moderate_risk'], 0.3)},
                                        {'range': [70, 100], 'color': hex_to_rgba(config.colors['high_risk'], 0.3)}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("🔍 Factores Clave:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"✓ {factor}", className="text-green-700 mb-1") 
                        for factor in case_examples[0]['factors']
                    ], className="mb-4"),
                    
                    html.H4("💡 Recomendaciones:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-blue-700 mb-1") 
                        for rec in case_examples[0]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Caso 2: Riesgo Moderado
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[1]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[1]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[1]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[1]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[1]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[1]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': hex_to_rgba(config.colors['low_risk'], 0.3)},
                                        {'range': [30, 70], 'color': hex_to_rgba(config.colors['moderate_risk'], 0.3)},
                                        {'range': [70, 100], 'color': hex_to_rgba(config.colors['high_risk'], 0.3)}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("⚠️ Factores de Preocupación:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"• {factor}", className="text-yellow-700 mb-1") 
                        for factor in case_examples[1]['factors']
                    ], className="mb-4"),
                    
                    html.H4("🎯 Plan de Acción:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-orange-700 mb-1") 
                        for rec in case_examples[1]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Caso 3: Riesgo Alto
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[2]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[2]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[2]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[2]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[2]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[2]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': hex_to_rgba(config.colors['low_risk'], 0.3)},
                                        {'range': [30, 70], 'color': hex_to_rgba(config.colors['moderate_risk'], 0.3)},
                                        {'range': [70, 100], 'color': hex_to_rgba(config.colors['high_risk'], 0.3)}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("🚨 Factores de Alto Riesgo:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"⚠️ {factor}", className="text-red-700 mb-1") 
                        for factor in case_examples[2]['factors']
                    ], className="mb-4"),
                    
                    html.H4("🏥 Intervención Urgente:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-red-700 mb-1") 
                        for rec in case_examples[2]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6"),
        
        # Lecciones aprendidas
        html.Div([
            html.H3("📚 Lecciones Clave de estos Casos", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("🎓 Educación como Protección", className="text-lg font-semibold text-green-600 mb-2"),
                    html.P("Mayor nivel educativo puede proporcionar 'reserva cognitiva' que ayuda a resistir el deterioro cerebral.", 
                           className="text-gray-600")
                ], className="bg-green-50 border border-green-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("🧬 Genética no es Destino", className="text-lg font-semibold text-blue-600 mb-2"),
                    html.P("Aunque el APOE4 aumenta el riesgo, el estilo de vida y otros factores siguen siendo cruciales.", 
                           className="text-gray-600")
                ], className="bg-blue-50 border border-blue-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("⏰ Detección Temprana", className="text-lg font-semibold text-yellow-600 mb-2"),
                    html.P("La intervención temprana en casos de riesgo moderado puede prevenir o retrasar la progresión.", 
                           className="text-gray-600")
                ], className="bg-yellow-50 border border-yellow-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("🏃‍♀️ Nunca es Tarde", className="text-lg font-semibold text-purple-600 mb-2"),
                    html.P("Incluso en casos de alto riesgo, los cambios en el estilo de vida pueden tener beneficios.", 
                           className="text-gray-600")
                ], className="bg-purple-50 border border-purple-200 rounded-lg p-4")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

def render_monitoring_tab():
    """Tab 4: Monitoreo y Tendencias"""
    # Generar datos de monitoreo
    dates = pd.date_range(start='2024-01-01', end='2025-08-26', freq='W')
    monitoring_data = pd.DataFrame({
        'date': dates,
        'total_patients': np.cumsum(np.random.poisson(10, len(dates))) + 100,
        'high_risk': np.cumsum(np.random.poisson(2, len(dates))) + 20,
        'moderate_risk': np.cumsum(np.random.poisson(3, len(dates))) + 30,
        'low_risk': np.cumsum(np.random.poisson(5, len(dates))) + 50,
        'model_accuracy': 0.87 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001
    })
    monitoring_data['model_accuracy'] = monitoring_data['model_accuracy'].clip(0.82, 0.92)
    
    return html.Div([
        html.H2("📈 Monitoreo del Sistema y Tendencias", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Métricas principales
        html.Div([
            # Card 1: Total de pacientes
            html.Div([
                html.Div([
                    html.I(className="fas fa-users text-3xl text-blue-500 mb-2"),
                    html.H3(f"{monitoring_data['total_patients'].iloc[-1]:,}", className="text-2xl font-bold text-gray-800"),
                    html.P("Total Pacientes", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-arrow-up text-green-500 mr-1"),
                        f"+{monitoring_data['total_patients'].iloc[-1] - monitoring_data['total_patients'].iloc[-8]:,} esta semana"
                    ], className="text-sm text-green-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 2: Precisión del modelo
            html.Div([
                html.Div([
                    html.I(className="fas fa-bullseye text-3xl text-green-500 mb-2"),
                    html.H3(f"{monitoring_data['model_accuracy'].iloc[-1]:.1%}", className="text-2xl font-bold text-gray-800"),
                    html.P("Precisión del Modelo", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-check-circle text-green-500 mr-1"),
                        "Rendimiento óptimo"
                    ], className="text-sm text-green-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 3: Casos de alto riesgo
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-3xl text-red-500 mb-2"),
                    html.H3(f"{monitoring_data['high_risk'].iloc[-1]:,}", className="text-2xl font-bold text-gray-800"),
                    html.P("Alto Riesgo", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-eye text-blue-500 mr-1"),
                        "Requieren seguimiento"
                    ], className="text-sm text-blue-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 4: Alertas activas
            html.Div([
                html.Div([
                    html.I(className="fas fa-bell text-3xl text-yellow-500 mb-2"),
                    html.H3("12", className="text-2xl font-bold text-gray-800"),
                    html.P("Alertas Activas", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-clock text-yellow-500 mr-1"),
                        "Pendientes de revisión"
                    ], className="text-sm text-yellow-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center")
            
        ], className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6"),
        
        # Gráficos de tendencias
        html.Div([
            html.Div([
                html.H3("📊 Evolución de Pacientes por Categoría de Riesgo", className="text-xl font-bold text-gray-800 mb-4"),
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['low_risk'], 
                                 mode='lines+markers', name='Riesgo Bajo', 
                                 line=dict(color=config.colors['low_risk'], width=3),
                                 marker=dict(size=6)),
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['moderate_risk'], 
                                 mode='lines+markers', name='Riesgo Moderado',
                                 line=dict(color=config.colors['moderate_risk'], width=3),
                                 marker=dict(size=6)),
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['high_risk'], 
                                 mode='lines+markers', name='Riesgo Alto',
                                 line=dict(color=config.colors['high_risk'], width=3),
                                 marker=dict(size=6))
                    ]).update_layout(
                        title="Tendencia de Casos por Nivel de Riesgo",
                        xaxis_title="Fecha",
                        yaxis_title="Número de Pacientes",
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                )
            ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
            
            html.Div([
                html.H3("🎯 Rendimiento del Modelo en el Tiempo", className="text-xl font-bold text-gray-800 mb-4"),
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['model_accuracy'], 
                                 mode='lines+markers', name='Precisión',
                                 line=dict(color=config.colors['info'], width=3),
                                 marker=dict(size=6),
                                 fill='tonexty'),
                        go.Scatter(x=monitoring_data['date'], y=[0.85]*len(monitoring_data), 
                                 mode='lines', name='Umbral Mínimo',
                                 line=dict(color='red', width=2, dash='dash'))
                    ]).update_layout(
                        title="Evolución de la Precisión del Modelo",
                        xaxis_title="Fecha",
                        yaxis_title="Precisión (%)",
                        yaxis=dict(tickformat='.1%', range=[0.8, 0.95]),
                        height=400,
                        hovermode='x unified'
                    )
                )
            ], className="bg-white rounded-lg shadow-md p-6")
        ]),
        
        # Estadísticas por subgrupos
        html.Div([
            html.H3("👥 Rendimiento por Subgrupos Poblacionales", className="text-xl font-bold text-gray-800 mb-4"),
            
            html.Div([
                # Por edad
                html.Div([
                    html.H4("🎂 Por Grupo de Edad", className="text-lg font-semibold text-gray-700 mb-3"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Grupo de Edad", "id": "age_group"},
                            {"name": "N° Pacientes", "id": "count"},
                            {"name": "Precisión", "id": "accuracy", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Sensibilidad", "id": "sensitivity", "type": "numeric", "format": {"specifier": ".1%"}}
                        ],
                        data=[
                            {"age_group": "50-64 años", "count": 245, "accuracy": 0.89, "sensitivity": 0.91},
                            {"age_group": "65-74 años", "count": 412, "accuracy": 0.87, "sensitivity": 0.88},
                            {"age_group": "75-84 años", "count": 278, "accuracy": 0.85, "sensitivity": 0.86},
                            {"age_group": "85+ años", "count": 65, "accuracy": 0.83, "sensitivity": 0.84}
                        ],
                        style_cell={'textAlign': 'center', 'fontSize': '14px'},
                        style_header={'backgroundColor': config.colors['info'], 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], className="bg-white rounded-lg border p-4"),
                
                # Por género
                html.Div([
                    html.H4("⚧️ Por Género", className="text-lg font-semibold text-gray-700 mb-3"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Género", "id": "gender"},
                            {"name": "N° Pacientes", "id": "count"},
                            {"name": "Precisión", "id": "accuracy", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Especificidad", "id": "specificity", "type": "numeric", "format": {"specifier": ".1%"}}
                        ],
                        data=[
                            {"gender": "Mujeres", "count": 580, "accuracy": 0.86, "specificity": 0.89},
                            {"gender": "Hombres", "count": 420, "accuracy": 0.88, "specificity": 0.87}
                        ],
                        style_cell={'textAlign': 'center', 'fontSize': '14px'},
                        style_header={'backgroundColor': config.colors['info'], 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], className="bg-white rounded-lg border p-4")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
        ], className="bg-gray-50 rounded-lg p-6 mt-6"),
        
        # Sistema de alertas
        html.Div([
            html.H3("🚨 Sistema de Alertas Activo", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                # Alerta crítica
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle text-red-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Pacientes de Alto Riesgo Sin Seguimiento", className="text-lg font-semibold text-red-600"),
                            html.P("8 pacientes clasificados como alto riesgo no tienen cita programada", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Revisar", className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg mb-3"),
                
                # Alerta moderada
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-yellow-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Biomarcadores Pendientes", className="text-lg font-semibold text-yellow-600"),
                            html.P("15 pacientes tienen resultados de biomarcadores pendientes de análisis", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Ver Lista", className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded-r-lg mb-3"),
                
                # Alerta informativa
                html.Div([
                    html.Div([
                        html.I(className="fas fa-info-circle text-blue-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Actualización de Modelo Disponible", className="text-lg font-semibold text-blue-600"),
                            html.P("Nueva versión del modelo con mejoras en precisión disponible para implementar", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Más Info", className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg")
            ])
        ], className="bg-white rounded-lg shadow-md p-6 mt-6")
    ])

def render_learning_center_tab():
    """Tab 5: Centro de Aprendizaje"""
    return html.Div([
        html.H2("🧠 Centro de Aprendizaje sobre Alzheimer", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Navegación del centro de aprendizaje
        html.Div([
            dcc.Tabs(id="learning-tabs", value="glossary", children=[
                dcc.Tab(label="📚 Glosario Médico", value="glossary"),
                dcc.Tab(label="🤖 Cómo Funciona el Modelo", value="model-explanation"),
                dcc.Tab(label="❓ Preguntas Frecuentes", value="faq"),
                dcc.Tab(label="🔗 Recursos Adicionales", value="resources")
            ], className="mb-4"),
            
            html.Div(id="learning-content", children=[
                # Contenido por defecto
                render_glossary()
            ])
        ])
    ])

@app.callback(
    Output("learning-content", "children"),
    Input("learning-tabs", "value")
)
def render_learning_content(active_tab):
    """Renderiza el contenido del centro de aprendizaje"""
    if active_tab == "glossary":
        return render_glossary()
    elif active_tab == "model-explanation":
        return render_model_explanation()
    elif active_tab == "faq":
        return render_faq()
    elif active_tab == "resources":
        return render_resources()
    else:
        return render_glossary()   # Por defecto

def render_glossary():
    """Glosario médico interactivo"""
    glossary_terms = [
        {
            "term": "Alzheimer",
            "definition": "Enfermedad neurodegenerativa progresiva que es la causa más común de demencia. Afecta principalmente la memoria, el pensamiento y el comportamiento.",
            "icon": "fas fa-brain",
            "color": "red"
        },
        {
            "term": "Beta-amiloide (Aβ42)",
            "definition": "Proteína que se acumula formando placas en el cerebro de personas con Alzheimer. Los niveles bajos en líquido cefalorraquídeo pueden indicar la presencia de placas cerebrales.",
            "icon": "fas fa-microscope",
            "color": "blue"
        },
        {
            "term": "Proteína Tau",
            "definition": "Proteína que normalmente estabiliza las estructuras celulares. En el Alzheimer, se vuelve anormal y forma ovillos neurofibrilares que dañan las neuronas.",
            "icon": "fas fa-dna",
            "color": "green"
        },
        {
            "term": "MMSE (Mini-Mental State Exam)",
            "definition": "Prueba cognitiva breve que evalúa orientación, memoria, atención y habilidades del lenguaje. Puntuación de 0-30, donde puntuaciones más bajas indican mayor deterioro.",
            "icon": "fas fa-clipboard-check",
            "color": "purple"
        },
        {
            "term": "CDR (Clinical Dementia Rating)",
            "definition": "Escala que evalúa el grado de deterioro cognitivo en 6 áreas. Va de 0 (normal) a 3 (demencia severa).",
            "icon": "fas fa-chart-line",
            "color": "yellow"
        },
        {
            "term": "APOE4",
            "definition": "Variante del gen APOE que aumenta el riesgo de desarrollar Alzheimer. Tener una copia aumenta el riesgo 2-3 veces, dos copias lo aumentan 8-12 veces.",
            "icon": "fas fa-dna",
            "color": "red"
        },
        {
            "term": "Hipocampo",
            "definition": "Región cerebral crucial para la formación de nuevas memorias. Es una de las primeras áreas afectadas en el Alzheimer.",
            "icon": "fas fa-brain",
            "color": "blue"
        },
        {
            "term": "Biomarcadores",
            "definition": "Indicadores biológicos medibles que reflejan procesos normales, patológicos o respuestas a tratamientos. En Alzheimer incluyen proteínas en sangre y líquido cefalorraquídeo.",
            "icon": "fas fa-flask",
            "color": "green"
        },
        {
            "term": "Reserva Cognitiva",
            "definition": "Capacidad del cerebro para mantener el funcionamiento normal a pesar del daño. Se asocia con mayor educación, actividad mental y social.",
            "icon": "fas fa-shield-alt",
            "color": "blue"
        },
        {
            "term": "Deterioro Cognitivo Leve (DCL)",
            "definition": "Condición intermedia entre el envejecimiento normal y la demencia. Cambios cognitivos notables pero que no interfieren significativamente con la vida diaria.",
            "icon": "fas fa-battery-half",
            "color": "yellow"
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-book text-blue-500 mr-2"),
            "Glosario de términos médicos explicados en lenguaje claro y comprensible para todos"
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className=f"{term['icon']} text-2xl text-{term['color']}-500 mr-4"),
                        html.Div([
                            html.H3(term['term'], className="text-xl font-bold text-gray-800 mb-2"),
                            html.P(term['definition'], className="text-gray-600 leading-relaxed")
                        ], className="flex-1")
                    ], className="flex items-start")
                ], className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-200")
            ]) for term in glossary_terms
        ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
    ])

def render_model_explanation():
    """Explicación del modelo en términos no técnicos"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-robot text-purple-500 mr-2"),
            "Entenda cómo funciona nuestro sistema de predicción de riesgo de Alzheimer"
        ], className="bg-purple-50 border-l-4 border-purple-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Paso 1
            html.Div([
                html.Div([
                    html.Div("1", className="bg-blue-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Recolección de Datos", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema recopila información de múltiples fuentes: ",
                        html.Strong("análisis de sangre"), ", ",
                        html.Strong("resonancias magnéticas"), ", ",
                        html.Strong("evaluaciones cognitivas"), " y ",
                        html.Strong("datos demográficos"), "."
                    ], className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("🩺 Biomarcadores en sangre y LCR", className="mb-1"),
                        html.Li("🧠 Imágenes del cerebro (RM)", className="mb-1"),
                        html.Li("📝 Pruebas de memoria y cognición", className="mb-1"),
                        html.Li("👤 Edad, educación, genética", className="mb-1")
                    ], className="text-sm text-gray-700")
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 2
            html.Div([
                html.Div([
                    html.Div("2", className="bg-green-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Análisis Inteligente", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "Un modelo de ",
                        html.Strong("inteligencia artificial"), 
                        " entrenado con datos de miles de pacientes analiza todos estos factores simultaneamente para identificar patrones complejos."
                    ], className="text-gray-600 mb-3"),
                    html.Div([
                        html.P("El modelo considera:", className="font-semibold text-gray-700 mb-2"),
                        html.Ul([
                            html.Li("🔄 Interacciones entre factores", className="mb-1"),
                            html.Li("⚖️ Peso relativo de cada variable", className="mb-1"),
                            html.Li("📊 Patrones en datos históricos", className="mb-1"),
                            html.Li("🎯 Umbrales clínicos validados", className="mb-1")
                        ], className="text-sm text-gray-700")
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 3
            html.Div([
                html.Div([
                    html.Div("3", className="bg-yellow-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Cálculo de Riesgo", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema calcula una ",
                        html.Strong("probabilidad de riesgo"), 
                        " entre 0% y 100%, que se traduce en categorías comprensibles: Bajo, Moderado o Alto riesgo."
                    ], className="text-gray-600 mb-3"),
                    html.Div([
                        html.Div("0-30%", className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Bajo", className="text-gray-600"),
                        html.Br(),
                        html.Div("30-70%", className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Moderado", className="text-gray-600"),
                        html.Br(),
                        html.Div("70-100%", className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Alto", className="text-gray-600")
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 4
            html.Div([
                html.Div([
                    html.Div("4", className="bg-purple-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Explicación y Recomendaciones", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema no solo da un número, sino que ",
                        html.Strong("explica por qué"), 
                        " y ",
                        html.Strong("qué se puede hacer"), 
                        ". Genera explicaciones personalizadas y recomendaciones específicas."
                    ], className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("🔍 Factores que más influyen en el riesgo", className="mb-1"),
                        html.Li("💡 Recomendaciones médicas específicas", className="mb-1"),
                        html.Li("🏃‍♀️ Cambios de estilo de vida sugeridos", className="mb-1"),
                        html.Li("📅 Cronograma de seguimiento", className="mb-1")
                    ], className="text-sm text-gray-700")
                ])
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6"),
        
        # Confiabilidad del modelo
        html.Div([
            html.H3("🎯 ¿Qué tan confiable es el modelo?", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("87% de Precisión", className="text-2xl font-bold text-green-600 mb-2"),
                    html.P("El modelo acierta en 87 de cada 100 predicciones", className="text-gray-600")
                ], className="text-center p-4 flex flex-col h-full justify-between"),

                html.Div([
                    html.H4("1,000+ Pacientes", className="text-2xl font-bold text-blue-600 mb-2"),
                    html.P("Entrenado con datos de más de 1,000 pacientes reales", className="text-gray-600")
                ], className="text-center p-4 flex flex-col h-full justify-between"),

                html.Div([
                    html.H4("6 Años de Datos", className="text-2xl font-bold text-purple-600 mb-2"),
                    html.P("Utiliza información longitudinal de seguimiento", className="text-gray-600")
                ], className="text-center p-4 flex flex-col h-full justify-between")

            ], className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 bg-gray-50 rounded-lg p-6 items-stretch")
        ], className="bg-white rounded-lg shadow-md p-6")

    ])

def render_faq():
    """Preguntas frecuentes"""
    faqs = [
        {
            "question": "¿Qué significa tener alto riesgo de Alzheimer?",
            "answer": "Tener alto riesgo NO significa que definitivamente desarrollarás Alzheimer. Significa que, basado en factores conocidos, tienes una probabilidad mayor que el promedio. Muchas personas con alto riesgo nunca desarrollan la enfermedad, especialmente si toman medidas preventivas."
        },
        {
            "question": "¿Puedo cambiar mi riesgo de Alzheimer?",
            "answer": "¡Sí! Aunque factores como la edad y la genética no se pueden cambiar, muchos otros sí: ejercicio regular, dieta saludable, actividad social, estimulación mental, buen sueño y control de factores cardiovasculares pueden reducir significativamente el riesgo."
        },
        {
            "question": "¿Con qué frecuencia debo hacerme la evaluación?",
            "answer": "Depende de tu nivel de riesgo: Riesgo bajo: cada 2-3 años. Riesgo moderado: cada año. Riesgo alto: cada 6 meses o según indicación médica. Tu doctor puede recomendar una frecuencia diferente basada en tu situación específica."
        },
        {
            "question": "¿Los biomarcadores son necesarios para la evaluación?",
            "answer": "No son estrictamente necesarios. El modelo puede hacer predicciones útiles solo con evaluaciones cognitivas y datos demográficos. Sin embargo, los biomarcadores proporcionan información más precisa y detallada sobre el estado del cerebro."
        },
        {
            "question": "¿Qué tan temprano se puede detectar el riesgo?",
            "answer": "Los cambios cerebrales pueden comenzar 10-20 años antes de los síntomas. Nuestro modelo puede detectar riesgo elevado en etapas muy tempranas, cuando las intervenciones son más efectivas para prevenir o retrasar la progresión."
        },
        {
            "question": "¿El resultado puede estar equivocado?",
            "answer": "Ningún test es 100% perfecto. Nuestro modelo tiene 87% de precisión, lo que significa que puede haber falsos positivos o negativos. Por eso es importante: 1) Usar múltiples evaluaciones, 2) Considerar el contexto clínico completo, 3) Seguimiento regular."
        },
        {
            "question": "¿Debo preocuparme si tengo el gen APOE4?",
            "answer": "Tener APOE4 aumenta el riesgo, pero NO garantiza que desarrollarás Alzheimer. Muchas personas con este gen nunca desarrollan la enfermedad. Es información útil para tomar medidas preventivas más agresivas."
        },
        {
            "question": "¿Los síntomas de memoria normal del envejecimiento son preocupantes?",
            "answer": "Es normal olvidar ocasionalmente nombres o donde pusiste las llaves. Preocúpate si: olvidas conversaciones recientes, te pierdes en lugares familiares, tienes dificultad con tareas habituales, o familiares notan cambios significativos."
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-question-circle text-green-500 mr-2"),
            "Respuestas a las preguntas más comunes sobre Alzheimer y evaluación de riesgo"
        ], className="bg-green-50 border-l-4 border-green-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.Details([
                    html.Summary(faq['question'], className="text-lg font-semibold text-gray-800 cursor-pointer hover:text-blue-600 py-3"),
                    html.Div([
                        html.P(faq['answer'], className="text-gray-600 leading-relaxed mt-2 pl-4")
                    ])
                ], className="border-b border-gray-200")
            ]) for faq in faqs
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

def render_resources():
    """Recursos adicionales específicos para Argentina"""
    resources = [
        {
            "category": "Organizaciones Oficiales",
            "items": [
                {
                    "name": "ADNI - Alzheimer's Disease Neuroimaging Initiative",
                    "url": "https://adni.loni.usc.edu/",
                    "description": "Iniciativa internacional líder en investigación de biomarcadores para Alzheimer. Argentina participa como Arg-ADNI desde 2013, siendo el primer centro ADNI en Latinoamérica."
                },
                {
                    "name": "A.L.M.A. - Asociación Lucha contra el Mal de Alzheimer",
                    "url": "https://www.alma-alzheimer.org.ar/",
                    "description": "Fundada en 1989, es la principal organización argentina dedicada al estudio, apoyo y tratamiento del Alzheimer y alteraciones cognitivas similares."
                },
                {
                    "name": "Alzheimer Argentina",
                    "url": "https://alzheimer.org.ar/",
                    "description": "Asociación enfocada en investigación, docencia y tratamiento del Alzheimer. Promueve la prevención, diagnóstico diferencial y estudios de tratamiento."
                },
                {
                    "name": "Programa Nacional de Deterioro Cognitivo - Ministerio de Desarrollo Social",
                    "url": "https://www.argentina.gob.ar/desarrollosocial/programas/deteriorocognitivo",
                    "description": "Programa gubernamental para promoción de calidad de vida de personas con deterioro cognitivo, Alzheimer y otras demencias."
                },
                {
                    "name": "Alzheimer's Association - Sección Argentina",
                    "url": "https://www.alz.org/ar/demencia-alzheimer-argentina.asp",
                    "description": "Brazo argentino de la organización mundial líder en cuidado, apoyo e investigación del Alzheimer con recursos en español."
                }
            ]
        },
        {
            "category": "Prevención y Estilo de Vida",
            "items": [
                {
                    "name": "Instituto de Neurología Cognitiva (INECO)",
                    "url": "https://www.ineco.org.ar/",
                    "description": "Centro líder en Argentina para evaluación neurocognitiva, programas de estimulación cognitiva y estrategias de prevención del deterioro mental."
                },
                {
                    "name": "Fundación INECO para la Investigación en Neurociencias Cognitivas",
                    "url": "https://www.ineco.org.ar/fundacion/",
                    "description": "Programas de investigación en prevención del Alzheimer, entrenamiento cognitivo y promoción de salud cerebral en población argentina."
                },
                {
                    "name": "Centro de Memoria del Hospital Italiano",
                    "url": "https://www.hospitalitaliano.org.ar/",
                    "description": "Programas especializados en evaluación de memoria, detección temprana y estrategias preventivas del deterioro cognitivo."
                },
                {
                    "name": "Neurología Cognitiva - Hospital de Clínicas",
                    "url": "https://www.hospitaldeclinicas.uba.ar/",
                    "description": "Servicio universitario público que ofrece evaluación neuropsicológica, programas de estimulación cognitiva y orientación en prevención."
                },
                {
                    "name": "Programa de Actividad Física para Adultos Mayores - Buenos Aires Ciudad",
                    "url": "https://www.buenosaires.gob.ar/desarrollohumanoyhabitat/adultos-mayores",
                    "description": "Actividades físicas estructuradas para adultos mayores como factor protector contra el deterioro cognitivo."
                }
            ]
        },
        {
            "category": "Apoyo y Cuidadores",
            "items": [
                {
                    "name": "Grupos de Apoyo A.L.M.A.",
                    "url": "https://www.alma-alzheimer.org.ar/es/servicios-alma/grupos-de-apoyo",
                    "description": "Grupos de apoyo presenciales y virtuales para familiares y cuidadores de personas con Alzheimer. Reuniones regulares en CABA y GBA."
                },
                {
                    "name": "Red Argentina de Cuidadores Domiciliarios",
                    "url": "https://www.argentina.gob.ar/desarrollosocial/programas/deteriorocognitivo",
                    "description": "Capacitación y formación de cuidadores especializados en demencia. Programa gubernamental de apoyo y certificación."
                },
                {
                    "name": "Fundación Navarro Viola - Programa Alzheimer",
                    "url": "https://www.navarroviola.org/",
                    "description": "Apoyo integral a familias afectadas por Alzheimer, incluyendo asistencia social, psicológica y recursos comunitarios."
                },
                {
                    "name": "Centro de Día para Adultos Mayores - CABA",
                    "url": "https://www.buenosaires.gob.ar/desarrollohumanoyhabitat/adultos-mayores",
                    "description": "Centros diurnos especializados en cuidado de personas con deterioro cognitivo, ofreciendo respiro a cuidadores familiares."
                },
                {
                    "name": "Línea de Asistencia Psicológica - Gobierno de la Ciudad",
                    "url": "tel:147",
                    "description": "Línea telefónica gratuita 147 disponible 24/7 para orientación y apoyo psicológico a familiares y cuidadores en crisis."
                }
            ]
        },
        {
            "category": "Investigación y Tratamientos",
            "items": [
                {
                    "name": "Arg-ADNI - Instituto de Neurociencias FLENI",
                    "url": "https://www.fleni.org.ar/",
                    "description": "Primer centro ADNI de Latinoamérica ubicado en Buenos Aires. Líder en investigación de biomarcadores y ensayos clínicos para Alzheimer."
                },
                {
                    "name": "Centro de Investigaciones en Neurociencias de Córdoba",
                    "url": "https://www.unc.edu.ar/",
                    "description": "Investigación universitaria en neurociencias cognitivas, estudios longitudinales de envejecimiento y desarrollo de biomarcadores."
                },
                {
                    "name": "Registro Nacional de Ensayos Clínicos - ANMAT",
                    "url": "https://www.argentina.gob.ar/anmat",
                    "description": "Base de datos oficial de ensayos clínicos en curso en Argentina, incluyendo estudios sobre Alzheimer y tratamientos experimentales."
                },
                {
                    "name": "Red Latinoamericana de Investigación en Demencia (ReDLat)",
                    "url": "https://redlat.org/",
                    "description": "Consorcio regional que incluye centros argentinos para investigación colaborativa en demencia y desarrollo de políticas públicas."
                },
                {
                    "name": "Instituto de Neurología Cognitiva (INECO) - Investigación",
                    "url": "https://www.ineco.org.ar/investigacion/",
                    "description": "Centro pionero en investigación sobre demencia frontotemporal, Alzheimer y desarrollo de herramientas diagnósticas innovadoras."
                },
                {
                    "name": "ClinicalTrials.gov - Estudios en Argentina",
                    "url": "https://clinicaltrials.gov/search?locn=Argentina&cond=Alzheimer%20Disease",
                    "description": "Base de datos internacional filtrada para ensayos clínicos de Alzheimer activos en Argentina."
                }
            ]
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-external-link-alt text-indigo-500 mr-2"),
            "Enlaces útiles a recursos argentinos confiables sobre Alzheimer, prevención y apoyo en español"
        ], className="bg-indigo-50 border-l-4 border-indigo-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.H3(category['category'], className="text-xl font-bold text-gray-800 mb-4"),
                html.Div([
                    html.Div([
                        html.H4(item['name'], className="text-lg font-semibold text-blue-600 mb-2"),
                        html.P(item['description'], className="text-gray-600 text-sm mb-3"),
                        html.A("Visitar →" if not item['url'].startswith('tel:') else "Llamar →", 
                              href=item['url'], target="_blank", 
                              className="text-blue-500 hover:text-blue-700 font-medium text-sm")
                    ], className="bg-gray-50 rounded-lg p-4 hover:shadow-md transition-shadow duration-200")
                    for item in category['items']
                ], className="space-y-3")
            ], className="bg-white rounded-lg shadow-md p-6")
            for category in resources
        ], className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6"),
        
        # Contactos de emergencia específicos para Argentina
        html.Div([
            html.H3("🚨 Contactos de Emergencia en Argentina", className="text-xl font-bold text-red-600 mb-4"),
            html.Div([
                # Emergencias médicas
                html.Div([
                    html.I(className="fas fa-phone text-red-500 text-2xl mr-4"),
                    html.Div([
                        html.H4("Emergencias Médicas SAME", className="text-lg font-semibold text-gray-800"),
                        html.P("107", className="text-xl font-bold text-red-600"),
                        html.P("Servicio de emergencias médicas gratuito 24/7", className="text-gray-600 text-sm")
                    ])
                ], className="flex items-center p-4 bg-red-50 rounded-lg border border-red-200 mb-3"),
                
                # Línea psicológica
                html.Div([
                    html.I(className="fas fa-heart text-blue-500 text-2xl mr-4"),
                    html.Div([
                        html.H4("Asistencia Psicológica", className="text-lg font-semibold text-gray-800"),
                        html.P("147", className="text-xl font-bold text-blue-600"),
                        html.P("Línea gratuita de apoyo psicológico - Ciudad de Buenos Aires", className="text-gray-600 text-sm")
                    ])
                ], className="flex items-center p-4 bg-blue-50 rounded-lg border border-blue-200 mb-3"),
                
                # A.L.M.A. contacto directo
                html.Div([
                    html.I(className="fas fa-users text-green-500 text-2xl mr-4"),
                    html.Div([
                        html.H4("A.L.M.A. - Consultas", className="text-lg font-semibold text-gray-800"),
                        html.P("(011) 4788-4129", className="text-xl font-bold text-green-600"),
                        html.P("Asociación Lucha contra el Mal de Alzheimer - Lunes a Viernes 9-17hs", className="text-gray-600 text-sm")
                    ])
                ], className="flex items-center p-4 bg-green-50 rounded-lg border border-green-200 mb-3"),
                
                # INECO contacto
                html.Div([
                    html.I(className="fas fa-brain text-purple-500 text-2xl mr-4"),
                    html.Div([
                        html.H4("INECO - Consultas Neurológicas", className="text-lg font-semibold text-gray-800"),
                        html.P("(011) 4807-4700", className="text-xl font-bold text-purple-600"),
                        html.P("Instituto de Neurología Cognitiva - Turnos y consultas especializadas", className="text-gray-600 text-sm")
                    ])
                ], className="flex items-center p-4 bg-purple-50 rounded-lg border border-purple-200")
            ])
        ], className="bg-white rounded-lg shadow-md p-6"),
        
        # Información adicional sobre recursos argentinos
        html.Div([
            html.H3("📋 Información Importante para Pacientes en Argentina", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("🏥 Sistema de Salud", className="text-lg font-semibold text-blue-600 mb-2"),
                    html.P("En Argentina, la evaluación y tratamiento del Alzheimer está cubierto por el sistema público de salud, obras sociales y medicina prepaga según la Ley 27.306 de Abordaje Integral e Interdisciplinario de las Demencias.", className="text-gray-600 text-sm")
                ], className="bg-blue-50 border border-blue-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("💊 Medicamentos", className="text-lg font-semibold text-green-600 mb-2"),
                    html.P("Los medicamentos específicos para Alzheimer (donepezilo, rivastigmina, memantina) están incluidos en el Programa Médico Obligatorio (PMO) y disponibles en farmacias públicas y del PAMI.", className="text-gray-600 text-sm")
                ], className="bg-green-50 border border-green-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("📄 Certificado de Discapacidad", className="text-lg font-semibold text-yellow-600 mb-2"),
                    html.P("Las personas con Alzheimer pueden obtener el Certificado Único de Discapacidad (CUD) que otorga beneficios como cobertura del 100% de tratamientos, medicamentos y asistencia domiciliaria.", className="text-gray-600 text-sm")
                ], className="bg-yellow-50 border border-yellow-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("👨‍⚕️ Especialistas", className="text-lg font-semibold text-purple-600 mb-2"),
                    html.P("Los neurólogos especializados en demencia y neuropsicólogos están disponibles en hospitales públicos como el Hospital de Clínicas, Hospital Italiano, FLENI, y centros especializados como INECO.", className="text-gray-600 text-sm")
                ], className="bg-purple-50 border border-purple-200 rounded-lg p-4")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
        ], className="bg-white rounded-lg shadow-md p-6 mt-6")
    ])

# ==========================================
# CSS PERSONALIZADO
# ==========================================
def get_updated_index_string():
    return f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            :root {{
                --color-primary: {config.colors['primary']};
                --color-secondary: {config.colors['secondary']};
                --color-text: {config.colors['text']};
                --color-low-risk: {config.colors['low_risk']};
                --color-moderate-risk: {config.colors['moderate_risk']};
                --color-high-risk: {config.colors['high_risk']};
                --color-neuro: {config.colors['neuro']};
                --color-background: {config.colors['background']};
                --color-card: {config.colors['card']};
                --color-data-bg: {config.colors['data_bg']};
            }}
            
            .tab-style {{
                border: none !important;
                border-radius: 8px 8px 0 0 !important;
                background-color: #e2e8f0 !important;
                color: #64748b !important;
                font-weight: 600 !important;
                padding: 12px 24px !important;
                margin-right: 4px !important;
                transition: all 0.2s ease !important;
            }}
            
            .tab-style:hover {{
                background-color: #cbd5e1 !important;
                color: var(--color-text) !important;
            }}
            
            .tab-selected {{
                background-color: var(--color-primary) !important;
                color: white !important;
                border-bottom: 3px solid #1d4ed8 !important;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                background-color: var(--color-background) !important;
                color: var(--color-text) !important;
            }}
        </style>
        
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''


app.index_string = get_updated_index_string()

# Header con nueva paleta
def create_header():
    return html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-brain mr-3", style={'color': config.colors['neuro']}),
                "Dashboard de Monitorización de Alzheimer"
            ], className="text-3xl font-bold", style={'color': config.colors['text']}),
            html.P(
                "Sistema inteligente de evaluación y prevención - Accesible para todos los usuarios",
                className="mt-2", style={'color': config.colors['muted']}
            )
        ], className="flex-1"),
        html.Div([
            html.Div([
                html.I(className="fas fa-calendar-alt mr-2", style={'color': config.colors['primary']}),
                datetime.now().strftime("%d/%m/%Y")
            ], className="text-sm", style={'color': config.colors['muted']}),
            html.Div([
                html.I(className="fas fa-users mr-2", style={'color': config.colors['secondary']}),
                f"≈2000 pacientes monitoreados"
            ], className="text-sm mt-1", style={'color': config.colors['muted']})
        ])
    ], className="shadow-lg rounded-lg p-6 mb-6 flex items-center justify-between",
       style={'backgroundColor': config.colors['card'], 'border': f"1px solid {config.colors['border']}"})


def create_risk_cards():
    """Crear cards de métricas con nueva paleta"""
    cards_data = [
        {
            'title': 'Total Pacientes',
            'value': f"{monitoring_data['total_patients'].iloc[-1]:,}",
            'icon': 'fas fa-users',
            'color': config.colors['primary'],
            'bg_color': config.alpha_colors['primary_10']
        },
        {
            'title': 'Precisión del Modelo',
            'value': f"{monitoring_data['model_accuracy'].iloc[-1]:.1%}",
            'icon': 'fas fa-bullseye',
            'color': config.colors['low_risk'],
            'bg_color': config.alpha_colors['low_risk_10']
        },
        {
            'title': 'Alto Riesgo',
            'value': f"{monitoring_data['high_risk'].iloc[-1]:,}",
            'icon': 'fas fa-exclamation-triangle',
            'color': config.colors['high_risk'],
            'bg_color': config.alpha_colors['high_risk_10']
        },
        {
            'title': 'Datos Neurológicos',
            'value': '1,247',
            'icon': 'fas fa-brain',
            'color': config.colors['neuro'],
            'bg_color': config.alpha_colors['neuro_10']
        }
    ]
    
    return html.Div([
        html.Div([
            html.Div([
                html.I(className=f"{card['icon']} text-3xl mb-2", style={'color': card['color']}),
                html.H3(card['value'], className="text-2xl font-bold", style={'color': config.colors['text']}),
                html.P(card['title'], style={'color': config.colors['muted']}),
            ], className="text-center")
        ], className="rounded-lg shadow-md p-6", 
           style={'backgroundColor': card['bg_color'], 'border': f"1px solid {card['color']}33"})
        for card in cards_data
    ], className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6")

def create_alert_components():
    """Crear componentes de alerta con nueva paleta"""
    alerts = [
        {
            'type': 'critical',
            'icon': 'fas fa-exclamation-circle',
            'title': 'Pacientes de Alto Riesgo Sin Seguimiento',
            'message': '8 pacientes clasificados como alto riesgo no tienen cita programada',
            'color': config.colors['high_risk'],
            'bg_color': config.alpha_colors['high_risk_10']
        },
        {
            'type': 'warning',
            'icon': 'fas fa-exclamation-triangle',
            'title': 'Biomarcadores Pendientes',
            'message': '15 pacientes tienen resultados de biomarcadores pendientes de análisis',
            'color': config.colors['moderate_risk'],
            'bg_color': config.alpha_colors['moderate_risk_10']
        },
        {
            'type': 'info',
            'icon': 'fas fa-info-circle',
            'title': 'Actualización de Modelo Disponible',
            'message': 'Nueva versión del modelo con mejoras en precisión disponible',
            'color': config.colors['primary'],
            'bg_color': config.alpha_colors['primary_10']
        }
    ]
    
    return html.Div([
        html.Div([
            html.Div([
                html.I(className=f"{alert['icon']} text-xl mr-3", style={'color': alert['color']}),
                html.Div([
                    html.H4(alert['title'], className="text-lg font-semibold", style={'color': alert['color']}),
                    html.P(alert['message'], className="mt-1", style={'color': config.colors['muted']})
                ], className="flex-1"),
                html.Button("Revisar", 
                          className="px-4 py-2 rounded-lg text-white font-medium",
                          style={'backgroundColor': alert['color']})
            ], className="flex items-center")
        ], className="border-l-4 p-4 rounded-r-lg mb-3",
           style={'backgroundColor': alert['bg_color'], 'borderLeftColor': alert['color']})
        for alert in alerts
    ])

def create_trend_charts():
    """Crear gráficos de tendencias con nueva paleta"""
    # Gráfico de evolución de pacientes
    evolution_fig = go.Figure([
        go.Scatter(x=monitoring_data['date'], y=monitoring_data['low_risk'], 
                 mode='lines+markers', name='Riesgo Bajo', 
                 line=dict(color=config.colors['low_risk'], width=3),
                 marker=dict(size=6, color=config.colors['low_risk'])),
        go.Scatter(x=monitoring_data['date'], y=monitoring_data['moderate_risk'], 
                 mode='lines+markers', name='Riesgo Moderado',
                 line=dict(color=config.colors['moderate_risk'], width=3),
                 marker=dict(size=6, color=config.colors['moderate_risk'])),
        go.Scatter(x=monitoring_data['date'], y=monitoring_data['high_risk'], 
                 mode='lines+markers', name='Riesgo Alto',
                 line=dict(color=config.colors['high_risk'], width=3),
                 marker=dict(size=6, color=config.colors['high_risk']))
    ]).update_layout(
        title="Tendencia de Casos por Nivel de Riesgo",
        xaxis_title="Fecha",
        yaxis_title="Número de Pacientes",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor=config.colors['card'],
        plot_bgcolor=config.colors['data_bg'],
        font={'color': config.colors['text']},
        title_font={'color': config.colors['text'], 'size': 18}
    )
    
    # Gráfico de rendimiento del modelo
    performance_fig = go.Figure([
        go.Scatter(x=monitoring_data['date'], y=monitoring_data['model_accuracy'], 
                 mode='lines+markers', name='Precisión',
                 line=dict(color=config.colors['primary'], width=3),
                 marker=dict(size=6, color=config.colors['primary']),
                 fill='tonexty',
                 fillcolor=config.alpha_colors['primary_10']),
        go.Scatter(x=monitoring_data['date'], y=[0.85]*len(monitoring_data), 
                 mode='lines', name='Umbral Mínimo',
                 line=dict(color=config.colors['high_risk'], width=2, dash='dash'))
    ]).update_layout(
        title="Evolución de la Precisión del Modelo",
        xaxis_title="Fecha",
        yaxis_title="Precisión (%)",
        yaxis=dict(tickformat='.1%', range=[0.8, 0.95]),
        height=400,
        hovermode='x unified',
        paper_bgcolor=config.colors['card'],
        plot_bgcolor=config.colors['data_bg'],
        font={'color': config.colors['text']},
        title_font={'color': config.colors['text'], 'size': 18}
    )
    
    return evolution_fig, performance_fig


# ==========================================
# CONFIGURACIÓN ESPECÍFICA PARA DASH TABLES
# ==========================================

def get_table_style_config():
    """Configuración de estilos para tablas Dash"""
    return {
        'style_cell': {
            'textAlign': 'center',
            'fontSize': '14px',
            'fontFamily': 'Inter, sans-serif',
            'color': config.colors['text'],
            'backgroundColor': config.colors['card'],
            'border': f'1px solid {config.colors["border"]}'
        },
        'style_header': {
            'backgroundColor': config.colors['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'border': f'1px solid {config.colors["primary"]}'
        },
        'style_data_conditional': [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': config.colors['data_bg']
            }
        ]
    }

# ==========================================
# CSS INLINE PARA COMPONENTES ESPECÍFICOS
# ==========================================

def get_component_styles():
    """Estilos inline específicos para componentes"""
    return {
        'main_container': {
            'backgroundColor': config.colors['background'],
            'minHeight': '100vh',
            'fontFamily': 'Inter, sans-serif'
        },
        'card_style': {
            'backgroundColor': config.colors['card'],
            'border': f'1px solid {config.colors["border"]}',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px -1px rgba(37, 99, 235, 0.1)',
            'padding': '1.5rem'
        },
        'primary_button': {
            'background': f'linear-gradient(135deg, {config.colors["primary"]}, #1d4ed8)',
            'color': 'white',
            'border': 'none',
            'borderRadius': '8px',
            'padding': '0.75rem 1.5rem',
            'fontWeight': '600',
            'cursor': 'pointer',
            'transition': 'all 0.2s ease'
        },
        'secondary_button': {
            'background': f'linear-gradient(135deg, {config.colors["secondary"]}, #0d9488)',
            'color': 'white',
            'border': 'none',
            'borderRadius': '8px',
            'padding': '0.75rem 1.5rem',
            'fontWeight': '600',
            'cursor': 'pointer',
            'transition': 'all 0.2s ease'
        },
        'neuro_highlight': {
            'background': f'linear-gradient(135deg, {config.alpha_colors["neuro_10"]}, rgba(124, 58, 237, 0.02))',
            'borderLeft': f'4px solid {config.colors["neuro"]}',
            'padding': '1rem',
            'borderRadius': '0 8px 8px 0'
        }
    }


def create_risk_gauge(risk_probability):
    """Crear gauge de riesgo con nueva paleta"""
    risk_color = get_risk_color(risk_probability)
    
    return go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Riesgo de Alzheimer (%)", 'font': {'color': config.colors['text']}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': config.colors['text']},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': hex_to_rgba(config.colors['low_risk'], 0.3)},
                {'range': [30, 70], 'color': hex_to_rgba(config.colors['moderate_risk'], 0.3)},
                {'range': [70, 100], 'color': hex_to_rgba(config.colors['high_risk'], 0.3)}
            ],
            'threshold': {
                'line': {'color': config.colors['high_risk'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    )).update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=config.colors['card'],
        font={'color': config.colors['text']}
    )

def create_feature_importance_chart():
    """Crear gráfico de importancia con nueva paleta"""
    return px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        color='group',
        orientation='h',
        title="Top 10 Factores Más Importantes para la Predicción",
        labels={'importance': 'Importancia (%)', 'feature': 'Factor'},
        color_discrete_map=config.color_maps['feature_groups']
    ).update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True,
        paper_bgcolor=config.colors['card'],
        plot_bgcolor=config.colors['card'],
        font={'color': config.colors['text']},
        title_font={'color': config.colors['text'], 'size': 18}
    )



def load_clinical_metrics():
    """Carga métricas de impacto clínico reales"""
    try:
        if config.clinical_metrics_file.exists():
            with open(config.clinical_metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                return metrics
        else:
            logger.warning("Métricas clínicas no encontradas")
            return {}
    except Exception as e:
        logger.error(f"Error cargando métricas clínicas: {e}")
        return {}

# Cargar métricas clínicas
clinical_metrics = load_clinical_metrics()

# ==========================================
# EJECUTAR LA APLICACIÓN
# ==========================================

if __name__ == '__main__':
    # Configuración para desarrollo local
    try:
        app.run_server(debug=True, host='127.0.0.1', port=8050, 
                      threaded=True, use_reloader=False, dev_tools_hot_reload=False)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
        app.run_server(debug=False, host='127.0.0.1', port=8050)
else:
    # Configuración para producción en Render
    port = int(os.environ.get('PORT', 10000))
    app.run_server(debug=False, host='0.0.0.0', port=port)