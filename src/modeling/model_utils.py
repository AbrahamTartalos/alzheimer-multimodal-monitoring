# Copyright 2025 Abraham Tartalos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilidades comunes para modelado de Alzheimer
============================================
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(file_path='../data/processed/features/alzheimer_features_selected_20250621.csv'):
    """
    Carga datos procesados para modelado
    
    Returns:
        pd.DataFrame: Dataset con features procesadas
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Datos cargados: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {file_path}")
        return None

def prepare_features(df, target_col, test_size=0.2, random_state=42):
    """
    Prepara features para entrenamiento
    
    Args:
        df: DataFrame con datos
        target_col: Columna objetivo
        test_size: Proporción de datos para test
        random_state: Semilla aleatoria
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    # Remover filas con target faltante
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Separar features y target
    feature_cols = [col for col in df_clean.columns if col != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Manejar valores faltantes en features
    X = X.fillna(X.median())
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if y.dtype == 'object' else None
    )
    
    print(f"📊 Train set: {X_train.shape[0]} muestras")
    print(f"📊 Test set: {X_test.shape[0]} muestras")
    print(f"🎯 Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def evaluate_model(model, X_test, y_test, model_type='regression'):
    """
    Evalúa modelo y retorna métricas como floats
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        model_type: 'regression' o 'classification'
    
    Returns:
        dict: Diccionario con métricas
    """
    y_pred = model.predict(X_test)
    metrics = {}
    
    if model_type == 'regression':
        # Calcular métricas clave como floats
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
        metrics['r2_score'] = float(r2_score(y_test, y_pred))
        
        print(f"📊 RMSE: {metrics['rmse']:.4f}")
        print(f"📊 MAE: {metrics['mae']:.4f}")
        print(f"📊 R²: {metrics['r2_score']:.4f}")
        
    elif model_type == 'classification':
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
        
        # AUC para clasificación binaria/multiclase
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    metrics['auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    metrics['auc'] = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        except:
            pass
        
        print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📊 F1-Score: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"📊 AUC: {metrics['auc']:.4f}")
    
    # Conversión adicional para garantizar que todo sea float
    for metric in list(metrics.keys()):
        value = metrics[metric]
        
        # Convertir arrays de tamaño 1 a float
        if isinstance(value, np.ndarray) and value.size == 1:
            metrics[metric] = float(value.item())
        
        # Convertir tipos numéricos de numpy a float de Python
        elif isinstance(value, np.generic):
            metrics[metric] = float(value)
        
        # Manejar casos inesperados
        elif not isinstance(value, (int, float)):
            print(f"⚠️ Advertencia: La métrica '{metric}' tiene tipo no numérico: {type(value)}")
            # Intentar conversión forzada si es posible
            try:
                metrics[metric] = float(value)
            except:
                print(f"❌ No se pudo convertir '{metric}', eliminando del reporte")
                del metrics[metric]

    return metrics

def log_model_metrics(metrics, model_name, model_type):
    """
    Registrar métricas en MLflow asegurando que sean escalares
    
    Args:
        metrics: Diccionario con métricas
        model_name: Nombre del modelo
        model_type: Tipo de modelo
    """
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        for metric_name, value in metrics.items():
            try:
                # Convertir cualquier tipo NumPy a float
                if isinstance(value, (np.ndarray, np.generic)):
                    if value.size == 1:
                        value = float(value.item())
                    else:
                        # Si es un array, registrar solo el primer valor
                        print(f"⚠️ Array detectado en {metric_name} - usando primer valor")
                        value = float(value[0])
            
                # Registrar solo si es un número
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)
                    print(f"   ✓ {metric_name}: {value}")
                else:
                    print(f"⚠️ Omitiendo {metric_name} - tipo no soportado: {type(value)}")
                    
            except Exception as e:
                print(f"❌ Error registrando {metric_name}={value}: {str(e)}")
        
        print(f"✅ Métricas registradas en MLflow para {model_name}")

def create_model_comparison(results_dict):
    """
    Crea comparación de modelos
    
    Args:
        results_dict: Diccionario con resultados de modelos
    
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 COMPARACIÓN DE MODELOS")
    print("=" * 50)
    print(comparison_df.round(4))
    
    return comparison_df

def save_model_artifacts(model, model_name, scaler=None, feature_names=None):
    """
    Guarda artefactos del modelo
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        scaler: Scaler usado (opcional)
        feature_names: Nombres de features (opcional)
    """
    # Crear directorio si no existe
    os.makedirs('../models', exist_ok=True)
    
    # Guardar modelo
    model_path = f'../models/{model_name}_model.joblib'
    joblib.dump(model, model_path)
    
    # Guardar scaler si existe
    if scaler is not None:
        scaler_path = f'../models/{model_name}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
    
    # Guardar nombres de features si existen
    if feature_names is not None:
        features_path = f'../models/{model_name}_features.joblib'
        joblib.dump(feature_names, features_path)
    
    print(f"💾 Modelo guardado: {model_path}")

def get_feature_importance(model, feature_names, top_n=20):
    """
    Obtiene importancia de features
    
    Args:
        model: Modelo entrenado
        feature_names: Nombres de features
        top_n: Número de features principales
    
    Returns:
        pd.DataFrame: Features ordenadas por importancia
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            print("⚠️ Modelo no soporta importancia de features")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
        
    except Exception as e:
        print(f"❌ Error calculando importancia: {e}")
        return None

def validate_data_quality(df, target_col):
    """
    Valida calidad de datos para modelado
    
    Args:
        df: DataFrame a validar
        target_col: Columna objetivo
    
    Returns:
        dict: Reporte de calidad
    """
    report = {
        'total_rows': len(df),
        'total_features': len(df.columns) - 1,
        'target_missing': df[target_col].isna().sum(),
        'features_missing_pct': (df.drop(target_col, axis=1).isna().sum() / len(df) * 100).mean(),
        'duplicate_rows': df.duplicated().sum(),
        'constant_features': (df.drop(target_col, axis=1).nunique() == 1).sum()
    }
    
    print("\n🔍 REPORTE DE CALIDAD DE DATOS")
    print("=" * 40)
    for key, value in report.items():
        print(f"📊 {key}: {value}")
    
    return report