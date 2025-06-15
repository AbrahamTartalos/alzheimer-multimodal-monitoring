"""
Temporal Modeling for Alzheimer Progression Analysis
====================================================

Modelos temporales para análisis de progresión y evolución del riesgo de Alzheimer.
Incluye análisis de series temporales y modelos de supervivencia.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AlzheimerTemporalModeling:
    """Clase para modelado temporal de progresión de Alzheimer"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_temporal_data(self, df: pd.DataFrame, 
                            subject_col: str = 'PTID',
                            time_col: str = 'VISCODE',
                            target_col: str = 'composite_risk_score') -> pd.DataFrame:
        """
        Prepara datos para análisis temporal
        
        Args:
            df: DataFrame con datos longitudinales
            subject_col: Columna de identificación de sujeto
            time_col: Columna de tiempo/visita
            target_col: Variable objetivo temporal
            
        Returns:
            DataFrame preparado para análisis temporal
        """
        # Filtrar datos con información temporal válida
        temporal_data = df.dropna(subset=[subject_col, time_col, target_col]).copy()
        
        # Convertir códigos de visita a valores numéricos
        if temporal_data[time_col].dtype == 'object':
            # Mapear códigos comunes de ADNI
            visit_mapping = {
                'bl': 0, 'sc': 0, 'scmri': 0,
                'm03': 3, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24,
                'm30': 30, 'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72
            }
            
            temporal_data[time_col] = temporal_data[time_col].str.lower()
            temporal_data['time_numeric'] = temporal_data[time_col].map(visit_mapping)
            temporal_data = temporal_data.dropna(subset=['time_numeric'])
        else:
            temporal_data['time_numeric'] = temporal_data[time_col]
        
        # Ordenar por sujeto y tiempo
        temporal_data = temporal_data.sort_values([subject_col, 'time_numeric'])
        
        return temporal_data
    
    def create_progression_features(self, df: pd.DataFrame,
                                  subject_col: str = 'PTID',
                                  time_col: str = 'time_numeric',
                                  target_col: str = 'composite_risk_score') -> pd.DataFrame:
        """
        Crea features de progresión temporal
        
        Args:
            df: DataFrame temporal
            subject_col: Columna de sujeto
            time_col: Columna de tiempo
            target_col: Variable objetivo
            
        Returns:
            DataFrame con features de progresión
        """
        progression_data = []
        
        for subject in df[subject_col].unique():
            subject_data = df[df[subject_col] == subject].copy()
            
            if len(subject_data) < 2:
                continue
                
            # Calcular features de progresión
            subject_data = subject_data.sort_values(time_col)
            
            # Valor baseline
            baseline_value = subject_data[target_col].iloc[0]
            
            # Cambio desde baseline
            subject_data['change_from_baseline'] = subject_data[target_col] - baseline_value
            
            # Tasa de cambio
            time_diff = subject_data[time_col].diff()
            value_diff = subject_data[target_col].diff()
            subject_data['rate_of_change'] = value_diff / time_diff.replace(0, np.nan)
            
            # Tendencia general (slope)
            if len(subject_data) >= 3:
                time_values = subject_data[time_col].values
                target_values = subject_data[target_col].values
                slope = np.polyfit(time_values, target_values, 1)[0]
                subject_data['overall_slope'] = slope
            
            # Valor máximo y mínimo
            subject_data['max_value'] = subject_data[target_col].max()
            subject_data['min_value'] = subject_data[target_col].min()
            
            # Variabilidad
            subject_data['variability'] = subject_data[target_col].std()
            
            progression_data.append(subject_data)
        
        return pd.concat(progression_data, ignore_index=True)
    
    def fit_progression_models(self, df: pd.DataFrame,
                             target_col: str = 'composite_risk_score',
                             time_col: str = 'time_numeric') -> Dict:
        """
        Entrena modelos de progresión temporal
        
        Args:
            df: DataFrame con datos temporales
            target_col: Variable objetivo
            time_col: Columna de tiempo
            
        Returns:
            Diccionario con modelos entrenados
        """
        models = {}
        
        # Preparar features
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'PTID', 'VISCODE', time_col]]
        
        # Filtrar features numéricas
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("⚠️  No se encontraron features numéricas para modelado temporal")
            return models
        
        X = df[numeric_cols + [time_col]].fillna(df[numeric_cols + [time_col]].median())
        y = df[target_col].fillna(df[target_col].median())
        
        # Modelo lineal simple
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        models['linear_progression'] = linear_model
        
        # Modelo Random Forest para capturar no-linealidades
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            max_depth=10
        )
        rf_model.fit(X, y)
        models['rf_progression'] = rf_model
        
        return models
    
    def predict_future_risk(self, df: pd.DataFrame, 
                          model_name: str = 'rf_progression',
                          future_time_points: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """
        Predice riesgo futuro para diferentes puntos temporales
        
        Args:
            df: DataFrame con datos actuales
            model_name: Nombre del modelo a usar
            future_time_points: Puntos temporales futuros (en meses)
            
        Returns:
            DataFrame con predicciones futuras
        """
        if model_name not in self.models:
            print(f"❌ Modelo {model_name} no encontrado")
            return pd.DataFrame()
        
        model = self.models[model_name]
        predictions = []
        
        for time_point in future_time_points:
            # Crear features para predicción futura
            future_df = df.copy()
            future_df['time_numeric'] = time_point
            
            # Seleccionar features apropiadas
            feature_cols = [col for col in future_df.columns 
                           if col not in ['composite_risk_score', 'PTID', 'VISCODE']]
            numeric_cols = future_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            if 'time_numeric' not in numeric_cols:
                numeric_cols.append('time_numeric')
            
            X_future = future_df[numeric_cols].fillna(future_df[numeric_cols].median())
            
            # Predicción
            y_pred = model.predict(X_future)
            
            prediction_df = pd.DataFrame({
                'future_months': time_point,
                'predicted_risk': y_pred,
                'subject_id': range(len(y_pred))
            })
            
            predictions.append(prediction_df)
        
        return pd.concat(predictions, ignore_index=True)
    
    def analyze_progression_patterns(self, df: pd.DataFrame,
                                   subject_col: str = 'PTID',
                                   target_col: str = 'composite_risk_score') -> Dict:
        """
        Analiza patrones de progresión en la población
        
        Args:
            df: DataFrame con datos temporales
            subject_col: Columna de sujeto
            target_col: Variable objetivo
            
        Returns:
            Diccionario con análisis de patrones
        """
        patterns = {}
        
        # Análisis por sujeto
        subject_stats = []
        
        for subject in df[subject_col].unique():
            subject_data = df[df[subject_col] == subject]
            
            if len(subject_data) < 2:
                continue
            
            # Calcular estadísticas de progresión
            initial_risk = subject_data[target_col].iloc[0]
            final_risk = subject_data[target_col].iloc[-1]
            total_change = final_risk - initial_risk
            
            # Clasificar patrón de progresión
            if total_change > 0.1:
                pattern = 'progressive'
            elif total_change < -0.1:
                pattern = 'improving'
            else:
                pattern = 'stable'
            
            subject_stats.append({
                'subject': subject,
                'initial_risk': initial_risk,
                'final_risk': final_risk,
                'total_change': total_change,
                'pattern': pattern,
                'n_visits': len(subject_data)
            })
        
        subject_stats_df = pd.DataFrame(subject_stats)
        
        # Resumen de patrones
        patterns['pattern_distribution'] = subject_stats_df['pattern'].value_counts()
        patterns['average_change_by_pattern'] = subject_stats_df.groupby('pattern')['total_change'].mean()
        patterns['subject_statistics'] = subject_stats_df
        
        return patterns
    
    def evaluate_temporal_models(self, df: pd.DataFrame, target_col: str = 'composite_risk_score') -> Dict:
        """
        Evalúa modelos temporales
        
        Args:
            df: DataFrame con datos
            target_col: Variable objetivo
            
        Returns:
            Métricas de evaluación
        """
        results = {}
        
        # Preparar datos para evaluación
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'PTID', 'VISCODE']]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return results
        
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col].fillna(df[target_col].median())
        
        # Evaluar cada modelo
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X)
                
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                }
            except Exception as e:
                print(f"⚠️  Error evaluando {name}: {e}")
                results[name] = {'mse': np.inf, 'rmse': np.inf, 'r2': -np.inf}
        
        return results
    
    def run_temporal_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Ejecuta análisis temporal completo
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Resultados del análisis temporal
        """
        print("🚀 Iniciando análisis temporal...")
        
        # Preparar datos temporales
        temporal_data = self.prepare_temporal_data(df)
        print(f"📊 Datos temporales preparados: {len(temporal_data)} registros")
        
        # Crear features de progresión
        progression_data = self.create_progression_features(temporal_data)
        print(f"📈 Features de progresión creadas: {len(progression_data)} registros")
        
        # Entrenar modelos
        self.models = self.fit_progression_models(progression_data)
        print(f"✅ {len(self.models)} modelos temporales entrenados")
        
        # Evaluar modelos
        evaluation_results = self.evaluate_temporal_models(progression_data)
        print("📊 Evaluación de modelos completada")
        
        # Analizar patrones de progresión
        progression_patterns = self.analyze_progression_patterns(progression_data)
        print("🔍 Análisis de patrones completado")
        
        # Registrar en MLflow
        self.log_temporal_results(evaluation_results, progression_patterns)
        
        return {
            'temporal_data': temporal_data,
            'progression_data': progression_data,
            'models': self.models,
            'evaluation': evaluation_results,
            'patterns': progression_patterns
        }
    
    def log_temporal_results(self, evaluation_results: Dict, progression_patterns: Dict):
        """
        Registra resultados temporales en MLflow
        
        Args:
            evaluation_results: Métricas de evaluación
            progression_patterns: Patrones de progresión
        """
        with mlflow.start_run():
            mlflow.set_tag("analysis_type", "temporal_modeling")
            
            # Registrar métricas de modelos
            for model_name, metrics in evaluation_results.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Registrar estadísticas de patrones
            if 'pattern_distribution' in progression_patterns:
                for pattern, count in progression_patterns['pattern_distribution'].items():
                    mlflow.log_metric(f"pattern_{pattern}_count", count)
            
            # Registrar mejor modelo
            if evaluation_results:
                best_model = min(evaluation_results.keys(), 
                               key=lambda x: evaluation_results[x]['mse'])
                mlflow.set_tag("best_temporal_model", best_model)
                
                if best_model in self.models:
                    mlflow.sklearn.log_model(self.models[best_model], "best_temporal_model")

def run_temporal_modeling(df: pd.DataFrame) -> Dict:
    """
    Función principal para modelado temporal
    
    Args:
        df: DataFrame con datos
        
    Returns:
        Resultados del análisis temporal
    """
    temporal_analyzer = AlzheimerTemporalModeling()
    return temporal_analyzer.run_temporal_analysis(df)

# Configuraciones para diferentes tipos de análisis temporal
TEMPORAL_CONFIGS = {
    'risk_progression': {
        'target': 'composite_risk_score',
        'time_windows': [6, 12, 18, 24, 36],
        'models': ['linear_progression', 'rf_progression']
    },
    'cognitive_decline': {
        'target': 'CDRSB_LOG',
        'time_windows': [12, 24, 36, 48],
        'models': ['linear_progression', 'rf_progression']
    }
}

