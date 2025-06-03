"""
Feature Engineering para datos sintéticos de actividad y sueño
Patrones conductuales relevantes para Alzheimer
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ActivitySleepFeatureEngineer:
    def __init__(self):
        self.activity_features = []
        self.sleep_features = []
        
    def transform(self, df):
        """
        Aplicar transformaciones de feature engineering para actividad y sueño
        """
        print("🏃 Procesando features de actividad y sueño...")
        df_transformed = df.copy()
        
        # 1. Features de actividad física
        df_transformed = self._process_activity_features(df_transformed)
        
        # 2. Features de patrones de sueño
        df_transformed = self._process_sleep_features(df_transformed)
        
        # 3. Features combinadas actividad-sueño
        df_transformed = self._create_combined_features(df_transformed)
        
        # 4. Indicadores de riesgo conductual
        df_transformed = self._create_behavioral_risk_indicators(df_transformed)
        
        total_features = len(self.activity_features) + len(self.sleep_features)
        print(f"   ✅ {total_features} features de actividad/sueño creadas")
        return df_transformed
    
    def _process_activity_features(self, df):
        """Procesar y crear features de actividad física"""
        
        # Identificar columnas de actividad
        activity_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['step', 'walk', 'exercise', 'activity', 'movement', 'pace'])]
        
        if activity_cols:
            # Nivel promedio de actividad
            numeric_activity = df[activity_cols].select_dtypes(include=[np.number])
            if not numeric_activity.empty:
                df['activity_mean_level'] = numeric_activity.mean(axis=1)
                df['activity_variability'] = numeric_activity.std(axis=1).fillna(0)
                self.activity_features.extend(['activity_mean_level', 'activity_variability'])
                
                # Indicador de baja actividad (factor de riesgo)
                activity_threshold = df['activity_mean_level'].quantile(0.25)
                df['low_activity_risk'] = (df['activity_mean_level'] <= activity_threshold).astype(int)
                self.activity_features.append('low_activity_risk')
        
        # Patrones de actividad específicos
        step_cols = [col for col in df.columns if 'step' in col.lower()]
        if step_cols:
            for col in step_cols[:3]:  # Limitar a primeras 3 columnas
                if df[col].dtype in ['int64', 'float64']:
                    # Categorizar nivel de pasos
                    df[f'{col}_category'] = pd.cut(df[col], 
                                                 bins=[0, 2000, 5000, 10000, float('inf')], 
                                                 labels=['sedentary', 'low', 'moderate', 'active'],
                                                 include_lowest=True)
                    df[f'{col}_category'] = df[f'{col}_category'].cat.codes
                    self.activity_features.append(f'{col}_category')
        
        return df
    
    def _process_sleep_features(self, df):
        """Procesar y crear features de sueño"""
        
        # Identificar columnas de sueño
        sleep_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['sleep', 'rest', 'bed', 'wake', 'rem', 'deep'])]
        
        if sleep_cols:
            numeric_sleep = df[sleep_cols].select_dtypes(include=[np.number])
            if not numeric_sleep.empty:
                # Calidad general del sueño
                df['sleep_quality_score'] = numeric_sleep.mean(axis=1)
                df['sleep_pattern_variability'] = numeric_sleep.std(axis=1).fillna(0)
                self.sleep_features.extend(['sleep_quality_score', 'sleep_pattern_variability'])
        
        # Procesar duración de sueño específica
        duration_cols = [col for col in df.columns if 'duration' in col.lower() and 'sleep' in col.lower()]
        if duration_cols:
            for col in duration_cols[:2]:  # Primeras 2 columnas
                if df[col].dtype in ['int64', 'float64']:
                    # Categorizar duración de sueño (clínicamente relevante)
                    df[f'{col}_adequate'] = ((df[col] >= 6) & (df[col] <= 9)).astype(int)
                    df[f'{col}_insufficient'] = (df[col] < 6).astype(int)
                    df[f'{col}_excessive'] = (df[col] > 9).astype(int)
                    self.sleep_features.extend([f'{col}_adequate', f'{col}_insufficient', f'{col}_excessive'])
        
        # Patrones de sueño fragmentado
        fragmentation_cols = [col for col in df.columns if any(keyword in col.lower() 
                             for keyword in ['awake', 'interrupt', 'fragment'])]
        if fragmentation_cols:
            numeric_frag = df[fragmentation_cols].select_dtypes(include=[np.number])
            if not numeric_frag.empty:
                df['sleep_fragmentation_index'] = numeric_frag.mean(axis=1)
                self.sleep_features.append('sleep_fragmentation_index')
        
        return df
    
    def _create_combined_features(self, df):
        """Crear features combinadas de actividad y sueño"""
        
        # Ratio actividad/calidad de sueño
        if 'activity_mean_level' in df.columns and 'sleep_quality_score' in df.columns:
            df['activity_sleep_ratio'] = df['activity_mean_level'] / (df['sleep_quality_score'] + 1e-6)
            df['activity_sleep_balance'] = np.where(
                (df['activity_mean_level'] > df['activity_mean_level'].median()) & 
                (df['sleep_quality_score'] > df['sleep_quality_score'].median()), 1, 0)
            self.activity_features.extend(['activity_sleep_ratio', 'activity_sleep_balance'])
        
        # Indicador de estilo de vida saludable
        healthy_indicators = []
        if 'low_activity_risk' in df.columns:
            healthy_indicators.append(1 - df['low_activity_risk'])
        
        sleep_adequate_cols = [col for col in df.columns if col.endswith('_adequate')]
        if sleep_adequate_cols:
            healthy_indicators.append(df[sleep_adequate_cols[0]])
        
        if healthy_indicators:
            df['healthy_lifestyle_score'] = np.mean(healthy_indicators, axis=0)
            self.activity_features.append('healthy_lifestyle_score')
        
        return df
    
    def _create_behavioral_risk_indicators(self, df):
        """Crear indicadores de riesgo conductual para Alzheimer"""
        
        risk_factors = []
        
        # Factor 1: Actividad física insuficiente
        if 'low_activity_risk' in df.columns:
            risk_factors.append(df['low_activity_risk'])
        
        # Factor 2: Sueño inadecuado
        insufficient_sleep_cols = [col for col in df.columns if col.endswith('_insufficient')]
        if insufficient_sleep_cols:
            risk_factors.append(df[insufficient_sleep_cols[0]])
        
        # Factor 3: Alta variabilidad en patrones
        if 'sleep_pattern_variability' in df.columns:
            high_variability = (df['sleep_pattern_variability'] > df['sleep_pattern_variability'].quantile(0.8)).astype(int)
            risk_factors.append(high_variability)
        
        # Score compuesto de riesgo conductual
        if risk_factors:
            df['behavioral_risk_score'] = np.mean(risk_factors, axis=0)
            df['high_behavioral_risk'] = (df['behavioral_risk_score'] >= 0.6).astype(int)
            self.activity_features.extend(['behavioral_risk_score', 'high_behavioral_risk'])
        
        return df
    
    def get_feature_names(self):
        """Retornar nombres de features creadas"""
        return self.activity_features + self.sleep_features

def engineer_activity_sleep_features(df):
    """
    Función principal para feature engineering de actividad y sueño
    """
    engineer = ActivitySleepFeatureEngineer()
    return engineer.transform(df)