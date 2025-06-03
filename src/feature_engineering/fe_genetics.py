"""
Feature Engineering para datos genéticos
Enfoque en APOE y marcadores de riesgo genético
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GeneticsFeatureEngineer:
    def __init__(self):
        self.genetic_features = []
        
    def transform(self, df):
        """
        Aplicar transformaciones de feature engineering para datos genéticos
        """
        print("🧬 Procesando features genéticas...")
        df_transformed = df.copy()
        
        # 1. Procesar APOE (principal factor de riesgo genético)
        df_transformed = self._process_apoe_variants(df_transformed)
        
        # 2. Features temporales genéticas
        df_transformed = self._create_genetic_temporal_features(df_transformed)
        
        # 3. Indicadores de completitud genética
        df_transformed = self._create_genetic_completeness_indicators(df_transformed)
        
        print(f"   ✅ {len(self.genetic_features)} features genéticas creadas")
        return df_transformed
    
    def _process_apoe_variants(self, df):
        """Procesar variantes APOE y crear features de riesgo"""
        
        # Identificar columnas APOE disponibles
        apoe_cols = [col for col in df.columns if 'apoe' in col.lower() or 'APOE' in col]
        
        if apoe_cols:
            # APOE ε4 carrier status (principal factor de riesgo)
            for col in apoe_cols:
                if df[col].dtype == 'object' or 'allele' in col.lower():
                    # Crear indicador de portador APOE ε4
                    df[f'APOE_e4_carrier'] = df[col].astype(str).str.contains('4', na=False).astype(int)
                    self.genetic_features.append('APOE_e4_carrier')
                    
                    # Crear indicador de homocigoto ε4/ε4 (mayor riesgo)
                    df[f'APOE_e4_homozygous'] = df[col].astype(str).str.count('4') >= 2
                    df[f'APOE_e4_homozygous'] = df[f'APOE_e4_homozygous'].astype(int)
                    self.genetic_features.append('APOE_e4_homozygous')
                    
                    # Crear score de riesgo APOE (0: no carrier, 1: heterozygous, 2: homozygous)
                    df[f'APOE_risk_score'] = df[col].astype(str).str.count('4')
                    df[f'APOE_risk_score'] = df[f'APOE_risk_score'].clip(0, 2)
                    self.genetic_features.append('APOE_risk_score')
                    break
        
        return df
    
    def _create_genetic_temporal_features(self, df):
        """Crear features temporales basadas en fechas genéticas"""
        
        # Procesar fechas de actualización APOE
        date_cols = ['update_stamp_apoe', 'USERDATE', 'APTESTDT']
        
        for date_col in date_cols:
            if date_col in df.columns:
                # Años desde la prueba genética
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    current_year = datetime.now().year
                    df[f'{date_col}_years_ago'] = current_year - df[date_col].dt.year
                    df[f'{date_col}_years_ago'] = df[f'{date_col}_years_ago'].clip(0, 50)  # Limit realistic range
                    self.genetic_features.append(f'{date_col}_years_ago')
                except:
                    pass
        
        # Indicador de datos genéticos recientes (últimos 5 años)
        recent_genetic_cols = [col for col in df.columns if col.endswith('_years_ago')]
        if recent_genetic_cols:
            df['genetic_data_recent'] = (df[recent_genetic_cols] <= 5).any(axis=1).astype(int)
            self.genetic_features.append('genetic_data_recent')
        
        return df
    
    def _create_genetic_completeness_indicators(self, df):
        """Indicadores de completitud de datos genéticos"""
        
        # Identificar todas las columnas genéticas
        genetic_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['apoe', 'genetic', 'allele', 'snp', 'gene'])]
        
        if genetic_cols:
            # Porcentaje de completitud genética
            df['genetic_completeness'] = df[genetic_cols].notna().mean(axis=1)
            self.genetic_features.append('genetic_completeness')
            
            # Indicador de perfil genético completo
            df['genetic_profile_complete'] = (df['genetic_completeness'] >= 0.8).astype(int)
            self.genetic_features.append('genetic_profile_complete')
        
        return df
    
    def get_feature_names(self):
        """Retornar nombres de features creadas"""
        return self.genetic_features

def engineer_genetic_features(df):
    """
    Función principal para feature engineering genético
    """
    engineer = GeneticsFeatureEngineer()
    return engineer.transform(df)