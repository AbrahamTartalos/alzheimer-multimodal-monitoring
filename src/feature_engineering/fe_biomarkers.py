"""
Feature Engineering - Biomarcadores
===================================
Procesamiento de biomarcadores en CSF, plasma y PET para Alzheimer.
Enfoque en ratios clínicos y umbrales diagnósticos establecidos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)

class BiomarkerFeatureEngineer:
    """
    Procesador de características de biomarcadores para Alzheimer.
    
    Incluye biomarcadores de amiloide, tau, neurodegeneración y ratios clínicos
    establecidos en las guías de diagnóstico de Alzheimer.
    """
    
    def __init__(self):
        self.feature_names = []
        self.clinical_thresholds = {
            'amyloid_beta_42': 192,  # pg/mL umbral patológico
            'tau_total': 240,        # pg/mL umbral elevado
            'ptau_181': 18,          # pg/mL umbral patológico
        }
        
    def transform(self, df):
        """
        Aplica feature engineering a biomarcadores.
        
        Args:
            df (pd.DataFrame): Dataset con columnas de biomarcadores
            
        Returns:
            pd.DataFrame: Dataset con características de biomarcadores procesadas
        """
        logger.info("🧪 Iniciando feature engineering de biomarcadores...")
        
        df_features = df.copy()
        
        # Identificar columnas de biomarcadores
        biomarker_cols = self._identify_biomarker_columns(df)
        
        if not biomarker_cols:
            logger.warning("No se encontraron columnas de biomarcadores")
            return df_features
            
        logger.info(f"📊 Procesando {len(biomarker_cols)} biomarcadores")
        
        # 1. Ratios clínicos establecidos
        self._create_clinical_ratios(df_features, biomarker_cols)
        
        # 2. Clasificación por umbrales clínicos
        self._apply_clinical_thresholds(df_features, biomarker_cols)
        
        # 3. Scores compuestos
        self._create_composite_scores(df_features, biomarker_cols)
        
        # 4. Transformaciones estadísticas
        self._apply_statistical_transforms(df_features, biomarker_cols)
        
        # 5. Interacciones entre biomarcadores
        self._create_biomarker_interactions(df_features, biomarker_cols)
        
        logger.info(f"✅ Feature engineering completado. Nuevas características: {len(self.feature_names)}")
        
        return df_features
    
    def _identify_biomarker_columns(self, df):
        """Identifica columnas de biomarcadores en el dataset."""
        
        biomarker_keywords = [
            'amyloid', 'abeta', 'ab42', 'ab40', 'tau', 'ptau', 'ttau',
            'biomarker', 'csf', 'plasma', 'pet', 'suvr', 'centiloid',
            'neurodegeneration', 'inflammation', 'apoe'
        ]
        
        biomarker_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in biomarker_keywords):
                if df[col].dtype in ['float64', 'int64']:  # Solo columnas numéricas
                    biomarker_cols.append(col)
        
        return biomarker_cols
    
    def _create_clinical_ratios(self, df, biomarker_cols):
        """Crea ratios clínicos establecidos para diagnóstico de Alzheimer."""
        
        # Ratio Aβ42/Aβ40 (más estable que Aβ42 solo)
        ab42_cols = [col for col in biomarker_cols if 'ab42' in col.lower() or 'amyloid_42' in col.lower()]
        ab40_cols = [col for col in biomarker_cols if 'ab40' in col.lower() or 'amyloid_40' in col.lower()]
        
        if ab42_cols and ab40_cols:
            ratio_col = 'ab42_ab40_ratio'
            df[ratio_col] = df[ab42_cols[0]] / (df[ab40_cols[0]] + 1e-8)
            self.feature_names.append(ratio_col)
        
        # Ratio Tau/Aβ42 (marcador de progresión)
        tau_cols = [col for col in biomarker_cols if 'tau' in col.lower() and 'p' not in col.lower()]
        
        if tau_cols and ab42_cols:
            ratio_col = 'tau_ab42_ratio'
            df[ratio_col] = df[tau_cols[0]] / (df[ab42_cols[0]] + 1e-8)
            self.feature_names.append(ratio_col)
        
        # Ratio pTau/Tau total (especificidad de Alzheimer)
        ptau_cols = [col for col in biomarker_cols if 'ptau' in col.lower() or 'p_tau' in col.lower()]
        
        if ptau_cols and tau_cols:
            ratio_col = 'ptau_tau_ratio'
            df[ratio_col] = df[ptau_cols[0]] / (df[tau_cols[0]] + 1e-8)
            self.feature_names.append(ratio_col)
    
    def _apply_clinical_thresholds(self, df, biomarker_cols):
        """Aplica umbrales clínicos establecidos para clasificación binaria."""
        
        threshold_mapping = {
            'amyloid': ('ab42', 192, 'low'),      # Aβ42 bajo = patológico
            'tau': ('tau', 240, 'high'),          # Tau alto = patológico  
            'ptau': ('ptau', 18, 'high')          # pTau alto = patológico
        }
        
        for biomarker_type, (keyword, threshold, direction) in threshold_mapping.items():
            matching_cols = [col for col in biomarker_cols if keyword in col.lower()]
            
            for col in matching_cols:
                binary_col = f"{col}_pathological"
                
                if direction == 'low':
                    df[binary_col] = (df[col] < threshold).astype(int)
                else:  # high
                    df[binary_col] = (df[col] > threshold).astype(int)
                
                self.feature_names.append(binary_col)
    
    def _create_composite_scores(self, df, biomarker_cols):
        """Crea scores compuestos de biomarcadores."""
        
        # Score AT(N) - Amyloid, Tau, Neurodegeneration
        amyloid_cols = [col for col in biomarker_cols if 'amyloid' in col.lower() or 'ab' in col.lower()]
        tau_cols = [col for col in biomarker_cols if 'tau' in col.lower()]
        
        if amyloid_cols and tau_cols:
            # Normalizar cada componente
            amyloid_norm = (df[amyloid_cols[0]] - df[amyloid_cols[0]].mean()) / df[amyloid_cols[0]].std()
            tau_norm = (df[tau_cols[0]] - df[tau_cols[0]].mean()) / df[tau_cols[0]].std()
            
            # Score compuesto (invertir amiloide porque valores bajos son patológicos)
            df['atn_composite_score'] = -amyloid_norm + tau_norm
            self.feature_names.append('atn_composite_score')
        
        # Score de riesgo de progresión
        if len(biomarker_cols) >= 3:
            # Usar los primeros 3 biomarcadores más correlacionados
            biomarker_subset = df[biomarker_cols[:3]].copy()
            biomarker_norm = (biomarker_subset - biomarker_subset.mean()) / biomarker_subset.std()
            
            df['biomarker_risk_score'] = biomarker_norm.mean(axis=1)
            self.feature_names.append('biomarker_risk_score')
    
    def _apply_statistical_transforms(self, df, biomarker_cols):
        """Aplica transformaciones estadísticas robustas."""
        
        for col in biomarker_cols[:3]:  # Limitar para eficiencia
            if col in df.columns:
                # Log transform para distribuciones sesgadas
                if df[col].min() > 0:
                    skewness = df[col].skew()
                    if abs(skewness) > 1.5:
                        log_col = f"{col}_log"
                        df[log_col] = np.log1p(df[col])
                        self.feature_names.append(log_col)
                
                # Percentiles para robustez
                percentile_col = f"{col}_percentile"
                df[percentile_col] = df[col].rank(pct=True)
                self.feature_names.append(percentile_col)
    
    def _create_biomarker_interactions(self, df, biomarker_cols):
        """Crea interacciones entre biomarcadores principales."""
        
        if len(biomarker_cols) >= 2:
            # Interacción multiplicativa entre los dos biomarcadores principales
            col1, col2 = biomarker_cols[0], biomarker_cols[1]
            
            # Normalizar antes de multiplicar
            col1_norm = (df[col1] - df[col1].mean()) / df[col1].std()
            col2_norm = (df[col2] - df[col2].mean()) / df[col2].std()
            
            interaction_col = f"{col1}_{col2}_interaction"
            df[interaction_col] = col1_norm * col2_norm
            self.feature_names.append(interaction_col)
            
            # Diferencia normalizada (efecto diferencial)
            diff_col = f"{col1}_{col2}_diff"
            df[diff_col] = col1_norm - col2_norm
            self.feature_names.append(diff_col)

def engineer_biomarker_features(df):
    """
    Función principal para feature engineering de biomarcadores.
    
    Args:
        df (pd.DataFrame): Dataset multimodal
        
    Returns:
        pd.DataFrame: Dataset con características de biomarcadores procesadas
    """
    engineer = BiomarkerFeatureEngineer()
    return engineer.transform(df)

