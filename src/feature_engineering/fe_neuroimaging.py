"""
Feature Engineering - Neuroimagen (MRI)
========================================
Extracción y transformación de características de neuroimagen estructural
relevantes para la detección temprana de Alzheimer.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class NeuroImagingFeatureEngineer:
    """
    Procesador de características de neuroimagen MRI para Alzheimer.
    
    Transforma datos de MRI en características clínicamente relevantes
    incluyendo medidas de atrofia, volúmenes normalizados y ratios críticos.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()  # Robusto a outliers
        self.feature_names = []
        
    def transform(self, df):
        """
        Aplica transformaciones de feature engineering a datos de MRI.
        
        Args:
            df (pd.DataFrame): Dataset con columnas de neuroimagen
            
        Returns:
            pd.DataFrame: Dataset con nuevas características de neuroimagen
        """
        logger.info("🧠 Iniciando feature engineering de neuroimagen...")
        
        df_features = df.copy()
        
        # Identificar columnas de neuroimagen
        mri_columns = [col for col in df.columns if 
                      any(keyword in col.lower() for keyword in 
                          ['mri', 'volume', 'thickness', 'area', 'hippocampus', 
                           'ventricle', 'cortical', 'subcortical', 'brain'])]
        
        if not mri_columns:
            logger.warning("No se encontraron columnas de neuroimagen")
            return df_features
            
        logger.info(f"📊 Procesando {len(mri_columns)} variables de neuroimagen")
        
        # 1. Normalización por volumen cerebral total (si disponible)
        brain_volume_cols = [col for col in mri_columns if 
                           'total' in col.lower() or 'brain' in col.lower()]
        
        if brain_volume_cols:
            self._create_normalized_volumes(df_features, mri_columns, brain_volume_cols[0])
        
        # 2. Ratios clínicamente relevantes
        self._create_clinical_ratios(df_features, mri_columns)
        
        # 3. Scores de atrofia compuestos
        self._create_atrophy_scores(df_features, mri_columns)
        
        # 4. Transformaciones de distribución
        self._apply_distributional_transforms(df_features, mri_columns)
        
        # 5. Detección de outliers neuroanatómicos
        self._flag_neuroanatomical_outliers(df_features, mri_columns)
        
        logger.info(f"✅ Feature engineering completado. Nuevas características: {len(self.feature_names)}")
        
        return df_features
    
    def _create_normalized_volumes(self, df, mri_cols, brain_vol_col):
        """Crea volúmenes normalizados por volumen cerebral total."""
        
        volume_cols = [col for col in mri_cols if 'volume' in col.lower() and col != brain_vol_col]
        
        for col in volume_cols:
            if col in df.columns and brain_vol_col in df.columns:
                # Ratio respecto al volumen cerebral total
                new_col = f"{col}_normalized"
                df[new_col] = df[col] / df[brain_vol_col]
                self.feature_names.append(new_col)
                
                # Percentil respecto a población (z-score)
                new_col_z = f"{col}_zscore"
                df[new_col_z] = (df[col] - df[col].mean()) / df[col].std()
                self.feature_names.append(new_col_z)
    
    def _create_clinical_ratios(self, df, mri_cols):
        """Crea ratios clínicamente relevantes para Alzheimer."""
        
        # Ratio hipocampo/ventrículo (indicador clave de atrofia)
        hippocampus_cols = [col for col in mri_cols if 'hippocampus' in col.lower()]
        ventricle_cols = [col for col in mri_cols if 'ventricle' in col.lower()]
        
        if hippocampus_cols and ventricle_cols:
            hip_col, vent_col = hippocampus_cols[0], ventricle_cols[0]
            df['hippocampus_ventricle_ratio'] = df[hip_col] / (df[vent_col] + 1e-8)  # Evitar división por 0
            self.feature_names.append('hippocampus_ventricle_ratio')
        
        # Ratio cortical/subcortical
        cortical_cols = [col for col in mri_cols if 'cortical' in col.lower()]
        subcortical_cols = [col for col in mri_cols if 'subcortical' in col.lower()]
        
        if cortical_cols and subcortical_cols:
            cort_col, subcort_col = cortical_cols[0], subcortical_cols[0]
            df['cortical_subcortical_ratio'] = df[cort_col] / (df[subcort_col] + 1e-8)
            self.feature_names.append('cortical_subcortical_ratio')
    
    def _create_atrophy_scores(self, df, mri_cols):
        """Crea scores compuestos de atrofia cerebral."""
        
        # Score de atrofia temporal (crítico en Alzheimer)
        temporal_keywords = ['hippocampus', 'temporal', 'entorhinal', 'parahippocampal']
        temporal_cols = [col for col in mri_cols if 
                        any(keyword in col.lower() for keyword in temporal_keywords)]
        
        if len(temporal_cols) >= 2:
            # Normalizar y promediar
            temporal_data = df[temporal_cols].copy()
            temporal_normalized = (temporal_data - temporal_data.mean()) / temporal_data.std()
            df['temporal_atrophy_score'] = temporal_normalized.mean(axis=1)
            self.feature_names.append('temporal_atrophy_score')
        
        # Score de atrofia global
        volume_cols = [col for col in mri_cols if 'volume' in col.lower()]
        if len(volume_cols) >= 3:
            volume_data = df[volume_cols].copy()
            volume_normalized = (volume_data - volume_data.mean()) / volume_data.std()
            df['global_atrophy_score'] = -volume_normalized.mean(axis=1)  # Negativo: menos volumen = más atrofia
            self.feature_names.append('global_atrophy_score')
    
    def _apply_distributional_transforms(self, df, mri_cols):
        """Aplica transformaciones para normalizar distribuciones."""
        
        for col in mri_cols[:5]:  # Limitar a las primeras 5 para eficiencia
            if col in df.columns:
                # Log transform para distribuciones sesgadas
                if df[col].min() > 0:  # Solo si todos los valores son positivos
                    skewness = df[col].skew()
                    if abs(skewness) > 1:  # Alta asimetría
                        log_col = f"{col}_log"
                        df[log_col] = np.log1p(df[col])  # log(1+x) para estabilidad
                        self.feature_names.append(log_col)
    
    def _flag_neuroanatomical_outliers(self, df, mri_cols):
        """Identifica outliers neuroanatómicos extremos."""
        
        # Usar IQR para detectar outliers extremos
        for col in mri_cols[:3]:  # Limitar para eficiencia
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # 3*IQR para outliers extremos
                upper_bound = Q3 + 3 * IQR
                
                outlier_col = f"{col}_extreme_outlier"
                df[outlier_col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                self.feature_names.append(outlier_col)

def engineer_neuroimaging_features(df):
    """
    Función principal para feature engineering de neuroimagen.
    
    Args:
        df (pd.DataFrame): Dataset multimodal
        
    Returns:
        pd.DataFrame: Dataset con características de neuroimagen procesadas
    """
    engineer = NeuroImagingFeatureEngineer()
    return engineer.transform(df)

