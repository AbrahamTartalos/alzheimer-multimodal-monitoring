"""
Feature Engineering - PET (Tomografía por Emisión de Positrones)
================================================================
Procesamiento de biomarcadores de neuroimagen molecular: amiloide, tau,
metabolismo cerebral y conversión a escalas clínicas estándar.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)

class PETFeatureEngineer:
    """
    Procesador de características PET para Alzheimer.
    
    Transforma datos de PET amiloide, tau y FDG en características
    clínicamente interpretables con umbrales diagnósticos establecidos.
    """
    
    def __init__(self):
        self.feature_names = []
        self.clinical_thresholds = {
            'amyloid_suvr': 1.11,      # Umbral positividad amiloide
            'centiloid': 20,           # Escala Centiloid estándar
            'tau_suvr': 1.3,           # Umbral tau patológico
            'fdg_suvr': 1.2,           # Umbral hipometabolismo
        }
        
    def transform(self, df):
        """
        Aplica feature engineering a datos PET.
        
        Args:
            df (pd.DataFrame): Dataset con columnas PET
            
        Returns:
            pd.DataFrame: Dataset con características PET procesadas
        """
        logger.info("🔬 Iniciando feature engineering de PET...")
        
        df_features = df.copy()
        
        # Identificar columnas PET por modalidad
        pet_cols = self._identify_pet_columns(df)
        
        if not pet_cols['all']:
            logger.warning("No se encontraron columnas PET")
            return df_features
            
        logger.info(f"📊 Procesando {len(pet_cols['all'])} variables PET")
        
        # 1. Clasificación por umbrales clínicos
        self._apply_clinical_thresholds(df_features, pet_cols)
        
        # 2. Conversión a escalas estándar
        self._convert_to_standard_scales(df_features, pet_cols)
        
        # 3. Ratios entre regiones cerebrales
        self._create_regional_ratios(df_features, pet_cols)
        
        # 4. Scores compuestos por modalidad
        self._create_pet_composite_scores(df_features, pet_cols)
        
        # 5. Detección de patrones espaciales
        self._detect_spatial_patterns(df_features, pet_cols)
        
        logger.info(f"✅ Feature engineering completado. Nuevas características: {len(self.feature_names)}")
        
        return df_features
    
    def _identify_pet_columns(self, df):
        """Identifica y categoriza columnas PET por modalidad."""
        
        pet_categories = {
            'amyloid': ['amyloid', 'pib', 'florbetapir', 'flutemetamol', 'av45', 'av1451'],
            'tau': ['tau', 'av1451', 'mk6240', 'pi2620'],
            'fdg': ['fdg', 'glucose', 'metabolism', 'metabolic'],
            'suvr': ['suvr', 'suv', 'uptake', 'binding'],
            'centiloid': ['centiloid', 'cl', 'standardized']
        }
        
        pet_cols = {'all': [], 'amyloid': [], 'tau': [], 'fdg': [], 'suvr': [], 'centiloid': []}
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                col_lower = col.lower()
                
                # Categorizar por tipo de PET
                for category, keywords in pet_categories.items():
                    if any(keyword in col_lower for keyword in keywords):
                        pet_cols[category].append(col)
                        if col not in pet_cols['all']:
                            pet_cols['all'].append(col)
                        break
                
                # Catch-all para otras columnas PET
                if 'pet' in col_lower and col not in pet_cols['all']:
                    pet_cols['all'].append(col)
        
        return pet_cols
    
    def _apply_clinical_thresholds(self, df, pet_cols):
        """Aplica umbrales clínicos para clasificación binaria."""
        
        # Amiloide SUVR > 1.11 = positivo
        for col in pet_cols['amyloid'] + pet_cols['suvr']:
            if 'amyloid' in col.lower() or 'pib' in col.lower():
                binary_col = f"{col}_amyloid_positive"
                df[binary_col] = (df[col] > self.clinical_thresholds['amyloid_suvr']).astype(int)
                self.feature_names.append(binary_col)
        
        # Centiloid > 20 = positivo
        for col in pet_cols['centiloid']:
            binary_col = f"{col}_positive"
            df[binary_col] = (df[col] > self.clinical_thresholds['centiloid']).astype(int)
            self.feature_names.append(binary_col)
        
        # Tau SUVR > 1.3 = elevado
        for col in pet_cols['tau']:
            if 'suvr' in col.lower():
                binary_col = f"{col}_tau_elevated"
                df[binary_col] = (df[col] > self.clinical_thresholds['tau_suvr']).astype(int)
                self.feature_names.append(binary_col)
        
        # FDG hipometabolismo (valores bajos son patológicos)
        for col in pet_cols['fdg']:
            binary_col = f"{col}_hypometabolism"
            df[binary_col] = (df[col] < df[col].quantile(0.25)).astype(int)  # Cuartil inferior
            self.feature_names.append(binary_col)
    
    def _convert_to_standard_scales(self, df, pet_cols):
        """Convierte a escalas clínicas estándar."""
        
        # Conversión aproximada SUVR a Centiloid (fórmula estándar)
        amyloid_suvr_cols = [col for col in pet_cols['amyloid'] if 'suvr' in col.lower()]
        
        for col in amyloid_suvr_cols:
            if col in df.columns:
                # Fórmula aproximada: CL = 188.22 * (SUVR - 1.0) - 18.85
                centiloid_col = f"{col}_centiloid_estimated"
                df[centiloid_col] = 188.22 * (df[col] - 1.0) - 18.85
                self.feature_names.append(centiloid_col)
                
                # Z-score respecto a población normal
                zscore_col = f"{col}_zscore"
                df[zscore_col] = (df[col] - df[col].mean()) / df[col].std()
                self.feature_names.append(zscore_col)
    
    def _create_regional_ratios(self, df, pet_cols):
        """Crea ratios entre regiones cerebrales."""
        
        # Identificar regiones específicas
        regions = ['cortical', 'subcortical', 'frontal', 'temporal', 'parietal', 'occipital']
        region_cols = {}
        
        for region in regions:
            region_cols[region] = [col for col in pet_cols['all'] if region in col.lower()]
        
        # Ratio cortical/subcortical (patrón típico en Alzheimer)
        if region_cols['cortical'] and region_cols['subcortical']:
            ratio_col = 'pet_cortical_subcortical_ratio'
            df[ratio_col] = (df[region_cols['cortical'][0]] / 
                           (df[region_cols['subcortical'][0]] + 1e-8))
            self.feature_names.append(ratio_col)
        
        # Ratio temporal/frontal (progresión típica)
        if region_cols['temporal'] and region_cols['frontal']:
            ratio_col = 'pet_temporal_frontal_ratio'
            df[ratio_col] = (df[region_cols['temporal'][0]] / 
                           (df[region_cols['frontal'][0]] + 1e-8))
            self.feature_names.append(ratio_col)
    
    def _create_pet_composite_scores(self, df, pet_cols):
        """Crea scores compuestos por modalidad PET."""
        
        # Score de carga amiloide global
        if len(pet_cols['amyloid']) >= 2:
            amyloid_data = df[pet_cols['amyloid'][:3]].copy()  # Máximo 3 regiones
            amyloid_normalized = (amyloid_data - amyloid_data.mean()) / amyloid_data.std()
            
            df['amyloid_burden_score'] = amyloid_normalized.mean(axis=1)
            self.feature_names.append('amyloid_burden_score')
        
        # Score de patología tau
        if len(pet_cols['tau']) >= 2:
            tau_data = df[pet_cols['tau'][:3]].copy()
            tau_normalized = (tau_data - tau_data.mean()) / tau_data.std()
            
            df['tau_pathology_score'] = tau_normalized.mean(axis=1)
            self.feature_names.append('tau_pathology_score')
        
        # Score de hipometabolismo (FDG)
        if len(pet_cols['fdg']) >= 2:
            fdg_data = df[pet_cols['fdg'][:3]].copy()
            fdg_normalized = (fdg_data - fdg_data.mean()) / fdg_data.std()
            
            # Invertir porque valores bajos indican patología
            df['hypometabolism_score'] = -fdg_normalized.mean(axis=1)
            self.feature_names.append('hypometabolism_score')
    
    def _detect_spatial_patterns(self, df, pet_cols):
        """Detecta patrones espaciales característicos de Alzheimer."""
        
        # Patrón de vulnerabilidad regional (coeficiente de variación)
        if len(pet_cols['all']) >= 3:
            pet_subset = df[pet_cols['all'][:5]].copy()  # Máximo 5 regiones
            
            # Coeficiente de variación inter-regional
            cv_col = 'pet_regional_variability'
            df[cv_col] = pet_subset.std(axis=1) / (pet_subset.mean(axis=1) + 1e-8)
            self.feature_names.append(cv_col)
            
            # Asimetría inter-hemisférica (si hay datos lateralizados)
            left_cols = [col for col in pet_cols['all'] if 'left' in col.lower() or '_l_' in col.lower()]
            right_cols = [col for col in pet_cols['all'] if 'right' in col.lower() or '_r_' in col.lower()]
            
            if left_cols and right_cols:
                asymmetry_col = 'pet_hemispheric_asymmetry'
                # Usar la primera región disponible de cada hemisferio
                df[asymmetry_col] = (df[left_cols[0]] - df[right_cols[0]]) / (df[left_cols[0]] + df[right_cols[0]] + 1e-8)
                self.feature_names.append(asymmetry_col)

def engineer_pet_features(df):
    """
    Función principal para feature engineering de PET.
    
    Args:
        df (pd.DataFrame): Dataset multimodal
        
    Returns:
        pd.DataFrame: Dataset con características PET procesadas
    """
    engineer = PETFeatureEngineer()
    return engineer.transform(df)

