"""
Feature Engineering - Datos Clínicos
====================================
Procesamiento de evaluaciones neuropsicológicas, escalas clínicas
y datos temporales para caracterizar progresión cognitiva.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ClinicalFeatureEngineer:
    """
    Procesador de características clínicas para Alzheimer.
    
    Incluye escalas cognitivas, evaluaciones funcionales,
    progresión temporal y detección de cambios sutiles.
    """
    
    def __init__(self):
        self.feature_names = []
        self.cognitive_scales = {
            'mmse': (30, 24),      # (max_score, mild_impairment_threshold)
            'moca': (30, 26),      # Montreal Cognitive Assessment
            'cdr': (0, 0.5),       # Clinical Dementia Rating (0=normal, 0.5=very mild)
            'adas': (0, 12),       # ADAS-Cog (0=best, higher=worse)
        }
        
    def transform(self, df):
        """
        Aplica feature engineering a datos clínicos.
        
        Args:
            df (pd.DataFrame): Dataset con columnas clínicas
            
        Returns:
            pd.DataFrame: Dataset con características clínicas procesadas
        """
        logger.info("🏥 Iniciando feature engineering de datos clínicos...")
        
        df_features = df.copy()
        
        # Identificar columnas clínicas
        clinical_cols = self._identify_clinical_columns(df)
        temporal_cols = self._identify_temporal_columns(df)
        
        if not clinical_cols:
            logger.warning("No se encontraron columnas clínicas")
            return df_features
            
        logger.info(f"📊 Procesando {len(clinical_cols)} variables clínicas")
        
        # 1. Normalización de escalas cognitivas
        self._normalize_cognitive_scales(df_features, clinical_cols)
        
        # 2. Categorización por severidad
        self._categorize_by_severity(df_features, clinical_cols)
        
        # 3. Features temporales y de progresión
        if temporal_cols:
            self._create_temporal_features(df_features, temporal_cols)
        
        # 4. Scores compuestos cognitivos
        self._create_composite_cognitive_scores(df_features, clinical_cols)
        
        # 5. Detección de declive cognitivo
        self._detect_cognitive_decline(df_features, clinical_cols)
        
        # 6. Features de variabilidad intra-sujeto
        self._create_variability_features(df_features, clinical_cols)
        
        logger.info(f"✅ Feature engineering completado. Nuevas características: {len(self.feature_names)}")
        
        return df_features
    
    def _identify_clinical_columns(self, df):
        """Identifica columnas de evaluaciones clínicas."""
        
        clinical_keywords = [
            'mmse', 'moca', 'cdr', 'adas', 'cognitive', 'memory', 'attention',
            'executive', 'language', 'visuospatial', 'orientation', 'recall',
            'recognition', 'fluency', 'trails', 'digit', 'clock', 'boston',
            'functional', 'adl', 'iadl', 'npi', 'gds', 'depression'
        ]
        
        clinical_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in clinical_keywords):
                if df[col].dtype in ['float64', 'int64']:
                    clinical_cols.append(col)
        
        return clinical_cols
    
    def _identify_temporal_columns(self, df):
        """Identifica columnas temporales para análisis de progresión."""
        
        temporal_keywords = ['date', 'visit', 'time', 'days_since', 'baseline']
        
        temporal_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in temporal_keywords):
                temporal_cols.append(col)
        
        return temporal_cols
    
    def _normalize_cognitive_scales(self, df, clinical_cols):
        """Normaliza escalas cognitivas a rangos 0-1."""
        
        for scale, (max_score, threshold) in self.cognitive_scales.items():
            matching_cols = [col for col in clinical_cols if scale in col.lower()]
            
            for col in matching_cols:
                if col in df.columns:
                    # Normalización 0-1
                    normalized_col = f"{col}_normalized"
                    df[normalized_col] = df[col] / max_score
                    self.feature_names.append(normalized_col)
                    
                    # Distancia del umbral de deterioro
                    threshold_col = f"{col}_threshold_distance"
                    df[threshold_col] = (df[col] - threshold) / max_score
                    self.feature_names.append(threshold_col)
    
    def _categorize_by_severity(self, df, clinical_cols):
        """Categoriza por niveles de severidad cognitiva."""
        
        # MMSE categorization
        mmse_cols = [col for col in clinical_cols if 'mmse' in col.lower()]
        for col in mmse_cols:
            if col in df.columns:
                severity_col = f"{col}_severity"
                df[severity_col] = pd.cut(df[col], 
                                        bins=[-1, 17, 23, 26, 30], 
                                        labels=[3, 2, 1, 0],  # 3=severe, 0=normal
                                        include_lowest=True).astype(float)
                self.feature_names.append(severity_col)
        
        # CDR categorization  
        cdr_cols = [col for col in clinical_cols if 'cdr' in col.lower()]
        for col in cdr_cols:
            if col in df.columns:
                # CDR binary (0=normal, ≥0.5=impaired)
                binary_col = f"{col}_impaired"
                df[binary_col] = (df[col] >= 0.5).astype(int)
                self.feature_names.append(binary_col)
    
    def _create_temporal_features(self, df, temporal_cols):
        """Crea características temporales y de progresión."""
        
        # Días desde baseline
        baseline_cols = [col for col in temporal_cols if 'baseline' in col.lower()]
        if baseline_cols:
            baseline_col = baseline_cols[0]
            
            # Tiempo en años
            years_col = f"{baseline_col}_years"
            df[years_col] = df[baseline_col] / 365.25
            self.feature_names.append(years_col)
            
            # Categorías temporales
            temporal_cat_col = f"{baseline_col}_category"
            df[temporal_cat_col] = pd.cut(df[baseline_col], 
                                        bins=[-1, 180, 365, 730, float('inf')],
                                        labels=[0, 1, 2, 3]).astype(float)  # 0-6m, 6m-1y, 1-2y, >2y
            self.feature_names.append(temporal_cat_col)
        
        # Edad del sujeto en la visita (si disponible)
        visit_date_cols = [col for col in temporal_cols if 'visit' in col.lower() and 'date' in col.lower()]
        if visit_date_cols and 'AGE' in df.columns:
            # Calcular edad aproximada en la visita
            age_at_visit_col = 'age_at_visit_estimated'
            if baseline_cols:
                df[age_at_visit_col] = df['AGE'] + (df[baseline_cols[0]] / 365.25)
                self.feature_names.append(age_at_visit_col)
    
    def _create_composite_cognitive_scores(self, df, clinical_cols):
        """Crea scores cognitivos compuestos."""
        
        # Score cognitivo global
        cognitive_subscales = []
        for keyword in ['memory', 'attention', 'executive', 'language']:
            matching_cols = [col for col in clinical_cols if keyword in col.lower()]
            if matching_cols:
                cognitive_subscales.extend(matching_cols[:2])  # Máximo 2 por dominio
        
        if len(cognitive_subscales) >= 3:
            # Normalizar y promediar
            cognitive_data = df[cognitive_subscales].copy()
            cognitive_normalized = (cognitive_data - cognitive_data.mean()) / cognitive_data.std()
            
            df['composite_cognitive_score'] = cognitive_normalized.mean(axis=1)
            self.feature_names.append('composite_cognitive_score')
        
        # Score de deterioro funcional
        functional_keywords = ['adl', 'iadl', 'functional']
        functional_cols = []
        for keyword in functional_keywords:
            matching_cols = [col for col in clinical_cols if keyword in col.lower()]
            functional_cols.extend(matching_cols)
        
        if functional_cols:
            functional_data = df[functional_cols[:3]].copy()  # Máximo 3 escalas
            functional_normalized = (functional_data - functional_data.mean()) / functional_data.std()
            
            df['functional_impairment_score'] = functional_normalized.mean(axis=1)
            self.feature_names.append('functional_impairment_score')
    
    def _detect_cognitive_decline(self, df, clinical_cols):
        """Detecta patrones de declive cognitivo."""
        
        # Para cada escala principal, detectar valores por debajo de percentiles críticos
        main_scales = ['mmse', 'moca', 'cdr', 'adas']
        
        for scale in main_scales:
            matching_cols = [col for col in clinical_cols if scale in col.lower()]
            
            for col in matching_cols:
                if col in df.columns:
                    # Percentil dentro de la población
                    percentile_col = f"{col}_percentile"
                    df[percentile_col] = df[col].rank(pct=True)
                    self.feature_names.append(percentile_col)
                    
                    # Flag de deterioro severo (percentil <10 para escalas positivas)
                    if scale in ['mmse', 'moca']:  # Escalas donde más alto = mejor
                        decline_col = f"{col}_severe_decline"
                        df[decline_col] = (df[percentile_col] < 0.10).astype(int)
                    else:  # Escalas donde más bajo = mejor (CDR, ADAS)
                        decline_col = f"{col}_severe_decline"
                        df[decline_col] = (df[percentile_col] > 0.90).astype(int)
                    
                    self.feature_names.append(decline_col)
    
    def _create_variability_features(self, df, clinical_cols):
        """Crea features de variabilidad y consistencia cognitiva."""
        
        # Variabilidad entre dominios cognitivos
        domain_scores = []
        domains = ['memory', 'attention', 'executive', 'language']
        
        for domain in domains:
            domain_cols = [col for col in clinical_cols if domain in col.lower()]
            if domain_cols:
                # Promedio del dominio (normalizado)
                domain_data = df[domain_cols[:2]].copy()  # Máximo 2 tests por dominio
                if len(domain_data.columns) > 0:
                    domain_norm = (domain_data - domain_data.mean()) / domain_data.std()
                    domain_score = domain_norm.mean(axis=1)
                    domain_scores.append(domain_score)
        
        if len(domain_scores) >= 3:
            # Coeficiente de variación entre dominios
            domain_matrix = pd.concat(domain_scores, axis=1)
            domain_matrix.columns = [f'domain_{i}' for i in range(len(domain_scores))]
            
            df['cognitive_variability'] = domain_matrix.std(axis=1) / (domain_matrix.mean(axis=1).abs() + 1e-8)
            self.feature_names.append('cognitive_variability')
            
            # Patrón de fortalezas/debilidades
            df['cognitive_profile_range'] = domain_matrix.max(axis=1) - domain_matrix.min(axis=1)
            self.feature_names.append('cognitive_profile_range')

def engineer_clinical_features(df):
    """
    Función principal para feature engineering de datos clínicos.
    
    Args:
        df (pd.DataFrame): Dataset multimodal
        
    Returns:
        pd.DataFrame: Dataset con características clínicas procesadas
    """
    engineer = ClinicalFeatureEngineer()
    return engineer.transform(df)

