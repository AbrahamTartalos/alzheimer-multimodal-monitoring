"""
Feature Engineering - Datos Demográficos
========================================
Procesamiento de características demográficas y socioeconómicas
con relevancia clínica para estratificación de riesgo de Alzheimer.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DemographicFeatureEngineer:
    """
    Procesador de características demográficas para Alzheimer.
    
    Incluye edad, género, educación, etnia y factores socioeconómicos
    con transformaciones clínicamente relevantes.
    """
    
    def __init__(self):
        self.feature_names = []
        self.age_risk_thresholds = {
            'early_onset': 65,      # Alzheimer de inicio temprano
            'high_risk': 75,        # Alto riesgo
            'very_high_risk': 85    # Muy alto riesgo
        }
        
    def transform(self, df):
        """
        Aplica feature engineering a datos demográficos.
        
        Args:
            df (pd.DataFrame): Dataset con columnas demográficas
            
        Returns:
            pd.DataFrame: Dataset con características demográficas procesadas
        """
        logger.info("👥 Iniciando feature engineering de demografía...")
        
        df_features = df.copy()
        
        # Identificar columnas demográficas
        demographic_cols = self._identify_demographic_columns(df)
        
        if not demographic_cols:
            logger.warning("No se encontraron columnas demográficas")
            return df_features
            
        logger.info(f"📊 Procesando {len(demographic_cols)} variables demográficas")
        
        # 1. Transformaciones de edad
        self._process_age_features(df_features)
        
        # 2. Procesamiento de género
        self._process_gender_features(df_features)
        
        # 3. Procesamiento de educación
        self._process_education_features(df_features)
        
        # 4. Procesamiento de etnia/raza
        self._process_ethnicity_features(df_features)
        
        # 5. Interacciones demográficas
        self._create_demographic_interactions(df_features)
        
        # 6. Factores de riesgo compuestos
        self._create_risk_composite_scores(df_features)
        
        logger.info(f"✅ Feature engineering completado. Nuevas características: {len(self.feature_names)}")
        
        return df_features
    
    def _identify_demographic_columns(self, df):
        """Identifica columnas demográficas en el dataset."""
        
        demographic_keywords = [
            'age', 'gender', 'sex', 'education', 'educ', 'race', 'ethnicity',
            'ethnic', 'income', 'socioeconomic', 'marital', 'marriage',
            'employment', 'occupation', 'rural', 'urban', 'geographic'
        ]
        
        demographic_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in demographic_keywords):
                demographic_cols.append(col)
        
        return demographic_cols
    
    def _process_age_features(self, df):
        """Procesa características relacionadas con la edad."""
        
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        
        for age_col in age_cols:
            if age_col in df.columns and df[age_col].dtype in ['float64', 'int64']:
                
                # Categorías de riesgo por edad
                age_risk_col = f"{age_col}_risk_category"
                df[age_risk_col] = pd.cut(df[age_col], 
                                        bins=[0, 65, 75, 85, 120],
                                        labels=[0, 1, 2, 3],  # 0=bajo, 3=muy alto
                                        include_lowest=True).astype(float)
                self.feature_names.append(age_risk_col)
                
                # Edad estandarizada
                age_std_col = f"{age_col}_standardized"
                df[age_std_col] = (df[age_col] - df[age_col].mean()) / df[age_col].std()
                self.feature_names.append(age_std_col)
                
                # Flags de riesgo específicos
                early_onset_col = f"{age_col}_early_onset_risk"
                df[early_onset_col] = (df[age_col] < 65).astype(int)
                self.feature_names.append(early_onset_col)
                
                high_risk_col = f"{age_col}_high_risk"
                df[high_risk_col] = (df[age_col] >= 75).astype(int)
                self.feature_names.append(high_risk_col)
                
                # Edad cuadrática (relación no lineal con riesgo)
                age_squared_col = f"{age_col}_squared"
                df[age_squared_col] = df[age_col] ** 2
                self.feature_names.append(age_squared_col)
    
    def _process_gender_features(self, df):
        """Procesa características de género."""
        
        gender_cols = [col for col in df.columns if 
                      any(keyword in col.lower() for keyword in ['gender', 'sex'])]
        
        for gender_col in gender_cols:
            if gender_col in df.columns:
                
                # Encoding binario robusto
                if df[gender_col].dtype == 'object':
                    # Mapear valores comunes
                    gender_mapping = {
                        'male': 1, 'female': 0, 'm': 1, 'f': 0,
                        'male': 1, 'female': 0, '1': 1, '0': 0,
                        'man': 1, 'woman': 0
                    }
                    
                    gender_binary_col = f"{gender_col}_binary"
                    df[gender_binary_col] = df[gender_col].str.lower().map(gender_mapping)
                    self.feature_names.append(gender_binary_col)
                
                # Indicador de género femenino (mayor riesgo en algunas edades)
                female_col = f"{gender_col}_female"
                if 'gender_binary' in df.columns:
                    df[female_col] = (df[f"{gender_col}_binary"] == 0).astype(int)
                    self.feature_names.append(female_col)
    
    def _process_education_features(self, df):
        """Procesa características educativas (factor protector)."""
        
        education_cols = [col for col in df.columns if 
                         any(keyword in col.lower() for keyword in ['education', 'educ'])]
        
        for educ_col in education_cols:
            if educ_col in df.columns and df[educ_col].dtype in ['float64', 'int64']:
                
                # Categorías educativas estándar
                educ_cat_col = f"{educ_col}_category"
                df[educ_cat_col] = pd.cut(df[educ_col], 
                                        bins=[0, 8, 12, 16, 25],
                                        labels=[0, 1, 2, 3],  # 0=primaria, 3=postgrado
                                        include_lowest=True).astype(float)
                self.feature_names.append(educ_cat_col)
                
                # Factor protector (reserva cognitiva)
                cognitive_reserve_col = f"{educ_col}_cognitive_reserve"
                df[cognitive_reserve_col] = np.log1p(df[educ_col])  # Log para rendimientos decrecientes
                self.feature_names.append(cognitive_reserve_col)
                
                # Indicador de baja educación (factor de riesgo)
                low_educ_col = f"{educ_col}_low_risk"
                df[low_educ_col] = (df[educ_col] < 12).astype(int)  # Menos de secundaria
                self.feature_names.append(low_educ_col)
                
                # Indicador de alta educación (factor protector)
                high_educ_col = f"{educ_col}_high_protection"
                df[high_educ_col] = (df[educ_col] >= 16).astype(int)  # Universitario o más
                self.feature_names.append(high_educ_col)
    
    def _process_ethnicity_features(self, df):
        """Procesa características étnicas/raciales."""
        
        ethnicity_cols = [col for col in df.columns if 
                         any(keyword in col.lower() for keyword in ['race', 'ethnicity', 'ethnic'])]
        
        for eth_col in ethnicity_cols:
            if eth_col in df.columns:
                
                # One-hot encoding para principales grupos étnicos
                if df[eth_col].dtype == 'object':
                    # Identificar valores más comunes
                    top_ethnicities = df[eth_col].value_counts().head(4).index
                    
                    for ethnicity in top_ethnicities:
                        eth_binary_col = f"{eth_col}_{ethnicity.lower().replace(' ', '_')}"
                        df[eth_binary_col] = (df[eth_col] == ethnicity).astype(int)
                        self.feature_names.append(eth_binary_col)
                
                # Indicador de diversidad étnica
                ethnicity_diversity_col = f"{eth_col}_diversity_flag"
                if df[eth_col].dtype == 'object':
                    minority_groups = df[eth_col].value_counts().iloc[1:].index  # Excluir el más común
                    df[ethnicity_diversity_col] = df[eth_col].isin(minority_groups).astype(int)
                    self.feature_names.append(ethnicity_diversity_col)
    
    def _create_demographic_interactions(self, df):
        """Crea interacciones entre variables demográficas."""
        
        # Interacción edad-género
        age_cols = [col for col in df.columns if 'age' in col.lower() and 'standardized' in col]
        gender_cols = [col for col in df.columns if 'gender' in col.lower() and 'binary' in col]
        
        if age_cols and gender_cols:
            age_col, gender_col = age_cols[0], gender_cols[0]
            
            # Interacción multiplicativa
            interaction_col = f"age_gender_interaction"
            df[interaction_col] = df[age_col] * df[gender_col]
            self.feature_names.append(interaction_col)
        
        # Interacción edad-educación
        age_cols = [col for col in df.columns if 'age' in col.lower() and df[col].dtype in ['float64', 'int64']]
        educ_cols = [col for col in df.columns if 'education' in col.lower() and df[col].dtype in ['float64', 'int64']]
        
        if age_cols and educ_cols:
            age_col, educ_col = age_cols[0], educ_cols[0]
            
            # Ratio edad/educación (mayor edad, menor educación = mayor riesgo)
            age_educ_ratio_col = f"age_education_risk_ratio"
            df[age_educ_ratio_col] = df[age_col] / (df[educ_col] + 1)  # +1 para evitar división por 0
            self.feature_names.append(age_educ_ratio_col)
    
    def _create_risk_composite_scores(self, df):
        """Crea scores compuestos de riesgo demográfico."""
        
        # Score de riesgo demográfico
        risk_components = []
        
        # Componente edad
        age_cols = [col for col in df.columns if 'age_standardized' in col]
        if age_cols:
            risk_components.append(df[age_cols[0]])
        
        # Componente educación (inverso - más educación = menos riesgo)
        educ_cols = [col for col in df.columns if 'cognitive_reserve' in col]
        if educ_cols:
            risk_components.append(-df[educ_cols[0]])  # Negativo porque es protector
        
        # Componente género (si aplica)
        gender_cols = [col for col in df.columns if 'gender_female' in col]
        if gender_cols:
            risk_components.append(df[gender_cols[0]] * 0.5)  # Peso menor
        
        if len(risk_components) >= 2:
            # Normalizar componentes y crear score compuesto
            risk_matrix = pd.concat(risk_components, axis=1)
            risk_normalized = (risk_matrix - risk_matrix.mean()) / risk_matrix.std()
            
            df['demographic_risk_score'] = risk_normalized.mean(axis=1)
            self.feature_names.append('demographic_risk_score')
            
            # Categorización del score de riesgo
            df['demographic_risk_category'] = pd.cut(df['demographic_risk_score'],
                                                   bins=[-np.inf, -0.5, 0, 0.5, np.inf],
                                                   labels=[0, 1, 2, 3]).astype(float)
            self.feature_names.append('demographic_risk_category')

def engineer_demographic_features(df):
    """
    Función principal para feature engineering de datos demográficos.
    
    Args:
        df (pd.DataFrame): Dataset multimodal
        
    Returns:
        pd.DataFrame: Dataset con características demográficas procesadas
    """
    engineer = DemographicFeatureEngineer()
    return engineer.transform(df)

