"""
Risk Stratification Algorithms for Alzheimer's Disease
======================================================

Este módulo contiene algoritmos especializados para la estratificación de riesgo
de pacientes con Alzheimer, incluyendo clustering, estratificación jerárquica
y modelos probabilísticos.

Autor: Abraham Tartalos
Fecha: 2025
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PhenotypeIdentifier:
    """
    Identificador de fenotipos de pacientes usando clustering
    """
    
    def __init__(self, method='kmeans', random_state=42):
        self.method = method
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.n_clusters = None
        self.labels_ = None
        
    def fit(self, X: pd.DataFrame, k_range: range = range(2, 8)) -> 'PhenotypeIdentifier':
        """
        Ajusta el modelo de clustering para identificar fenotipos
        
        Args:
            X: DataFrame con features para clustering
            k_range: Rango de número de clusters a evaluar
            
        Returns:
            self: Instancia ajustada
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'kmeans':
            self.n_clusters = self._optimize_kmeans(X_scaled, k_range)
            self.model = KMeans(n_clusters=self.n_clusters, 
                               random_state=self.random_state, n_init=10)
            
        elif self.method == 'gmm':
            self.n_clusters = self._optimize_gmm(X_scaled, k_range)
            self.model = GaussianMixture(n_components=self.n_clusters,
                                        random_state=self.random_state)
            
        elif self.method == 'hierarchical':
            self.n_clusters = self._optimize_hierarchical(X_scaled, k_range)
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
            
        self.labels_ = self.model.fit_predict(X_scaled)
        return self
    
    def _optimize_kmeans(self, X_scaled: np.ndarray, k_range: range) -> int:
        """Optimiza número de clusters para K-means"""
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            
        return k_range[np.argmax(silhouette_scores)]
    
    def _optimize_gmm(self, X_scaled: np.ndarray, k_range: range) -> int:
        """Optimiza número de componentes para GMM"""
        bic_scores = []
        
        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=self.random_state)
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
            
        return k_range[np.argmin(bic_scores)]
    
    def _optimize_hierarchical(self, X_scaled: np.ndarray, k_range: range) -> int:
        """Optimiza número de clusters para clustering jerárquico"""
        silhouette_scores = []
        
        for k in k_range:
            hierarchical = AgglomerativeClustering(n_clusters=k)
            labels = hierarchical.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            
        return k_range[np.argmax(silhouette_scores)]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice fenotipos para nuevos datos"""
        if self.model is None:
            raise ValueError("Modelo no ajustado. Ejecutar fit() primero.")
            
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # Para modelos sin método predict (como AgglomerativeClustering)
            return self.labels_
    
    def get_cluster_characteristics(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analiza características de cada cluster
        
        Args:
            X: Features originales
            y: Variable objetivo (composite_risk_score)
            
        Returns:
            DataFrame con características por cluster
        """
        if self.labels_ is None:
            raise ValueError("Modelo no ajustado.")
            
        analysis_df = X.copy()
        analysis_df['cluster'] = self.labels_
        analysis_df['risk_score'] = y
        
        cluster_stats = analysis_df.groupby('cluster').agg({
            'risk_score': ['mean', 'std', 'count', 'median'],
            **{col: 'mean' for col in X.columns}
        }).round(3)
        
        return cluster_stats


class GeneticStratifier:
    """
    Estratificador basado en características genéticas (principalmente APOE)
    """
    
    def __init__(self):
        self.stratification_rules = {}
        self.risk_profiles = {}
        
    def fit(self, df: pd.DataFrame, genetic_features: List[str], 
            target_col: str = 'composite_risk_score') -> 'GeneticStratifier':
        """
        Ajusta el estratificador genético
        
        Args:
            df: DataFrame con datos
            genetic_features: Lista de features genéticas
            target_col: Columna objetivo
            
        Returns:
            self: Instancia ajustada
        """
        # Estratificación por APOE
        if 'APOE_e4_carrier' in genetic_features:
            self._create_apoe_stratification(df, target_col)
            
        # Estratificación por homocigotos APOE4
        if 'APOE_e4_homozygous' in genetic_features:
            self._create_apoe_homozygous_stratification(df, target_col)
            
        return self
    
    def _create_apoe_stratification(self, df: pd.DataFrame, target_col: str):
        """Crea estratificación basada en APOE e4"""
        apoe_groups = df.groupby('APOE_e4_carrier')[target_col].agg(['mean', 'std', 'count'])
        
        self.risk_profiles['APOE_e4_carrier'] = {
            'non_carrier': {
                'mean_risk': apoe_groups.loc[0, 'mean'],
                'std_risk': apoe_groups.loc[0, 'std'],
                'count': apoe_groups.loc[0, 'count']
            },
            'carrier': {
                'mean_risk': apoe_groups.loc[1, 'mean'],
                'std_risk': apoe_groups.loc[1, 'std'],
                'count': apoe_groups.loc[1, 'count']
            }
        }
        
        # Regla de estratificación
        self.stratification_rules['APOE_e4'] = {
            'high_risk_threshold': apoe_groups.loc[1, 'mean'] - apoe_groups.loc[1, 'std'],
            'genetic_risk_multiplier': apoe_groups.loc[1, 'mean'] / apoe_groups.loc[0, 'mean']
        }
    
    def _create_apoe_homozygous_stratification(self, df: pd.DataFrame, target_col: str):
        """Crea estratificación para homocigotos APOE4"""
        if 'APOE_e4_homozygous' in df.columns:
            homozygous_analysis = df.groupby('APOE_e4_homozygous')[target_col].agg(['mean', 'std', 'count'])
            
            self.risk_profiles['APOE_e4_homozygous'] = {
                'heterozygous_or_none': {
                    'mean_risk': homozygous_analysis.loc[0, 'mean'],
                    'count': homozygous_analysis.loc[0, 'count']
                },
                'homozygous': {
                    'mean_risk': homozygous_analysis.loc[1, 'mean'],
                    'count': homozygous_analysis.loc[1, 'count']
                }
            }
    
    def stratify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica estratificación genética a los datos
        
        Args:
            df: DataFrame con datos a estratificar
            
        Returns:
            DataFrame con columnas de estratificación añadidas
        """
        result_df = df.copy()
        
        # Estratificación APOE
        if 'APOE_e4_carrier' in df.columns:
            result_df['genetic_risk_stratum'] = df['APOE_e4_carrier'].map({
                0: 'Low_Genetic_Risk',
                1: 'High_Genetic_Risk'
            })
            
            # Calcular riesgo genético ajustado
            if 'APOE_e4' in self.stratification_rules:
                multiplier = self.stratification_rules['APOE_e4']['genetic_risk_multiplier']
                result_df['genetic_adjusted_risk'] = np.where(
                    df['APOE_e4_carrier'] == 1,
                    df['composite_risk_score'] * multiplier,
                    df['composite_risk_score']
                )
        
        return result_df


class HierarchicalStratifier:
    """
    Estratificador jerárquico que combina múltiples características
    """
    
    def __init__(self):
        self.hierarchy_rules = {}
        self.substrata_definitions = {}
        
    def define_hierarchy(self, hierarchy_config: Dict[str, Any]) -> 'HierarchicalStratifier':
        """
        Define la jerarquía de estratificación
        
        Args:
            hierarchy_config: Configuración de la jerarquía
            
        Returns:
            self: Instancia configurada
        """
        self.hierarchy_rules = hierarchy_config
        return self
    
    def create_substrata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea subestratos jerárquicos
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con subestratos asignados
        """
        result_df = df.copy()
        result_df['hierarchical_stratum'] = 'Other'
        
        # Nivel 1: Riesgo alto + APOE4
        if all(col in df.columns for col in ['risk_category', 'APOE_e4_carrier']):
            high_risk_apoe = (df['risk_category'] == 'High') & (df['APOE_e4_carrier'] == 1)
            result_df.loc[high_risk_apoe, 'hierarchical_stratum'] = 'Critical_High_Risk_APOE4'
            
            # Nivel 2: Riesgo alto sin APOE4
            high_risk_no_apoe = (df['risk_category'] == 'High') & (df['APOE_e4_carrier'] == 0)
            result_df.loc[high_risk_no_apoe, 'hierarchical_stratum'] = 'High_Risk_Non_APOE4'
        
        # Nivel 3: Riesgo moderado con biomarcadores elevados
        if 'biomarker_risk_score' in df.columns:
            biomarker_threshold = df['biomarker_risk_score'].quantile(0.75)
            mod_risk_high_bio = (df['risk_category'] == 'Moderate') & \
                               (df['biomarker_risk_score'] > biomarker_threshold)
            result_df.loc[mod_risk_high_bio, 'hierarchical_stratum'] = 'Moderate_Risk_High_Biomarkers'
        
        # Nivel 4: Riesgo bajo pero con factores de riesgo
        if all(col in df.columns for col in ['risk_category', 'APOE_e4_carrier']):
            low_risk_apoe = (df['risk_category'] == 'Low') & (df['APOE_e4_carrier'] == 1)
            result_df.loc[low_risk_apoe, 'hierarchical_stratum'] = 'Low_Risk_APOE4_Carrier'
        
        return result_df
    
    def get_stratum_analysis(self, df: pd.DataFrame, target_col: str = 'composite_risk_score') -> pd.DataFrame:
        """
        Analiza características de cada estrato
        
        Args:
            df: DataFrame con estratos asignados
            target_col: Columna objetivo
            
        Returns:
            DataFrame con análisis por estrato
        """
        if 'hierarchical_stratum' not in df.columns:
            raise ValueError("Estratos no asignados. Ejecutar create_substrata() primero.")
        
        stratum_analysis = df.groupby('hierarchical_stratum').agg({
            target_col: ['mean', 'std', 'count', 'median', 'min', 'max']
        }).round(3)
        
        return stratum_analysis


class BiomarkerStratifier:
    """
    Estratificador basado en biomarcadores específicos
    """
    
    def __init__(self, biomarker_features: List[str]):
        self.biomarker_features = biomarker_features
        self.thresholds = {}
        self.clustering_model = None
        self.scaler = StandardScaler()
        
    def fit(self, df: pd.DataFrame, method='clustering') -> 'BiomarkerStratifier':
        """
        Ajusta el estratificador de biomarcadores
        
        Args:
            df: DataFrame con datos
            method: Método de estratificación ('clustering' o 'thresholds')
            
        Returns:
            self: Instancia ajustada
        """
        biomarker_data = df[self.biomarker_features].dropna()
        
        if method == 'clustering':
            self._fit_clustering(biomarker_data)
        elif method == 'thresholds':
            self._fit_thresholds(biomarker_data)
            
        return self
    
    def _fit_clustering(self, biomarker_data: pd.DataFrame):
        """Ajusta clustering para biomarcadores"""
        X_scaled = self.scaler.fit_transform(biomarker_data)
        
        # Optimizar número de clusters
        silhouette_scores = []
        k_range = range(2, 6)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.clustering_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.clustering_model.fit(X_scaled)
    
    def _fit_thresholds(self, biomarker_data: pd.DataFrame):
        """Define umbrales basados en percentiles"""
        for feature in self.biomarker_features:
            self.thresholds[feature] = {
                'low': biomarker_data[feature].quantile(0.33),
                'high': biomarker_data[feature].quantile(0.67)
            }
    
    def stratify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica estratificación por biomarcadores
        
        Args:
            df: DataFrame a estratificar
            
        Returns:
            DataFrame con estratificación aplicada
        """
        result_df = df.copy()
        biomarker_subset = df[self.biomarker_features].dropna()
        
        if self.clustering_model is not None:
            # Estratificación por clustering
            X_scaled = self.scaler.transform(biomarker_subset)
            biomarker_clusters = self.clustering_model.predict(X_scaled)
            
            # Asignar clusters a DataFrame original
            cluster_mapping = pd.Series(biomarker_clusters, index=biomarker_subset.index)
            result_df['biomarker_cluster'] = cluster_mapping
            result_df['biomarker_stratum'] = result_df['biomarker_cluster'].map({
                0: 'Low_Biomarker_Risk',
                1: 'Moderate_Biomarker_Risk', 
                2: 'High_Biomarker_Risk'
            })
            
        elif self.thresholds:
            # Estratificación por umbrales
            result_df['biomarker_stratum'] = 'Moderate_Biomarker_Risk'
            
            # Lógica de umbrales (ejemplo con primer biomarcador)
            first_biomarker = self.biomarker_features[0]
            if first_biomarker in df.columns:
                low_threshold = self.thresholds[first_biomarker]['low']
                high_threshold = self.thresholds[first_biomarker]['high']
                
                result_df.loc[df[first_biomarker] <= low_threshold, 'biomarker_stratum'] = 'Low_Biomarker_Risk'
                result_df.loc[df[first_biomarker] >= high_threshold, 'biomarker_stratum'] = 'High_Biomarker_Risk'
        
        return result_df


def create_comprehensive_stratification(df: pd.DataFrame, 
                                      genetic_features: List[str],
                                      biomarker_features: List[str],
                                      target_col: str = 'composite_risk_score') -> pd.DataFrame:
    """
    Crea estratificación comprehensiva combinando múltiples enfoques
    
    Args:
        df: DataFrame con datos
        genetic_features: Lista de features genéticas
        biomarker_features: Lista de features de biomarcadores
        target_col: Columna objetivo
        
    Returns:
        DataFrame con estratificación comprehensiva
    """
    result_df = df.copy()
    
    # 1. Estratificación genética
    genetic_stratifier = GeneticStratifier()
    genetic_stratifier.fit(df, genetic_features, target_col)
    result_df = genetic_stratifier.stratify(result_df)
    
    # 2. Estratificación jerárquica
    hierarchical_stratifier = HierarchicalStratifier()
    result_df = hierarchical_stratifier.create_substrata(result_df)
    
    # 3. Estratificación por biomarcadores
    if biomarker_features:
        biomarker_stratifier = BiomarkerStratifier(biomarker_features)
        biomarker_stratifier.fit(df[df[biomarker_features].notna().all(axis=1)])
        result_df = biomarker_stratifier.stratify(result_df)
    
    # 4. Estratificación combinada final
    result_df['comprehensive_stratum'] = 'Standard_Risk'
    
    # Reglas de combinación
    high_risk_conditions = [
        (result_df['genetic_risk_stratum'] == 'High_Genetic_Risk') & 
        (result_df['hierarchical_stratum'] == 'Critical_High_Risk_APOE4'),
        
        (result_df.get('biomarker_stratum') == 'High_Biomarker_Risk') & 
        (result_df['risk_category'] == 'High')
    ]
    
    for condition in high_risk_conditions:
        result_df.loc[condition, 'comprehensive_stratum'] = 'Critical_Multi_Risk'
    
    return result_df


def log_stratification_results(stratification_results: pd.DataFrame, 
                             experiment_name: str = "risk_stratification") -> Dict[str, Any]:
    """
    Registra resultados de estratificación en MLflow
    
    Args:
        stratification_results: DataFrame con resultados
        experiment_name: Nombre del experimento
        
    Returns:
        Dict con métricas calculadas
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        metrics = {}
        
        # Métricas generales
        metrics['total_samples'] = len(stratification_results)
        metrics['stratification_completeness'] = stratification_results.notna().mean().mean()
        
        # Análisis por estratos
        stratum_columns = [col for col in stratification_results.columns if 'stratum' in col]
        
        for col in stratum_columns:
            if col in stratification_results.columns:
                unique_strata = stratification_results[col].nunique()
                metrics[f'{col}_unique_strata'] = unique_strata
                
                # Distribución de estratos
                stratum_dist = stratification_results[col].value_counts(normalize=True)
                for stratum, proportion in stratum_dist.items():
                    metrics[f'{col}_{stratum}_proportion'] = proportion
        
        # Registrar métricas
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        mlflow.set_tag("stratification_type", "comprehensive")
        mlflow.set_tag("phase", "development")
        
    return metrics