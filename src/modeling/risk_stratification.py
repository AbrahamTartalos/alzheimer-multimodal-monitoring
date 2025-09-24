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
Risk Stratification Pipeline for Alzheimer's Disease
==================================================

Pipeline unificado para estratificación de riesgo compatible con 09e_risk_stratification.py

Autor: Abraham Tartalos
Fecha: 2025
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class RiskStratificationPipeline:
    """
    Pipeline principal para estratificación de riesgo
    """
    
    def __init__(self, target_column='composite_risk_score', 
                 category_column='risk_category', random_state=42):
        self.target_column = target_column
        self.category_column = category_column
        self.random_state = random_state
        self.results = {}
        self.models = {}
        
    def identify_phenotypes(self, X, method='kmeans', k_range=(2, 8)):
        """
        Identifica fenotipos usando clustering
        
        Args:
            X: Features procesadas
            method: Método de clustering ('kmeans', 'gmm')
            k_range: Rango de clusters a evaluar
            
        Returns:
            Dict con resultados del clustering
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimizar número de clusters
        silhouette_scores = []
        models = []
        
        for k in range(k_range[0], k_range[1] + 1):
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            elif method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=self.random_state)
            
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            models.append(model)
        
        # Seleccionar mejor modelo
        best_idx = np.argmax(silhouette_scores)
        best_model = models[best_idx]
        optimal_k = range(k_range[0], k_range[1] + 1)[best_idx]
        
        results = {
            'model': best_model,
            'n_clusters': optimal_k,
            'labels': best_model.fit_predict(X_scaled),
            'scaler': scaler,
            'params': {
                'method': method,
                'n_clusters': optimal_k,
                'random_state': self.random_state
            },
            'metrics': {
                'silhouette_score': silhouette_scores[best_idx],
                'optimal_k': optimal_k
            }
        }
        
        self.results['phenotypes'] = results
        self.models['phenotype_model'] = best_model
        return results
    
    def genetic_stratification(self, df, genetic_markers=['APOE_e4_carrier']):
        """
        Estratificación genética basada en APOE
        
        Args:
            df: DataFrame con datos
            genetic_markers: Lista de marcadores genéticos
            
        Returns:
            Dict con resultados de estratificación genética
        """
        results = {}
        
        if 'APOE_e4_carrier' in genetic_markers and 'APOE_e4_carrier' in df.columns:
            # Análisis APOE
            apoe_analysis = df.groupby('APOE_e4_carrier')[self.target_column].agg(['mean', 'std', 'count'])
            
            carrier_risk = apoe_analysis.loc[1, 'mean']
            non_carrier_risk = apoe_analysis.loc[0, 'mean']
            risk_difference = carrier_risk - non_carrier_risk
            
            results = {
                'apoe_carrier_stats': {
                    'carrier_mean_risk': carrier_risk,
                    'non_carrier_mean_risk': non_carrier_risk,
                    'count': apoe_analysis.loc[1, 'count']
                },
                'metrics': {
                    'risk_difference': risk_difference,
                    'risk_ratio': carrier_risk / non_carrier_risk if non_carrier_risk > 0 else 0
                }
            }
        
        self.results['genetic'] = results
        return results
    
    def probabilistic_stratification(self, X, method='gmm', n_components_range=(2, 6)):
        """
        Estratificación probabilística usando GMM
        
        Args:
            X: Features procesadas
            method: Método ('gmm')
            n_components_range: Rango de componentes a evaluar
            
        Returns:
            Dict con resultados GMM
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimizar número de componentes usando BIC
        bic_scores = []
        models = []
        
        for n in range(n_components_range[0], n_components_range[1] + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
            models.append(gmm)
        
        # Seleccionar mejor modelo (menor BIC)
        best_idx = np.argmin(bic_scores)
        best_model = models[best_idx]
        optimal_n = range(n_components_range[0], n_components_range[1] + 1)[best_idx]
        
        results = {
            'model': best_model,
            'n_components': optimal_n,
            'labels': best_model.predict(X_scaled),
            'scaler': scaler,
            'params': {
                'method': method,
                'n_components': optimal_n,
                'random_state': self.random_state
            },
            'metrics': {
                'bic_score': bic_scores[best_idx],
                'optimal_n': optimal_n
            }
        }
        
        self.results['gmm'] = results
        self.models['gmm_model'] = best_model
        return results
    
    def hierarchical_stratification(self, df, criteria):
        """
        Estratificación jerárquica basada en criterios múltiples
        
        Args:
            df: DataFrame con datos
            criteria: Dict con criterios de estratificación
            
        Returns:
            Dict con resultados jerárquicos
        """
        df_result = df.copy()
        df_result['hierarchical_stratum'] = 'Standard'
        
        substrata_count = 0
        
        for stratum_name, config in criteria.items():
            conditions = config['conditions']
            operator = config.get('operator', 'and')
            
            # Construir máscara de condiciones
            if operator == 'and':
                mask = pd.Series(True, index=df.index)
                for condition in conditions:
                    mask = mask & df.eval(condition)
            else:  # or
                mask = pd.Series(False, index=df.index)
                for condition in conditions:
                    mask = mask | df.eval(condition)
            
            df_result.loc[mask, 'hierarchical_stratum'] = stratum_name
            substrata_count += mask.sum()
        
        results = {
            'n_substrata': len(criteria),
            'substrata_assigned': substrata_count,
            'stratified_df': df_result,
            'metrics': {
                'coverage': substrata_count / len(df),
                'unique_strata': df_result['hierarchical_stratum'].nunique()
            }
        }
        
        self.results['hierarchical'] = results
        return results
    
    def biomarker_stratification(self, df, biomarker_columns, n_clusters=3):
        """
        Estratificación basada en biomarcadores
        
        Args:
            df: DataFrame con datos
            biomarker_columns: Lista de columnas de biomarcadores
            n_clusters: Número de clusters
            
        Returns:
            Dict con resultados de biomarcadores
        """
        # Filtrar datos con biomarcadores válidos
        biomarker_data = df[biomarker_columns].dropna()
        
        if len(biomarker_data) == 0:
            return {'error': 'No hay datos válidos de biomarcadores'}
        
        # Clustering de biomarcadores
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(biomarker_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        results = {
            'model': kmeans,
            'n_clusters': n_clusters,
            'labels': labels,
            'scaler': scaler,
            'valid_indices': biomarker_data.index,
            'params': {
                'n_clusters': n_clusters,
                'biomarker_features': biomarker_columns
            },
            'metrics': {
                'silhouette_score': silhouette_score(X_scaled, labels),
                'valid_samples': len(biomarker_data)
            }
        }
        
        self.results['biomarker'] = results
        self.models['biomarker_model'] = kmeans
        return results
    
    def plot_risk_distribution_by_strata(self, ax=None):
        """
        Grafica distribución de riesgo por estratos
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Usar resultados disponibles
        if 'phenotypes' in self.results:
            ax.hist(self.results['phenotypes']['labels'], alpha=0.7, label='Phenotypes')
        
        ax.set_xlabel('Stratum')
        ax.set_ylabel('Count')
        ax.set_title('Risk Distribution by Strata')
        ax.legend()
        
        return ax
    
    def compute_stratification_quality(self):
        """
        Calcula métricas de calidad de estratificación
        
        Returns:
            Dict con métricas de calidad
        """
        quality_metrics = {}
        
        # Métricas de clustering
        if 'phenotypes' in self.results:
            quality_metrics['phenotype_silhouette'] = self.results['phenotypes']['metrics']['silhouette_score']
        
        if 'gmm' in self.results:
            quality_metrics['gmm_bic'] = self.results['gmm']['metrics']['bic_score']
        
        # Métricas de cobertura
        if 'hierarchical' in self.results:
            quality_metrics['hierarchical_coverage'] = self.results['hierarchical']['metrics']['coverage']
        
        return quality_metrics
    
    def generate_summary_report(self):
        """
        Genera reporte resumen de estratificación
        
        Returns:
            Dict con estadísticas resumen
        """
        summary = {}
        
        if 'phenotypes' in self.results:
            summary['phenotypes'] = {
                'n_clusters': self.results['phenotypes']['n_clusters'],
                'silhouette_score': f"{self.results['phenotypes']['metrics']['silhouette_score']:.3f}"
            }
        
        if 'genetic' in self.results:
            summary['genetic'] = {
                'risk_difference': f"{self.results['genetic']['metrics']['risk_difference']:.3f}",
                'carriers_analyzed': self.results['genetic']['apoe_carrier_stats']['count']
            }
        
        if 'gmm' in self.results:
            summary['gmm'] = {
                'n_components': self.results['gmm']['n_components'],
                'bic_score': f"{self.results['gmm']['metrics']['bic_score']:.1f}"
            }
        
        return summary
    
    def consolidate_results(self):
        """
        Consolida todos los resultados en un DataFrame
        
        Returns:
            DataFrame con resultados consolidados
        """
        consolidated_data = []
        
        for method, result in self.results.items():
            if 'metrics' in result:
                for metric_name, metric_value in result['metrics'].items():
                    consolidated_data.append({
                        'method': method,
                        'metric': metric_name,
                        'value': metric_value
                    })
        
        return pd.DataFrame(consolidated_data)