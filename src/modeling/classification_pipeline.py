"""
Classification Pipeline for Alzheimer Risk Categorization
=========================================================

Pipeline para modelos de clasificación de categorías de riesgo de Alzheimer.
Incluye preprocesamiento, entrenamiento y evaluación de modelos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AlzheimerClassificationPipeline:
    """Pipeline completo para clasificación de riesgo de Alzheimer"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'risk_category', 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara los datos para clasificación
        
        Args:
            df: DataFrame con features y target
            target_col: Nombre de la columna objetivo
            test_size: Proporción para conjunto de prueba
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remover filas con target nulo
        df_clean = df.dropna(subset=[target_col])
        
        # Separar features y target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Codificar target si es categórico
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Remover columnas con demasiados valores nulos
        null_threshold = 0.5
        X = X.loc[:, X.isnull().mean() < null_threshold]
        
        # Rellenar valores nulos con mediana
        X = X.fillna(X.median())
        
        # División train/test estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_models(self) -> Dict:
        """
        Crea diccionario con modelos de clasificación
        
        Returns:
            Diccionario con modelos configurados
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=6
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        }
        
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Entrena múltiples modelos de clasificación
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Diccionario con modelos entrenados
        """
        # Estandarizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Crear y entrenar modelos
        models = self.create_models()
        trained_models = {}
        
        for name, model in models.items():
            print(f"Entrenando {name}...")
            
            # Crear pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Entrenar modelo
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline
            
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa modelos entrenados
        
        Args:
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        results = {}
        
        for name, model in self.models.items():
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Métricas
            metrics = {
                'accuracy': model.score(X_test, y_test),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'f1_macro': f1_score(y_test, y_pred, average='macro')
            }
            
            # AUC para clasificación multiclase
            if len(np.unique(y_test)) > 2:
                try:
                    metrics['auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    metrics['auc_ovr'] = 0.0
            else:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            results[name] = metrics
            
        return results
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Validación cruzada de modelos
        
        Args:
            X: Features completas
            y: Target completo
            cv: Número de folds
            
        Returns:
            Diccionario con métricas de validación cruzada
        """
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"Validación cruzada para {name}...")
            
            # Validación cruzada estratificada
            cv_scores = cross_val_score(
                model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring='f1_weighted'
            )
            
            cv_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
        return cv_results
    
    def select_best_model(self, results: Dict, metric: str = 'f1_weighted') -> str:
        """
        Selecciona el mejor modelo basado en métrica
        
        Args:
            results: Resultados de evaluación
            metric: Métrica para selección
            
        Returns:
            Nombre del mejor modelo
        """
        best_score = 0
        best_model_name = None
        
        for name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        return best_model_name
    
    def log_to_mlflow(self, results: Dict, cv_results: Dict, best_model_name: str):
        """
        Registra resultados en MLflow
        
        Args:
            results: Métricas de evaluación
            cv_results: Resultados de validación cruzada
            best_model_name: Nombre del mejor modelo
        """
        with mlflow.start_run():
            mlflow.set_tag("pipeline_type", "classification")
            mlflow.set_tag("best_model", best_model_name)
            
            # Registrar métricas de todos los modelos
            for name, metrics in results.items():
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{name}_{metric}", value)
            
            # Registrar métricas de validación cruzada
            for name, cv_metrics in cv_results.items():
                mlflow.log_metric(f"{name}_cv_mean", cv_metrics['cv_mean'])
                mlflow.log_metric(f"{name}_cv_std", cv_metrics['cv_std'])
            
            # Registrar mejor modelo
            mlflow.sklearn.log_model(self.best_model, "best_classification_model")
    
    def run_pipeline(self, df: pd.DataFrame, target_col: str = 'risk_category') -> Dict:
        """
        Ejecuta pipeline completo de clasificación
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo
            
        Returns:
            Diccionario con resultados completos
        """
        print("🚀 Iniciando pipeline de clasificación...")
        
        # Preparar datos
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        print(f"📊 Datos preparados: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Entrenar modelos
        trained_models = self.train_models(X_train, y_train)
        print(f"✅ {len(trained_models)} modelos entrenados")
        
        # Evaluar modelos
        results = self.evaluate_models(X_test, y_test)
        print("📈 Evaluación completada")
        
        # Validación cruzada
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        cv_results = self.cross_validate_models(X_full, y_full)
        print("🔄 Validación cruzada completada")
        
        # Seleccionar mejor modelo
        best_model_name = self.select_best_model(results)
        print(f"🏆 Mejor modelo: {best_model_name}")
        
        # Registrar en MLflow
        self.log_to_mlflow(results, cv_results, best_model_name)
        print("📁 Resultados registrados en MLflow")
        
        return {
            'results': results,
            'cv_results': cv_results,
            'best_model': best_model_name,
            'trained_models': trained_models
        }

def run_classification_analysis(df: pd.DataFrame, target_col: str = 'risk_category') -> Dict:
    """
    Función principal para análisis de clasificación
    
    Args:
        df: DataFrame con datos
        target_col: Columna objetivo
        
    Returns:
        Resultados del análisis
    """
    pipeline = AlzheimerClassificationPipeline()
    return pipeline.run_pipeline(df, target_col)

# Configuraciones específicas para diferentes tipos de clasificación
CLASSIFICATION_CONFIGS = {
    'risk_category': {
        'target': 'risk_category',
        'metric': 'f1_weighted',
        'models': ['logistic_regression', 'random_forest', 'gradient_boosting']
    },
    'cognitive_status': {
        'target': 'DIAGNOSIS',
        'metric': 'f1_macro',
        'models': ['svm', 'random_forest', 'gradient_boosting']
    }
}

