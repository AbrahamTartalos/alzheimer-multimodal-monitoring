"""
Pipeline de Regresión para Score de Riesgo Compuesto
===================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
import mlflow
import mlflow.sklearn
from .model_utils import evaluate_model, log_model_metrics, save_model_artifacts
import warnings
warnings.filterwarnings('ignore')

class RegressionPipeline:
    """Pipeline para modelos de regresión del score de riesgo"""
    
    def __init__(self, target_col='composite_risk_score'):
        self.target_col = target_col
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def setup_models(self):
        """Configura modelos de regresión"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            'svr': SVR()
        }
        
        print(f"🔧 Configurados {len(self.models)} modelos de regresión")
    
    def get_hyperparameter_grids(self):
        """Define grids de hiperparámetros para optimización"""
        return {
            'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svr': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def train_single_model(self, model_name, X_train, y_train, optimize=False):
        """
        Entrena un modelo individual
        
        Args:
            model_name: Nombre del modelo
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            optimize: Si optimizar hiperparámetros
        
        Returns:
            Modelo entrenado
        """
        model = self.models[model_name].copy() if hasattr(self.models[model_name], 'copy') else self.models[model_name]
        
        # Escalar datos si es necesario
        if model_name in ['svr', 'ridge', 'lasso', 'elastic_net']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            self.scalers[model_name] = None
        
        # Optimización de hiperparámetros
        if optimize and model_name in self.get_hyperparameter_grids():
            param_grid = self.get_hyperparameter_grids()[model_name]
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            print(f"🎯 {model_name} - Mejores parámetros: {grid_search.best_params_}")
        else:
            model.fit(X_train_scaled, y_train)
        
        return model
    
    def evaluate_single_model(self, model_name, model, X_test, y_test):
        """Evalúa un modelo individual"""
        # Aplicar scaling si es necesario
        if self.scalers[model_name] is not None:
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Evaluar modelo
        metrics = evaluate_model(model, X_test_scaled, y_test, 'regression')
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """Validación cruzada para un modelo"""
        model = self.models[model_name]
        
        # Escalar si es necesario
        if model_name in ['svr', 'ridge', 'lasso', 'elastic_net']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Validación cruzada
        cv_scores = cross_val_score(
            model, X_scaled, y, cv=cv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        rmse_scores = np.sqrt(-cv_scores)
        
        return {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'cv_scores': rmse_scores
        }
    
    def run_regression_pipeline(self, X_train, X_test, y_train, y_test, 
                               optimize_hyperparams=False, cross_validate=True):
        """
        Ejecuta pipeline completo de regresión
        
        Args:
            X_train, X_test, y_train, y_test: Datos de entrenamiento y test
            optimize_hyperparams: Si optimizar hiperparámetros
            cross_validate: Si realizar validación cruzada
        
        Returns:
            dict: Resultados de todos los modelos
        """
        if not self.models:
            self.setup_models()
        
        print("\n🚀 INICIANDO PIPELINE DE REGRESIÓN")
        print("=" * 50)
        
        trained_models = {}
        
        for model_name in self.models.keys():
            print(f"\n🔄 Entrenando {model_name}...")
            
            with mlflow.start_run(nested=True, run_name=f"regression_{model_name}"):
                mlflow.set_tag("model_type", "regression")
                mlflow.set_tag("target", self.target_col)
                
                # Entrenar modelo
                trained_model = self.train_single_model(
                    model_name, X_train, y_train, optimize_hyperparams
                )
                trained_models[model_name] = trained_model
                
                # Evaluar en test set
                test_metrics = self.evaluate_single_model(
                    model_name, trained_model, X_test, y_test
                )
                
                # Validación cruzada
                if cross_validate:
                    cv_results = self.cross_validate_model(
                        model_name, X_train, y_train
                    )
                    test_metrics.update(cv_results)
                    print(f"📊 CV RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
                
                # Registrar en MLflow
                log_model_metrics(test_metrics, model_name, "regression")
                mlflow.sklearn.log_model(trained_model, f"model_{model_name}")
                
                # Guardar modelo
                save_model_artifacts(
                    trained_model, f"regression_{model_name}",
                    self.scalers[model_name], X_train.columns.tolist()
                )
        
        self.trained_models = trained_models
        return self.results
    
    def get_best_model(self, metric='r2'):
        """
        Obtiene el mejor modelo según métrica especificada
        
        Args:
            metric: Métrica para comparar ('r2', 'rmse', 'mae')
        
        Returns:
            tuple: (nombre_modelo, modelo, métricas)
        """
        if not self.results:
            print("❌ No hay resultados disponibles. Ejecuta el pipeline primero.")
            return None
        
        if metric == 'rmse':
            best_model_name = min(self.results.keys(), 
                                key=lambda x: self.results[x].get('rmse', float('inf')))
        else:
            best_model_name = max(self.results.keys(),
                                key=lambda x: self.results[x].get(metric, -float('inf')))
        
        best_model = self.trained_models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        print(f"🏆 Mejor modelo ({metric}): {best_model_name}")
        print(f"📊 Métricas: {best_metrics}")
        
        return best_model_name, best_model, best_metrics
    
    def predict_risk_score(self, model_name, X_new):
        """
        Predice score de riesgo para nuevos datos
        
        Args:
            model_name: Nombre del modelo a usar
            X_new: Nuevos datos para predicción
        
        Returns:
            np.array: Predicciones
        """
        if model_name not in self.trained_models:
            print(f"❌ Modelo {model_name} no está entrenado")
            return None
        
        model = self.trained_models[model_name]
        
        # Aplicar scaling si es necesario
        if self.scalers[model_name] is not None:
            X_new_scaled = self.scalers[model_name].transform(X_new)
        else:
            X_new_scaled = X_new
        
        predictions = model.predict(X_new_scaled)
        
        # Asegurar que las predicciones estén en rango válido [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        return predictions