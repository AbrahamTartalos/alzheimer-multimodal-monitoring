"""
Ensemble Methods for Alzheimer Risk Prediction
==============================================

Este módulo implementa métodos ensemble para combinar múltiples modelos
de predicción de riesgo de Alzheimer, integrando regresión, clasificación
y estratificación de riesgo.

Autor: Abraham Tartalos
Fecha: 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor, 
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

class AlzheimerEnsemble:
    """
    Clase principal para métodos ensemble en predicción de Alzheimer
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regression_ensemble = None
        self.classification_ensemble = None
        self.meta_ensemble = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_base_models(self) -> Dict[str, Any]:
        """Crear modelos base para ensemble"""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVR, SVC
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        
        base_models = {
            'regression': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'svr': SVR(kernel='rbf', C=1.0),
                'mlp': MLPRegressor(hidden_layer_sizes=(100,), random_state=self.random_state, max_iter=500)
            },
            'classification': {
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'svc': SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state),
                'mlp': MLPClassifier(hidden_layer_sizes=(100,), random_state=self.random_state, max_iter=500)
            }
        }
        return base_models
    
    def create_voting_ensemble(self, task: str = 'regression') -> Any:
        """Crear ensemble de voting"""
        base_models = self.create_base_models()
        
        if task == 'regression':
            estimators = [(name, model) for name, model in base_models['regression'].items()]
            ensemble = VotingRegressor(estimators=estimators)
        else:
            estimators = [(name, model) for name, model in base_models['classification'].items()]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
        return ensemble
    
    def create_boosting_ensemble(self, task: str = 'regression') -> Any:
        """Crear ensemble de boosting"""
        if task == 'regression':
            ensemble = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        else:
            ensemble = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        return ensemble
    
    def create_bagging_ensemble(self, task: str = 'regression') -> Any:
        """Crear ensemble de bagging"""
        if task == 'regression':
            from sklearn.tree import DecisionTreeRegressor
            base_estimator = DecisionTreeRegressor(random_state=self.random_state)
            ensemble = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=self.random_state
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)
            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=self.random_state
            )
        return ensemble

class StackingEnsemble:
    """
    Implementación de Stacking Ensemble para Alzheimer
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.level_0_models = {}
        self.level_1_model = None
        self.is_fitted = False
        
    def create_level_0_models(self, task: str = 'regression') -> Dict[str, Any]:
        """Crear modelos de nivel 0 (base learners)"""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVR, SVC
        
        if task == 'regression':
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=self.random_state),
                'ridge': Ridge(alpha=1.0),
                'svr': SVR(kernel='rbf', C=1.0)
            }
        else:
            models = {
                'rf': RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'svc': SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state)
            }
        return models
    
    def create_level_1_model(self, task: str = 'regression') -> Any:
        """Crear modelo de nivel 1 (meta-learner)"""
        if task == 'regression':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=0.1)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
    
    def fit_stacking(self, X: np.ndarray, y: np.ndarray, task: str = 'regression') -> 'StackingEnsemble':
        """Entrenar ensemble de stacking"""
        from sklearn.model_selection import cross_val_predict
        
        # Crear modelos
        self.level_0_models = self.create_level_0_models(task)
        self.level_1_model = self.create_level_1_model(task)
        
        # Entrenar modelos de nivel 0 y generar predicciones
        level_1_features = np.zeros((X.shape[0], len(self.level_0_models)))
        
        for i, (name, model) in enumerate(self.level_0_models.items()):
            # Cross-validation predictions para evitar overfitting
            cv_preds = cross_val_predict(model, X, y, cv=5, method='predict')
            level_1_features[:, i] = cv_preds
            
            # Entrenar modelo completo
            model.fit(X, y)
        
        # Entrenar meta-learner
        self.level_1_model.fit(level_1_features, y)
        self.is_fitted = True
        
        return self
    
    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Realizar predicciones con stacking"""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Predicciones de nivel 0
        level_1_features = np.zeros((X.shape[0], len(self.level_0_models)))
        
        for i, (name, model) in enumerate(self.level_0_models.items()):
            level_1_features[:, i] = model.predict(X)
        
        # Predicción final del meta-learner
        return self.level_1_model.predict(level_1_features)

class MultimodalEnsemble:
    """
    Ensemble específico para datos multimodales de Alzheimer
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.modality_models = {}
        self.fusion_model = None
        self.feature_groups = {
            'genetic': [],
            'biomarkers': [],
            'cognitive': [],
            'activity': [],
            'demographics': []
        }
        
    def identify_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Identificar grupos de features por modalidad"""
        groups = {
            'genetic': [],
            'biomarkers': [],
            'cognitive': [],
            'activity': [],
            'demographics': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            if any(x in feature_lower for x in ['apoe', 'genetic']):
                groups['genetic'].append(feature)
            elif any(x in feature_lower for x in ['tau', 'abeta', 'biomarker']):
                groups['biomarkers'].append(feature)
            elif any(x in feature_lower for x in ['cdrsb', 'diagnosis', 'cognitive']):
                groups['cognitive'].append(feature)
            elif any(x in feature_lower for x in ['sleep', 'activity', 'steps']):
                groups['activity'].append(feature)
            elif any(x in feature_lower for x in ['age', 'gender', 'demo']):
                groups['demographics'].append(feature)
            else:
                # Asignar a demographics por defecto
                groups['demographics'].append(feature)
        
        return groups
    
    def train_modality_experts(self, X: pd.DataFrame, y: np.ndarray, task: str = 'regression'):
        """Entrenar modelos expertos por modalidad"""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        self.feature_groups = self.identify_feature_groups(X.columns.tolist())
        
        for modality, features in self.feature_groups.items():
            if len(features) > 0:
                X_modality = X[features]
                
                if task == 'regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                
                model.fit(X_modality, y)
                self.modality_models[modality] = model
    
    def create_fusion_features(self, X: pd.DataFrame) -> np.ndarray:
        """Crear features de fusión a partir de predicciones de modalidades"""
        fusion_features = []
        
        for modality, model in self.modality_models.items():
            features = self.feature_groups[modality]
            if len(features) > 0:
                X_modality = X[features]
                pred = model.predict(X_modality)
                fusion_features.append(pred)
        
        return np.column_stack(fusion_features) if fusion_features else np.array([])

def evaluate_ensemble_performance(models: Dict[str, Any], X_test: np.ndarray, 
                                y_test: np.ndarray, task: str = 'regression') -> Dict[str, float]:
    """Evaluar rendimiento de múltiples modelos ensemble"""
    results = {}
    
    for name, model in models.items():
        try:
            predictions = model.predict(X_test)
            
            if task == 'regression':
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                results[f'{name}_mse'] = mse
                results[f'{name}_rmse'] = rmse
            else:
                accuracy = accuracy_score(y_test, predictions)
                results[f'{name}_accuracy'] = accuracy
                
        except Exception as e:
            print(f"Error evaluando {name}: {e}")
            results[f'{name}_error'] = str(e)
    
    return results

def create_ensemble_pipeline(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray,
                           task: str = 'regression') -> Dict[str, Any]:
    """Crear y evaluar pipeline completo de ensemble"""
    
    # Inicializar ensemble principal
    alzheimer_ensemble = AlzheimerEnsemble()
    
    # Crear diferentes tipos de ensemble
    ensembles = {
        'voting': alzheimer_ensemble.create_voting_ensemble(task),
        'boosting': alzheimer_ensemble.create_boosting_ensemble(task),
        'bagging': alzheimer_ensemble.create_bagging_ensemble(task)
    }
    
    # Entrenar modelos
    trained_models = {}
    for name, model in ensembles.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Error entrenando {name}: {e}")
    
    # Evaluar rendimiento
    performance = evaluate_ensemble_performance(trained_models, X_test, y_test, task)
    
    return {
        'models': trained_models,
        'performance': performance,
        'best_model': min(performance.items(), key=lambda x: x[1] if 'mse' in x[0] else -x[1])
    }

def log_ensemble_results(results: Dict[str, Any], experiment_name: str = "ensemble_methods"):
    """Registrar resultados de ensemble en MLflow"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.set_tag("model_type", "ensemble")
        mlflow.set_tag("approach", "multiple_ensemble_comparison")
        
        # Registrar métricas de rendimiento
        for metric_name, value in results['performance'].items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
        
        # Registrar mejor modelo
        if results['best_model']:
            best_name, best_score = results['best_model']
            mlflow.log_metric("best_score", best_score)
            mlflow.set_tag("best_model", best_name.split('_')[0])
        
        # Guardar modelos
        for name, model in results['models'].items():
            try:
                mlflow.sklearn.log_model(model, f"ensemble_{name}")
            except Exception as e:
                print(f"Error guardando modelo {name}: {e}")

def save_ensemble_models(models: Dict[str, Any], filepath: str = "../models/ensemble/"):
    """Guardar modelos ensemble entrenados"""
    import os
    
    os.makedirs(filepath, exist_ok=True)
    
    for name, model in models.items():
        try:
            model_path = os.path.join(filepath, f"{name}_ensemble.pkl")
            joblib.dump(model, model_path)
            print(f"✅ Modelo {name} guardado en: {model_path}")
        except Exception as e:
            print(f"❌ Error guardando {name}: {e}")

def load_ensemble_models(filepath: str = "../models/ensemble/") -> Dict[str, Any]:
    """Cargar modelos ensemble guardados"""
    import os
    
    models = {}
    
    if os.path.exists(filepath):
        for filename in os.listdir(filepath):
            if filename.endswith('_ensemble.pkl'):
                model_name = filename.replace('_ensemble.pkl', '')
                model_path = os.path.join(filepath, filename)
                try:
                    models[model_name] = joblib.load(model_path)
                    print(f"✅ Modelo {model_name} cargado")
                except Exception as e:
                    print(f"❌ Error cargando {model_name}: {e}")
    
    return models

# Funciones de utilidad adicionales
def ensemble_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extraer importancia de features de modelos ensemble"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("El modelo no tiene atributo feature_importances_")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error extrayendo importancia: {e}")
        return pd.DataFrame()

def ensemble_cross_validation(model: Any, X: np.ndarray, y: np.ndarray, 
                            cv: int = 5, task: str = 'regression') -> Dict[str, float]:
    """Realizar validación cruzada para modelos ensemble"""
    try:
        if task == 'regression':
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            return {
                'cv_mse_mean': -scores.mean(),
                'cv_mse_std': scores.std(),
                'cv_rmse_mean': np.sqrt(-scores.mean())
            }
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return {
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std()
            }
    except Exception as e:
        print(f"Error en validación cruzada: {e}")
        return {}

