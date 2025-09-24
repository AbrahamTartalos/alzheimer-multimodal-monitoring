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
Classification Pipeline for Alzheimer Risk Categorization
=========================================================

Pipeline para modelos de clasificación de categorías de riesgo de Alzheimer.
Incluye preprocesamiento, entrenamiento y evaluación de modelos.
"""

# 09c_classification_models.py (Refactor v2)
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

# ----------------------------------------
# 1. Configuración y carga de parámetros
def load_params(path: str = 'params.yaml') -> dict:
    """Carga hiperparámetros desde YAML."""
    with open(path) as f:
        return yaml.safe_load(f)

# ----------------------------------------
# 2. Carga y pre-procesado de datos
def load_data(path: str) -> pd.DataFrame:
    """Carga CSV y muestra distribución de la etiqueta."""
    df = pd.read_csv(path)
    print(f"📊 Dataset cargado: {df.shape}")
    print(df['risk_category'].value_counts(normalize=False))
    print(df['risk_category'].value_counts(normalize=True) * 100)
    return df

# ----------------------------------------
# 3. Eliminar data leakage
def purge_leakage(df: pd.DataFrame, leakage_cols: list) -> pd.DataFrame:
    """Elimina columnas con data leakage."""
    df = df.drop(columns=leakage_cols, errors='ignore')
    print(f"✅ Features eliminadas: {leakage_cols}")
    return df

# ----------------------------------------
# 4. Imputación avanzada
def impute_missing(df: pd.DataFrame, cols: list, imputer_path: str) -> pd.DataFrame:
    """Imputa valores faltantes con IterativeImputer y guarda el imputador."""
    imputer = IterativeImputer(max_iter=20, random_state=42)
    df[cols] = imputer.fit_transform(df[cols])
    joblib.dump(imputer, imputer_path)
    print(f"✅ Imputador guardado en {imputer_path}")
    return df

# ----------------------------------------
# 5. Selección de features automática
def select_features(X: pd.DataFrame, y: pd.Series, params: dict) -> (pd.DataFrame, list):
    """Reduce dimensionalidad a top-N features con GBM."""
    selector = SelectFromModel(
        GradientBoostingClassifier(random_state=42, **params['gbm']),
        max_features=params['feature_selection']['max_features'],
        threshold=-np.inf
    )
    selector.fit(X, y)
    support = selector.get_support()
    selected_cols = X.columns[support].tolist()
    print(f"✅ Selected {len(selected_cols)} features")
    return X[selected_cols], selected_cols

# ----------------------------------------
# 6. Entrenamiento de modelos base

def run_models(X_train, y_train, params: dict) -> dict:
    """Entrena modelos y retorna diccionario con instancias."""
    models = {
        'logistic': LogisticRegression(max_iter=1000, **params['logistic']),
        'random_forest': RandomForestClassifier(random_state=42, **params['random_forest']),
        'gbm': HistGradientBoostingClassifier(random_state=42, **params['histgb']),
        'svm': SVC(probability=True, **params['svm'])
    }
    for name, m in models.items():
        print(f"Entrenando {name}...")
        if name == 'svm':
            X_sub, y_sub = X_train.sample(n=10000, random_state=42), y_train.sample(n=10000, random_state=42)
            m.fit(X_sub, y_sub)
        else:
            m.fit(X_train, y_train)
    return models

# ----------------------------------------
# 7. Evaluation y ensambles

def evaluate_ensembles(models: dict, X, y) -> (object, dict):
    """Crea Voting y Stacking, evalúa y retorna modelo final."""
    voting = VotingClassifier(
        estimators=[(k, v) for k, v in models.items()], voting='soft'
    )
    stacking = StackingClassifier(
        estimators=[(k, v) for k, v in models.items()], final_estimator=LogisticRegression(), cv=3
    )
    voting.fit(X, y)
    stacking.fit(X, y)
    print("Voting F1:", np.mean(cross_val_score(voting, X, y, cv=5, scoring='f1_weighted')))
    print("Stacking F1:", np.mean(cross_val_score(stacking, X, y, cv=5, scoring='f1_weighted')))
    final_model = stacking
    return final_model

# ----------------------------------------
# 8. Interpretabilidad global

def plot_feature_importances(model, feature_names, n_top=20, save_path='../reports/figures/feature_importances.png'):
    """Grafica y guarda las importancias globales del GBM."""
    importances = model.feature_importances_
    idxs = np.argsort(importances)[-n_top:][::-1]
    plt.figure()
    plt.barh([feature_names[i] for i in idxs], importances[idxs])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Global feature importances saved to {save_path}")

# ----------------------------------------
# 9. Permutation importance local ligera

def permutation_importance_ultralight(model, X, y, features, save_path):
    r = permutation_importance(
        estimator=model,
        X=X[features].sample(20, random_state=42),
        y=y.loc[X.sample(20, random_state=42).index],
        n_repeats=3,
        random_state=42,
        n_jobs=1,
        max_samples=1.0
    )
    perm_df = pd.DataFrame({
        'feature': features,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)
    print(perm_df)
    plt.figure()
    plt.barh(perm_df['feature'], perm_df['importance_mean'], xerr=perm_df['importance_std'])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Permutation importance ultra-light saved to {save_path}")

# ----------------------------------------
# 10. Main

def main():
    params = load_params()
    df = load_data('../data/processed/features/dataset.csv')
    df = purge_leakage(df, params['leakage_cols'])
    df = impute_missing(df, params['impute_cols'], '../models/imputer.pkl')

    # Split temporal
    X = df.drop(columns=['risk_category'])
    y = df['risk_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Feature selection
    X_train_sel, selected_feats = select_features(X_train, y_train, params)
    X_test_sel = X_test[selected_feats]

    # Run models & ensembles
    models = run_models(X_train_sel, y_train, params)
    final_model = evaluate_ensembles(models, X_train_sel, y_train)

    # Evaluation temporal
    print(classification_report(y_test, final_model.predict(X_test_sel)))

    # Interpretability
    plot_feature_importances(final_model, selected_feats)
    permutation_importance_ultralight(final_model, X_test, y_test, params['clinical_features'], '../reports/figures/permutation_ultralight.png')

    # Save model and selected features
    joblib.dump(final_model, '../models/final_model.pkl')
    pd.Series(selected_feats).to_csv('../models/selected_features.csv', index=False)
    print("✅ Model and features saved.")

if __name__ == '__main__':
    main()
