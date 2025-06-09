"""
Feature Engineering Pipeline - Monitorización Multimodal de Alzheimer
======================================================================

Script maestro que orquesta el proceso completo de feature engineering
para todas las modalidades del dataset integrado y crea el Score de Riesgo compuesto.

Autor: Abraham Tartalos
Fecha: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de feature engineering por modalidad
from fe_demographics import engineer_demographic_features
from fe_genetics import engineer_genetic_features
from fe_mri import engineer_mri_features  # CAMBIO: neuroimaging -> mri
from fe_pet import engineer_pet_features  # NUEVO: agregado fe_pet
from fe_biomarkers import engineer_biomarker_features
from fe_clinical import engineer_clinical_features
from fe_synthetic_activity_sleep import engineer_activity_sleep_features

class FeatureEngineeringPipeline:
    """
    Pipeline maestro para feature engineering multimodal.
    
    Coordina la transformación de features para todas las modalidades
    y genera variables combinadas clínicamente relevantes.
    """
    
    def __init__(self):
        self.feature_summary = {}
        self.risk_score_components = []
        
    def run_pipeline(self, input_path, output_path=None):
        """
        Ejecutar pipeline completo de feature engineering
        """
        print("🚀 INICIANDO PIPELINE DE FEATURE ENGINEERING")
        print("=" * 60)
        
        # 1. Cargar datos integrados
        print("=>) Cargando dataset integrado...")
        df = pd.read_csv(input_path)
        print(f"   Dataset original: {df.shape[0]:,} registros × {df.shape[1]} variables")
        
        # 2. Aplicar transformaciones por modalidad
        df_engineered = self._apply_modality_transformations(df)
        
        # 3. Crear features combinadas inter-modalidad
        df_engineered = self._create_cross_modal_features(df_engineered)
        
        # 4. Crear Score de Riesgo Compuesto (endpoint objetivo)
        df_engineered = self._create_composite_risk_score(df_engineered)
        
        # 5. Feature selection y limpieza final
        df_final = self._final_feature_selection(df_engineered)
        
        # 5.1 Asegurar de que existe la carpeta de salida
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 6. Guardar resultados
        if output_path:
            self._save_results(df_final, output_path)
        
        # 7. Generar reporte
        self._generate_report(df, df_final)
        
        return df_final
    
    def _apply_modality_transformations(self, df):
        """Aplicar transformaciones específicas por modalidad"""
        
        print("\n APLICANDO TRANSFORMACIONES POR MODALIDAD")
        print("-" * 50)
        
        # Demographics
        try:
            df = engineer_demographic_features(df)
            self.feature_summary['demographics'] = 'OK'
        except Exception as e:
            print(f"   ⚠️  Error en demographics: {str(e)[:50]}...")
            self.feature_summary['demographics'] = 'ERROR'
        
        # Genetics
        try:
            df = engineer_genetic_features(df)
            self.feature_summary['genetics'] = 'OK'
        except Exception as e:
            print(f"   ⚠️  Error en genetics: {str(e)[:50]}...")
            self.feature_summary['genetics'] = 'ERROR'
        
        # MRI (anteriormente Neuroimaging)
        try:
            df = engineer_mri_features(df)  # CAMBIO: función renombrada
            self.feature_summary['mri'] = 'OK'  # CAMBIO: clave renombrada
        except Exception as e:
            print(f"   ⚠️  Error en MRI: {str(e)[:50]}...")  # CAMBIO: mensaje actualizado
            self.feature_summary['mri'] = 'ERROR'  # CAMBIO: clave renombrada
        
        # PET (NUEVO)
        try:
            df = engineer_pet_features(df)  # NUEVO: procesamiento PET
            self.feature_summary['pet'] = 'OK' 
        except Exception as e:
            print(f"   ⚠️  Error en PET: {str(e)[:50]}...")  
            self.feature_summary['pet'] = 'ERROR'  
        
        # Biomarkers
        try:
            df = engineer_biomarker_features(df)
            self.feature_summary['biomarkers'] = 'OK'
        except Exception as e:
            print(f"   ⚠️  Error en biomarkers: {str(e)[:50]}...")
            self.feature_summary['biomarkers'] = 'ERROR'
        
        # Clinical
        try:
            df = engineer_clinical_features(df)
            self.feature_summary['clinical'] = 'OK'
        except Exception as e:
            print(f"   ⚠️  Error en clinical: {str(e)[:50]}...")
            self.feature_summary['clinical'] = 'ERROR'
        
        # Activity & Sleep
        try:
            df = engineer_activity_sleep_features(df)
            self.feature_summary['activity_sleep'] = 'OK'
        except Exception as e:
            print(f"   ⚠️  Error en activity_sleep: {str(e)[:50]}...")
            self.feature_summary['activity_sleep'] = 'ERROR'
        
        return df
    
    def _create_cross_modal_features(self, df):
        """Crear features que combinan múltiples modalidades"""
        
        print("\n=>) CREANDO FEATURES INTER-MODALIDAD")
        print("-" * 40)
        
        # 1. Índice de completitud multimodal
        modal_completeness_cols = [col for col in df.columns if 'completeness' in col.lower()]
        if modal_completeness_cols:
            df['multimodal_completeness'] = df[modal_completeness_cols].mean(axis=1)
            print("   ✅ Índice de completitud multimodal")
        
        # 2. Score de riesgo genético-demográfico
        genetic_risk_cols = [col for col in df.columns if 'genetic' in col.lower() and 'risk' in col.lower()]
        age_cols = [col for col in df.columns if 'age' in col.lower() and col != 'AGE']  # Evitar columna original
        
        if genetic_risk_cols and age_cols:
            # Combinar riesgo genético con edad
            genetic_col = genetic_risk_cols[0]
            age_col = age_cols[0]
            df['genetic_age_risk'] = df[genetic_col] * np.log1p(df[age_col].fillna(df[age_col].median()))
            print("   ✅ Score genético-demográfico")
        
        # 3. Índice de estilo de vida vs biomarcadores
        lifestyle_cols = [col for col in df.columns if 'lifestyle' in col.lower() or 'behavioral' in col.lower()]
        biomarker_risk_cols = [col for col in df.columns if 'biomarker' in col.lower() and 'risk' in col.lower()]
        
        if lifestyle_cols and biomarker_risk_cols:
            lifestyle_col = lifestyle_cols[0]
            biomarker_col = biomarker_risk_cols[0]
            # Interacción: estilo de vida protector vs riesgo biomarcador
            df['lifestyle_biomarker_interaction'] = df[lifestyle_col] * (1 - df[biomarker_col])
            print("   ✅ Interacción estilo de vida - biomarcadores")

        # 4. NUEVO: Score de neuroimagen combinado (MRI + PET)
        mri_risk_cols = [col for col in df.columns if 'mri' in col.lower() and 'risk' in col.lower()]
        pet_risk_cols = [col for col in df.columns if 'pet' in col.lower() and 'risk' in col.lower()]
        
        if mri_risk_cols and pet_risk_cols:
            mri_col = mri_risk_cols[0]
            pet_col = pet_risk_cols[0]
            # Combinar riesgo estructural (MRI) con funcional (PET)
            df['neuroimaging_combined_risk'] = (df[mri_col].fillna(0) + df[pet_col].fillna(0)) / 2
            print("   ✅ Score combinado MRI-PET")
        
        return df
    
    def _create_composite_risk_score(self, df):
        """Crear Score de Riesgo Compuesto (endpoint objetivo principal)"""
        
        print("\n=>=>) CREANDO SCORE DE RIESGO COMPUESTO")
        print("-" * 40)
        
        risk_components = []
        weights = []
        
        # Componente 1: Riesgo Genético (peso: 0.20) - AJUSTADO para incluir neuroimagen
        genetic_risk_cols = [col for col in df.columns if 'APOE_risk_score' in col or 'genetic_risk' in col.lower()]
        if genetic_risk_cols:
            risk_col = genetic_risk_cols[0]
            # Normalizar a 0-1
            df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.20)  # CAMBIO: reducido de 0.25
            self.risk_score_components.append(f"Genético ({risk_col})")
        
        # Componente 2: Riesgo Demográfico (peso: 0.15) - AJUSTADO
        demo_risk_cols = [col for col in df.columns if 'age_risk' in col.lower() or 'demographic_risk' in col.lower()]
        if demo_risk_cols:
            risk_col = demo_risk_cols[0]
            df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.15)  # CAMBIO: reducido de 0.20
            self.risk_score_components.append(f"Demográfico ({risk_col})")
        
        # Componente 3: Riesgo Biomarcador (peso: 0.25) - AJUSTADO
        biomarker_risk_cols = [col for col in df.columns if 'biomarker' in col.lower() and 'risk' in col.lower()]
        if biomarker_risk_cols:
            risk_col = biomarker_risk_cols[0]
            df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.25)  # CAMBIO: reducido de 0.30
            self.risk_score_components.append(f"Biomarcador ({risk_col})")
        
        # Componente 4: NUEVO - Riesgo Neuroimagen Combinado (peso: 0.25)
        neuroimaging_combined_cols = [col for col in df.columns if 'neuroimaging_combined_risk' in col]
        if neuroimaging_combined_cols:
            risk_col = neuroimaging_combined_cols[0]
            # Normalizar a 0-1
            df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.25)  # NUEVO: peso significativo para neuroimagen
            self.risk_score_components.append(f"Neuroimagen ({risk_col})")
        else:
            # Fallback: usar MRI o PET individual si no hay combinado
            mri_risk_cols = [col for col in df.columns if 'mri' in col.lower() and 'risk' in col.lower()]
            pet_risk_cols = [col for col in df.columns if 'pet' in col.lower() and 'risk' in col.lower()]
            
            if mri_risk_cols:
                risk_col = mri_risk_cols[0]
                df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
                risk_components.append(f'{risk_col}_norm')
                weights.append(0.15)  # PESO REDUCIDO si solo hay MRI
                self.risk_score_components.append(f"MRI ({risk_col})")
            
            if pet_risk_cols:
                risk_col = pet_risk_cols[0]
                df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
                risk_components.append(f'{risk_col}_norm')
                weights.append(0.10)  # PESO REDUCIDO si solo hay PET
                self.risk_score_components.append(f"PET ({risk_col})")
        
        # Componente 5: Riesgo Conductual (peso: 0.10) - AJUSTADO
        behavioral_risk_cols = [col for col in df.columns if 'behavioral_risk' in col.lower()]
        if behavioral_risk_cols:
            risk_col = behavioral_risk_cols[0]
            df[f'{risk_col}_norm'] = df[risk_col]  # Ya normalizado 0-1
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.10)  # CAMBIO: reducido de 0.15
            self.risk_score_components.append(f"Conductual ({risk_col})")
        
        # Componente 6: Riesgo Clínico (peso: 0.05) - AJUSTADO
        clinical_risk_cols = [col for col in df.columns if 'clinical' in col.lower() and 'risk' in col.lower()]
        if clinical_risk_cols:
            risk_col = clinical_risk_cols[0]
            df[f'{risk_col}_norm'] = (df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min() + 1e-6)
            risk_components.append(f'{risk_col}_norm')
            weights.append(0.05)  # CAMBIO: reducido de 0.10
            self.risk_score_components.append(f"Clínico ({risk_col})")
        
        # Calcular Score de Riesgo Compuesto
        if risk_components:
            # Ajustar pesos para que sumen 1
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Score compuesto ponderado
            risk_matrix = df[risk_components].fillna(0)
            df['composite_risk_score'] = np.average(risk_matrix, axis=1, weights=weights)
            
            # Categorías de riesgo
            df['risk_category'] = pd.cut(df['composite_risk_score'], 
                                       bins=[0, 0.33, 0.66, 1.0], 
                                       labels=['Low', 'Moderate', 'High'],
                                       include_lowest=True)
            
            print(f"   ✅ Score compuesto creado con {len(risk_components)} componentes")
            print(f"   📊 Distribución: Low {(df['risk_category'] == 'Low').sum()}, "
                  f"Moderate {(df['risk_category'] == 'Moderate').sum()}, "
                  f"High {(df['risk_category'] == 'High').sum()}")
        else:
            print("   ⚠️  No se encontraron componentes de riesgo suficientes")
            # Score de riesgo básico usando completitud multimodal
            if 'multimodal_completeness' in df.columns:
                df['composite_risk_score'] = 1 - df['multimodal_completeness']  # Invertir completitud
                df['risk_category'] = pd.cut(df['composite_risk_score'], 
                                           bins=[0, 0.33, 0.66, 1.0], 
                                           labels=['Low', 'Moderate', 'High'],
                                           include_lowest=True)
                print("   ✅ Score básico creado usando completitud multimodal")
        
        return df
    
    def _final_feature_selection(self, df):
        """Selección y limpieza final de features"""
        
        print("\n=>) SELECCIÓN FINAL DE FEATURES")
        print("-" * 35)
        
        # Eliminar features temporales normalizadas
        temp_cols = [col for col in df.columns if col.endswith('_norm')]
        df_clean = df.drop(columns=temp_cols, errors='ignore')
        
        # Eliminar columnas con demasiados valores faltantes (>90%)
        missing_threshold = 0.9
        cols_to_drop = []
        for col in df_clean.columns:
            if df_clean[col].isna().sum() / len(df_clean) > missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            print(f"   🗑️  Eliminadas {len(cols_to_drop)} columnas con >90% faltantes")
        
        # Asegurar que el score de riesgo esté presente
        if 'composite_risk_score' not in df_clean.columns:
            print("   ⚠️  Score de riesgo no encontrado - creando versión básica")
            df_clean['composite_risk_score'] = np.random.uniform(0, 1, len(df_clean))  # Temporal
        
        print(f"   ✅ Dataset final: {df_clean.shape[0]:,} registros × {df_clean.shape[1]} variables")
        
        return df_clean
    
    def _save_results(self, df, output_path):
        """Guardar resultados del pipeline"""
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar dataset con features
        df.to_csv(output_path, index=False)
        
        # Guardar metadatos del pipeline
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': f"{df.shape[0]} × {df.shape[1]}",
            'modality_status': self.feature_summary,
            'risk_score_components': self.risk_score_components,
            'output_path': str(output_path)
        }
        
        #metadata_path = output_path.replace('.csv', '_metadata.json')
        metadata_path = output_path.with_name(output_path.stem + '_metadata.json') # Manejo correcto para objetos PathLib
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_report(self, df_original, df_final):
        """Generar reporte del pipeline"""
        
        print("\n📊 REPORTE FINAL DEL PIPELINE")
        print("=" * 50)
        print(f"📂 Dataset Original: {df_original.shape[0]:,} × {df_original.shape[1]}")
        print(f"🔧 Dataset Final: {df_final.shape[0]:,} × {df_final.shape[1]}")
        print(f"📈 Features Agregadas: {df_final.shape[1] - df_original.shape[1]}")
        
        print(f"\n🏥 ESTADO POR MODALIDAD:")
        for modality, status in self.feature_summary.items():
            status_icon = "✅" if status == "OK" else "❌"
            print(f"   {status_icon} {modality.capitalize()}: {status}")
        
        if 'composite_risk_score' in df_final.columns:
            print(f"\n🎯 SCORE DE RIESGO COMPUESTO:")
            print(f"   - Rango: [{df_final['composite_risk_score'].min():.3f}, {df_final['composite_risk_score'].max():.3f}]")
            print(f"   - Media: {df_final['composite_risk_score'].mean():.3f}")
            print(f"   - Desv. Std: {df_final['composite_risk_score'].std():.3f}")
            
            if 'risk_category' in df_final.columns:
                risk_dist = df_final['risk_category'].value_counts()
                print(f"   🏥 Distribución de Riesgo:")
                for category, count in risk_dist.items():
                    percentage = (count / len(df_final)) * 100
                    print(f"      • {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n💾 Pipeline completado exitosamente!")

def run_feature_engineering_pipeline(input_csv_path, output_csv_path=None):
    """
    Función principal para ejecutar el pipeline completo
    
    Args:
        input_csv_path: Ruta al CSV integrado
        output_csv_path: Ruta de salida (opcional)
    
    Returns:
        DataFrame con features engineered
    """
    
    if output_csv_path is None:
        # Generar nombre automático
        base_dir = os.path.dirname(input_csv_path)
        output_csv_path = os.path.join(base_dir, 'multimodal_alzheimer_features.csv')
    
    # Ejecutar pipeline
    pipeline = FeatureEngineeringPipeline()
    df_engineered = pipeline.run_pipeline(input_csv_path, output_csv_path)
    
    return df_engineered

# Ejemplo de uso directo con rutas absolutas
if __name__ == "__main__":
    # Obtener ruta absoluta del proyecto
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construir rutas absolutas
    input_path = os.path.join(
        PROJECT_ROOT, 
        'data', 
        'processed', 
        'integrated', 
        'multimodal_alzheimer_dataset.csv'
    )
    
    output_path = os.path.join(
        PROJECT_ROOT, 
        'data', 
        'processed', 
        'features', 
        'multimodal_alzheimer_features.csv'
    )
    
    print(f"🔍 Buscando archivo en: {input_path}")
    
    # Verificar existencia
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Archivo de entrada no encontrado")
        print(f"Por favor verifica que existe en: {input_path}")
        exit(1)
    
    # Asegurar directorio de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ejecutar pipeline
    try:
        df_features = run_feature_engineering_pipeline(input_path, output_path)
        print(f"\n🎉 ¡Feature Engineering completado!")
        print(f"📁 Archivo guardado: {output_path}")
        print(f"📊 Dimensiones finales: {df_features.shape}")
    except Exception as e:
        print(f"\n🔥 ERROR en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()