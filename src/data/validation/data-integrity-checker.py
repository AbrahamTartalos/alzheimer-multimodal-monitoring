import os
import pandas as pd
import numpy as np
import nibabel as nib
from datetime import datetime

def verify_data_integrity(base_dir='data/raw'):
    """
    Verifica la integridad de los datos descargados y genera un informe.
    
    Args:
        base_dir: Directorio base donde se encuentran los datos
    
    Returns:
        dict: Informe de estado de los datos
    """
    print(f"Verificando integridad de datos en {base_dir}...")
    
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'datasets': {},
        'missing_files': [],
        'corrupted_files': [],
        'summary': {}
    }
    
    # Verificar datos ADNI
    adni_dir = os.path.join(base_dir, 'adni')
    if os.path.exists(adni_dir):
        adni_files = os.listdir(adni_dir)
        report['datasets']['adni'] = {
            'path': adni_dir,
            'files_count': len(adni_files),
            'files': adni_files
        }
        
        # Verificar archivos importantes
        essential_files = ['ADNIMERGE.csv', 'Demographics.csv']
        for file in essential_files:
            file_path = os.path.join(adni_dir, file)
            if not os.path.exists(file_path):
                report['missing_files'].append(file_path)
            else:
                try:
                    # Intentar cargar el archivo CSV
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        report['datasets']['adni'][file] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': df.columns.tolist()
                        }
                except Exception as e:
                    report['corrupted_files'].append({
                        'file': file_path,
                        'error': str(e)
                    })
    else:
        report['missing_files'].append(adni_dir)
    
    # Verificar datos OASIS
    oasis_dir = os.path.join(base_dir, 'oasis')
    if os.path.exists(oasis_dir):
        oasis_files = os.listdir(oasis_dir)
        report['datasets']['oasis'] = {
            'path': oasis_dir,
            'files_count': len(oasis_files),
            'files': oasis_files
        }
        
        # Verificar archivos de clínica y MRI
        if 'oasis3_clinical_data.csv' in oasis_files:
            file_path = os.path.join(oasis_dir, 'oasis3_clinical_data.csv')
            try:
                df = pd.read_csv(file_path)
                report['datasets']['oasis']['clinical_data'] = {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            except Exception as e:
                report['corrupted_files'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        # Verificar archivos de neuroimagen
        mri_count = 0
        for file in oasis_files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                mri_count += 1
                # Verificar aleatoriamente algunas imágenes
                if mri_count < 5:  # Solo verificamos las primeras imágenes
                    try:
                        nib.load(os.path.join(oasis_dir, file))
                    except Exception as e:
                        report['corrupted_files'].append({
                            'file': os.path.join(oasis_dir, file),
                            'error': str(e)
                        })
        
        report['datasets']['oasis']['mri_count'] = mri_count
    else:
        report['missing_files'].append(oasis_dir)
    
    # Verificar datos de actividad
    activity_dir = os.path.join(base_dir, 'activity')
    if os.path.exists(activity_dir):
        activity_files = os.listdir(activity_dir)
        report['datasets']['activity'] = {
            'path': activity_dir,
            'files_count': len(activity_files),
            'files': activity_files
        }
        
        # Verificar archivo de actividad sintética
        if 'synthetic_activity_data.csv' in activity_files:
            file_path = os.path.join(activity_dir, 'synthetic_activity_data.csv')
            try:
                df = pd.read_csv(file_path)
                report['datasets']['activity']['synthetic_data'] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'unique_subjects': df['subject_id'].nunique() if 'subject_id' in df.columns else 'unknown',
                    'column_names': df.columns.tolist()
                }
            except Exception as e:
                report['corrupted_files'].append({
                    'file': file_path,
                    'error': str(e)
                })
    else:
        report['missing_files'].append(activity_dir)
    
    # Generar resumen
    report['summary'] = {
        'total_datasets': len(report['datasets']),
        'missing_files_count': len(report['missing_files']),
        'corrupted_files_count': len(report['corrupted_files']),
        'adni_available': 'adni' in report['datasets'],
        'oasis_available': 'oasis' in report['datasets'],
        'activity_available': 'activity' in report['datasets']
    }
    
    # Informe de verificación
    print(f"Verificación completada.")
    print(f"Datasets encontrados: {report['summary']['total_datasets']}")
    print(f"Archivos faltantes: {report['summary']['missing_files_count']}")
    print(f"Archivos corruptos: {report['summary']['corrupted_files_count']}")
    
    return report

def save_verification_report(report, output_path='data/verification_report.json'):
    """Guarda el informe de verificación en formato JSON"""
    import json
    
    # Convertir a JSON amigable (eliminar objetos no serializables)
    json_report = {k: v for k, v in report.items() if k != 'df_samples'}
    
    with open(output_path, 'w') as f:
        json.dump(json_report, f, indent=4)
    
    print(f"Informe guardado en: {output_path}")

# Ejecutar la verificación
if __name__ == "__main__":
    report = verify_data_integrity()
    save_verification_report(report)
