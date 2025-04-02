import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_activity_data(n_subjects=100, days=30, seed=42):
    """
    Genera datos sintéticos de actividad diaria y sueño para pacientes control, 
    con deterioro cognitivo leve (MCI) y con Alzheimer.
    
    Parámetros:
    - n_subjects: Número de sujetos a generar
    - days: Número de días de datos para cada sujeto
    - seed: Semilla para reproducibilidad
    
    Retorna:
    - DataFrame con datos de actividad y sueño
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Definir grupos diagnósticos
    diagnostic_groups = ['Control', 'MCI', 'Alzheimer']
    group_probabilities = [0.4, 0.3, 0.3]  # Proporciones de cada grupo
    
    # Parámetros para cada grupo (basados en literatura)
    activity_params = {
        'Control': {
            'steps_mean': 7500, 'steps_std': 1500,
            'activity_duration_mean': 240, 'activity_duration_std': 45,  # minutos
            'sleep_duration_mean': 420, 'sleep_duration_std': 30,  # minutos
            'sleep_disruptions_mean': 2, 'sleep_disruptions_std': 1,
            'day_variance': 0.1  # variabilidad día a día
        },
        'MCI': {
            'steps_mean': 6000, 'steps_std': 1800,
            'activity_duration_mean': 180, 'activity_duration_std': 60,
            'sleep_duration_mean': 390, 'sleep_duration_std': 45,
            'sleep_disruptions_mean': 4, 'sleep_disruptions_std': 1.5,
            'day_variance': 0.2
        },
        'Alzheimer': {
            'steps_mean': 4000, 'steps_std': 2000,
            'activity_duration_mean': 120, 'activity_duration_std': 75,
            'sleep_duration_mean': 360, 'sleep_duration_std': 60,
            'sleep_disruptions_mean': 6, 'sleep_disruptions_std': 2,
            'day_variance': 0.3
        }
    }
    
    # Características temporales que cambian con el tiempo (para simular progresión)
    def apply_temporal_changes(base_value, day, group, feature, rate=0.002):
        # Los grupos Control, MCI y Alzheimer tienen diferentes tasas de cambio
        group_factors = {'Control': 0.0005, 'MCI': 0.002, 'Alzheimer': 0.004}
        
        if feature in ['steps', 'activity_duration', 'sleep_duration']:
            # Estas características tienden a disminuir con el tiempo
            return base_value * (1 - group_factors[group] * day)
        else:
            # Las disrupciones tienden a aumentar
            return base_value * (1 + group_factors[group] * day)
    
    # Generación de datos
    data = []
    start_date = datetime(2023, 1, 1)
    
    for subject_id in range(1, n_subjects + 1):
        # Asignar grupo diagnóstico
        diagnosis = np.random.choice(diagnostic_groups, p=group_probabilities)
        
        # Características demográficas
        age = np.random.normal(75, 8) if diagnosis == 'Alzheimer' else \
              np.random.normal(70, 7) if diagnosis == 'MCI' else \
              np.random.normal(65, 10)
        
        gender = np.random.choice(['M', 'F'])
        
        # Generar datos diarios
        for day in range(days):
            date = start_date + timedelta(days=day)
            params = activity_params[diagnosis]
            
            # Aplicar variabilidad día a día y tendencia temporal
            day_factor = 1 + np.random.uniform(-params['day_variance'], params['day_variance'])
            
            # Características de actividad
            steps = max(0, apply_temporal_changes(
                np.random.normal(params['steps_mean'], params['steps_std']), 
                day, diagnosis, 'steps') * day_factor)
            
            activity_duration = max(0, apply_temporal_changes(
                np.random.normal(params['activity_duration_mean'], params['activity_duration_std']), 
                day, diagnosis, 'activity_duration') * day_factor)
            
            # Características de sueño
            sleep_duration = max(0, apply_temporal_changes(
                np.random.normal(params['sleep_duration_mean'], params['sleep_duration_std']), 
                day, diagnosis, 'sleep_duration') * day_factor)
            
            sleep_disruptions = max(0, apply_temporal_changes(
                np.random.normal(params['sleep_disruptions_mean'], params['sleep_disruptions_std']), 
                day, diagnosis, 'sleep_disruptions'))
            
            # Características adicionales
            sedentary_hours = max(8, 16 - (activity_duration / 60)) * day_factor
            
            # Patrones de actividad nocturna (wandering) - más común en Alzheimer
            night_activity = 0
            if diagnosis == 'Alzheimer':
                night_activity = np.random.exponential(scale=3) * (1 + 0.02 * day)
            elif diagnosis == 'MCI':
                night_activity = np.random.exponential(scale=1.5) * (1 + 0.01 * day)
            else:
                night_activity = np.random.exponential(scale=0.5)
            
            # Añadir ruido aleatorio para días atípicos (ej. visitas médicas, etc.)
            if random.random() < 0.05:  # 5% de días atípicos
                steps *= random.uniform(0.3, 1.5)
                activity_duration *= random.uniform(0.3, 1.5)
                sleep_duration *= random.uniform(0.7, 1.2)
            
            # Crear registro
            record = {
                'subject_id': f'S_{subject_id:03d}',
                'date': date.strftime('%Y-%m-%d'),
                'diagnosis': diagnosis,
                'age': round(age + day/365, 1),  # Incrementar edad ligeramente
                'gender': gender,
                'steps': int(steps),
                'activity_minutes': round(activity_duration, 1),
                'sleep_minutes': round(sleep_duration, 1),
                'sleep_disruptions': round(sleep_disruptions, 1),
                'sedentary_hours': round(sedentary_hours, 1),
                'night_activity_minutes': round(night_activity, 1)
            }
            
            data.append(record)
    
    # Convertir a DataFrame
    df = pd.DataFrame(data)
    
    # Añadir algunas características calculadas que podrían ser útiles
    df['sleep_efficiency'] = df['sleep_minutes'] / (df['sleep_minutes'] + df['sleep_disruptions'] * 5)
    df['activity_ratio'] = df['activity_minutes'] / (df['activity_minutes'] + df['sedentary_hours'] * 60)
    
    return df
