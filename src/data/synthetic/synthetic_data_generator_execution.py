# Importar el generador de datos creado anteriormente
from synthetic_activity_data_generator import generate_synthetic_activity_data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear directorio para datos sintéticos si no existe
os.makedirs('data/raw/synthetic', exist_ok=True)

# Configuración de parámetros
n_subjects = 300  # 300 sujetos (similar a cohortes de investigación)
days = 180  # 6 meses de datos

# Generar datos
print("Generando datos sintéticos de actividad y sueño...")
activity_data = generate_synthetic_activity_data(n_subjects=n_subjects, days=days, seed=42)

# Guardar datos
output_path = 'data/raw/synthetic/synthetic_activity_sleep_data.csv'
activity_data.to_csv(output_path, index=False)
print(f"Datos guardados en: {output_path}")

# Resumen de estadísticas
print("\nEstadísticas por grupo diagnóstico:")
stats = activity_data.groupby('diagnosis')[['steps', 'activity_minutes', 'sleep_minutes', 
                                          'sleep_disruptions', 'night_activity_minutes']].mean()
print(stats)

# Generar visualizaciones básicas para verificación
print("\nGenerando visualizaciones para verificación de datos...")
os.makedirs('data/raw/synthetic/plots', exist_ok=True)

# Distribución de pasos por grupo diagnóstico
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='steps', data=activity_data)
plt.title('Distribución de pasos diarios por grupo diagnóstico')
plt.savefig('data/raw/synthetic/plots/steps_distribution.png')

# Patrones de sueño
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='sleep_minutes', data=activity_data)
plt.title('Duración del sueño por grupo diagnóstico')
plt.savefig('data/raw/synthetic/plots/sleep_duration.png')

# Actividad nocturna (deambulación)
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='night_activity_minutes', data=activity_data)
plt.title('Actividad nocturna por grupo diagnóstico')
plt.savefig('data/raw/synthetic/plots/night_activity.png')

print("Verificación completa. La fase de preparación de datos está finalizada.")
