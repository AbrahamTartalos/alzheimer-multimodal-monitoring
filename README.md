# Monitorización y Predicción Multimodal del Alzheimer

<div align="center">
  <h3>Sistema de Detección Temprana y Monitoreo de Alzheimer mediante Análisis Multimodal</h3>
  <p>Un proyecto de ciencia de datos end-to-end para la predicción y monitorización del Alzheimer</p>
  
  [![Licencia](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Contribuciones Bienvenidas](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
</div>

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Objetivos](#-objetivos)
- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Fases del Proyecto](#-fases-del-proyecto)
- [Datos](#-datos)
- [Modelos Implementados](#-modelos-implementados)
- [Dashboard Interactivo](#-dashboard-interactivo)
- [Resultados](#-resultados)
- [Privacidad y Consideraciones Éticas](#-privacidad-y-consideraciones-éticas)
- [Contribuciones](#-contribuciones)
- [Sobre el Autor](#-sobre-el-autor)
- [Licencia](#-licencia)


## 🔍 Descripción

Este proyecto desarrolla un sistema integral de detección temprana y monitoreo de Alzheimer mediante análisis multimodal de datos. Integra datos de neuroimagen, biomarcadores clínicos, información genética y patrones de actividad diaria para identificar señales tempranas y monitorear la progresión de la enfermedad de Alzheimer.

El sistema utiliza técnicas avanzadas de machine learning y análisis de series temporales para crear un score de riesgo compuesto que permite una evaluación comprehensiva del estado cognitivo del paciente.

## 🎯 Objetivos

- **Detección Temprana**: Desarrollar un modelo predictivo para identificar signos tempranos de Alzheimer con >85% de precisión
- **Estratificación de Riesgo**: Crear un sistema de estratificación de riesgo con explicabilidad mediante técnicas de XAI
- **Monitoreo Continuo**: Implementar algoritmos de detección de cambios cognitivos sutiles a través del tiempo
- **Visualización Interactiva**: Diseñar un dashboard interactivo para visualización de factores de riesgo y progresión
- **Validación Clínica**: Validar el sistema con evaluación retrospectiva (>80% sensibilidad, >75% especificidad)

## ✨ Características Principales

- **Análisis Multimodal**: Integración de datos de neuroimagen, biomarcadores, genética y actividad diaria
- **Modelos Avanzados**: Implementación de múltiples algoritmos de ML incluyendo Random Forest, XGBoost y redes neuronales
- **Series Temporales**: Análisis de la evolución temporal del score de riesgo compuesto
- **Explicabilidad**: Utilización de SHAP para interpretación de modelos
- **Dashboard Interactivo**: Interfaz web desarrollada con Plotly-Dash
- **Balanceo de Datos**: Implementación de técnicas SMOTE para manejo de datos desbalanceados
- **Seguimiento de Experimentos**: Integración con MLflow para tracking de modelos y métricas

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Datos ADNI    │    │  Datos Sintéticos│    │  Procesamiento  │
│                 │    │                 │    │                 │
│ • Neuroimagen   │    │ • Actividad     │    │ • Limpieza      │
│ • Biomarcadores │───▶│ • Sueño         │───▶│ • Normalización │
│ • Genética      │    │                 │    │ • Feature Eng.  │
│ • Clínicos      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   Evaluación    │    │   Modelos ML    │
│                 │    │                 │    │                 │
│ • Visualización │◀───│ • Métricas      │◀───│ • Clasificación │
│ • Interactividad│    │ • Validación    │    │ • Regresión     │
│ • Monitoreo     │    │ • Explicabilidad│    │ • Series Temp.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Estructura del Proyecto

```plaintext
Alzheimer-Multimodal-Monitoring/
│
├── data/                         # Datos utilizados en el proyecto
│   ├── raw/                      # Datos sin procesar (originales)
│   │   ├── ADNI/                 # Datos originales de ADNI
│   │   │   ├── images/           # Imágenes DICOM o NIfTI
│   │   │   │   ├── MRI/          # MRI estructural
│   │   │   │   └── PET/          # Imágenes PET
│   │   │   ├── biomarkers/       # Biomarcadores CSF y plasma
│   │   │   ├── genetics/         # Datos genéticos
│   │   │   ├── clinical/         # Diagnóstico y datos clínicos
│   │   │   └── demographics/     # Datos demográficos
│   │   └── synthetic/            # Datos sintéticos de actividad y sueño
│   ├── interim/                  # Datos en transformación
│   ├── processed/                # Datos finales listos para modelado
│   └── external/                 # Datos de fuentes externas adicionales
│
├── src/                          # Código fuente del proyecto
│   ├── data/                     # Scripts de carga y procesamiento
│   ├── feature_engineering/      # Extracción y transformación de características
│   ├── models/                   # Modelos de machine learning
│   └── visualization/            # Scripts para visualización
│
├── models/                       # Modelos entrenados y resultados
├── docs/                         # Documentación del proyecto
├── notebooks/                    # Jupyter notebooks para exploración
├── reports/                      # Informes generados
│   └── figures/                  # Gráficos y visualizaciones
├── dashboard/                    # Aplicación web del dashboard
├── landing_page/                 # Landing page del proyecto
│
├── README.md                     # Este archivo
├── requirements.txt              # Dependencias del proyecto
├── LICENSE                       # Licencia Apache 2.0
└── .gitignore                    # Archivos a ignorar en Git
```

## 🛠️ Tecnologías Utilizadas

### **Lenguajes y Frameworks**
- **Python 3.10+**: Lenguaje principal del proyecto
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Machine Learning

### **Machine Learning y Análisis**
- **XGBoost**: Gradient boosting
- **MLflow**: Tracking de experimentos y modelos
- **SHAP**: Explicabilidad de modelos
- **SMOTE**: Balanceo de datos desbalanceados

### **Visualización y Dashboard**
- **Plotly**: Visualizaciones interactivas
- **Dash**: Framework para aplicaciones web
- **Seaborn**: Visualizaciones estadísticas

### **Procesamiento de Neuroimagen**
- **nibabel**: Manipulación de imágenes médicas

## 🚀 Instalación

### **Prerrequisitos**
- Python 3.10 o superior
- pip o conda

### **Instalación**

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/Alzheimer-Multimodal-Monitoring.git
cd Alzheimer-Multimodal-Monitoring
```

2. **Crear entorno virtual**
```bash
# Con conda
conda create -n alzheimer_project python=3.10
conda activate alzheimer_project

# Con venv
python -m venv alzheimer_env
source alzheimer_env/bin/activate  # En Windows: alzheimer_env\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar MLflow (opcional)**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## 📖 Uso

### **Exploración de Datos**
```bash
# Ejecutar notebooks de exploración
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### **Entrenamiento de Modelos**
```bash
# Entrenar modelos de clasificación
python src/models/train_classification_models.py

# Entrenar modelos de series temporales
python src/models/train_time_series_models.py
```

### **Ejecutar Dashboard**
```bash
# Lanzar dashboard interactivo
python dashboard/app.py
```

### **Visualizar Landing Page**
```bash
# Ejecutar landing page
python landing_page/app.py
```

## 📊 Fases del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **1. Preparación y Adquisición de Datos** | Descarga y organización de datos ADNI | ✅ Completado |
| **2. Exploración y Preprocesamiento** | EDA, limpieza y transformación de datos | ✅ Completado |
| **3. Feature Engineering y Selección** | Extracción y selección de características | ✅ Completado |
| **4. Desarrollo de Modelos** | Implementación y entrenamiento de modelos | ✅ Completado |
| **5. Evaluación y Validación** | Evaluación de performance y validación | ✅ Completado |
| **6. Implementación Dashboard** | Desarrollo de interfaz web interactiva | ✅ Completado |

## 📊 Datos

### **Fuentes de Datos**
- **ADNI (Alzheimer's Disease Neuroimaging Initiative)**: Datos principales del proyecto
  - Neuroimagen: MRI estructural y PET
  - Biomarcadores: CSF y plasma
  - Datos genéticos: APOE y otros marcadores
  - Datos clínicos: Diagnósticos y evaluaciones cognitivas
  - Datos demográficos: Edad, género, educación

- **Datos Sintéticos**: Generados para complementar el análisis
  - Patrones de actividad diaria
  - Datos de sueño

### **Consideraciones de Privacidad**
Todos los datos utilizados siguen las directrices de privacidad establecidas por ADNI. Los datos están desidentificados y se utilizan únicamente para fines de investigación.

## 🤖 Modelos Implementados

### **Modelos de Clasificación**
- **Random Forest**: Modelo ensemble para clasificación multiclase
- **XGBoost**: Gradient boosting optimizado
- **Redes Neuronales**: Deep learning para patrones complejos
- **Regresión Logística**: Modelo baseline interpretable

### **Modelos de Regresión**
- **Regresión Lineal**: Para variables continuas
- **Support Vector Regression**: Para relaciones no lineales
- **Random Forest Regressor**: Ensemble para regresión

### **Modelos de Series Temporales**
- **ARIMA**: Para análisis de tendencias temporales
- **LSTM**: Redes neuronales recurrentes
- **Prophet**: Para forecasting del score de riesgo

### **Estratificación de Riesgo**
- **Clustering**: K-means y clustering jerárquico
- **Score Compuesto**: Combinación ponderada de múltiples factores

## 📈 Dashboard Interactivo

El dashboard desarrollado en Plotly-Dash incluye:

- **Visualización de Factores de Riesgo**: Gráficos interactivos de biomarcadores y neuroimagen
- **Evolución Temporal**: Seguimiento del score de riesgo a través del tiempo
- **Explicabilidad**: Visualización de importancia de características usando SHAP
- **Comparación de Modelos**: Métricas de performance de diferentes algoritmos
- **Predicciones en Tiempo Real**: Interfaz para nuevas predicciones

## 📊 Resultados

### **Métricas Objetivo**
- **Precisión**: >85% (objetivo)
- **Sensibilidad**: >80% (objetivo)
- **Especificidad**: >75% (objetivo)

### **Modelos de Mejor Performance**
*[Esta sección se actualizará conforme se completen las evaluaciones]*

## 🔒 Privacidad y Consideraciones Éticas

- **Datos ADNI**: Cumplimiento total con las directrices de privacidad de ADNI
- **Desidentificación**: Todos los datos están completamente desidentificados
- **Uso Ético**: Los modelos se desarrollan para fines de investigación y mejora del diagnóstico
- **Transparencia**: Código abierto y metodología transparente

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Este proyecto está completamente abierto a colaboraciones.

### **Cómo Contribuir**
1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### **Áreas de Contribución**
- Mejoras en modelos de ML
- Nuevas visualizaciones para el dashboard
- Optimización de código
- Documentación
- Testing y validación

## 👨‍💻 Sobre el Autor

**Abraham Tartalos**

Soy un profesional apasionado por la ciencia de datos y la inteligencia artificial aplicada. Mi enfoque se centra en desarrollar soluciones innovadoras que puedan tener un impacto real en la vida de las personas. Este proyecto representa mi compromiso con la investigación de vanguardia en el campo de las enfermedades neurodegenerativas.

Mi experiencia abarca desde el análisis de datos complejos hasta el desarrollo de sistemas de machine learning escalables, siempre con un enfoque en la aplicabilidad práctica y la ética en el uso de datos médicos.

**Conecta conmigo:**
- 💼 LinkedIn: [Abraham Tartalos](https://www.linkedin.com/in/abrahamtartalos)
- 📧 Email: [a través de LinkedIn]

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">
  <p>⭐ Si este proyecto te parece útil, ¡no olvides darle una estrella! ⭐</p>
  <p>Desarrollado con ❤️ por Abraham Tartalos</p>
</div>