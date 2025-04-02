# Documentación de Datos - Proyecto Alzheimer Multimodal Monitoring

## Descripción General

Este documento describe los conjuntos de datos utilizados en el proyecto, su estructura, origen y consideraciones importantes.

## 1. Datos ADNI (Alzheimer's Disease Neuroimaging Initiative)

### Origen
- **Fuente**: Alzheimer's Disease Neuroimaging Initiative
- **Sitio Web**: [adni.loni.usc.edu](https://adni.loni.usc.edu/)
- **Versión**: ADNI-3
- **Fecha de Acceso**: [Fecha]

### Conjuntos de Datos Disponibles
- `ADNIMERGE.csv`: Datos consolidados de participantes
- `Demographics.csv`: Información demográfica
- Archivos MRI: Imágenes estructurales T1
- Archivos PET: Imágenes FDG-PET y Amyloid-PET
- Biomarcadores: Datos de líquido cefalorraquídeo

### Variables Clave
- **ID**: Identificador único de participante
- **DX_bl**: Diagnóstico basal (CN: Control Normal, MCI: Deterioro Cognitivo Leve, AD: Alzheimer)
- **AGE**: Edad del participante
- **PTGENDER**: Género
- **MMSE**: Mini-Mental State Examination (evaluación cognitiva)
- **CDRSB**: Clinical Dementia Rating Sum of Boxes
- **APOE4**: Número de alelos APOE ε4 (factor de riesgo genético)

### Limitaciones
- Los datos requieren preprocesamiento para homogeneizar formatos
- Datos faltantes en algunas variables
- Las imágenes requieren procesamiento especializado

## 2. Datos OASIS-3 (Open Access Series of Imaging Studies)

### Origen
- **Fuente**: Washington University, Harvard University, Massachusetts General Hospital
- **Sitio Web**: [oasis-brains.org](https://www.oasis-brains.org/)
- **Versión**: OASIS-3
- **Fecha de Acceso**: [Fecha]

### Conjuntos de Datos Disponibles
- `oasis3_clinical_data.csv`: Datos clínicos y demográficos
- Archivos MRI: Imágenes estructurales
- Datos longitudinales: Seguimiento de participantes

### Variables Clave
- **ID**: Identificador único de participante
- **Age**: Edad del participante
- **Gender**: Género
- **CDR**: Clinical Dementia Rating
- **MMSE**: Mini-Mental State Examination
- **Diagnosis**: Diagnóstico clínico

### Limitaciones
- Menor número de biomarcadores que ADNI
- Heterogeneidad en protocolos de adquisición
- Datos faltantes en seguimiento longitudinal

## 3. Datos Sintéticos de Actividad y Sueño

### Origen
- **Fuente**: Generación sintética basada en literatura científica
- **Método**: Algoritmo de simulación paramétrica
- **Fecha de Generación**: [Fecha]

### Estructura del Dataset
- `synthetic_activity_data.csv`: Datos de actividad diaria

### Variables Clave
- **subject_id**: Identificador único del participante
- **date**: Fecha de registro
- **diagnosis**: Diagnóstico (Control, MCI, Alzheimer)
- **steps**: Número de pasos diarios
- **activity_minutes**: Minutos de actividad física
- **sleep_minutes**: Duración del sueño
- **sleep_disruptions**: Número de interrupciones del sueño
- **night_activity_minutes**: Actividad nocturna (potencial deambulación)

### Limitaciones
- Datos sintéticos basados en promedios de literatura
- No representan variabilidad individual real
- Sirven como complemento a datos reales

## Consideraciones de Privacidad y Ética

- Los datos de ADNI y OASIS están desidentificados siguiendo normativas HIPAA
- El uso de estos datos está sujeto a los términos y condiciones de cada repositorio
- Los resultados derivados deben citarse adecuadamente
- No se debe intentar reidentificar a los participantes

## Uso de los Datos

Este proyecto utiliza estos datos únicamente con fines educativos y de investigación, como parte de un portfolio profesional. No se realizarán diagnósticos clínicos basados en estos análisis.
