# Análisis Exploratorio de Datos Genéticos para Proyecto de Detección de Alzheimer
# ===================================================================

# Importamos las bibliotecas necesarias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuraciones básicas
%matplotlib inline
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Iniciando análisis exploratorio de datos genéticos...")

# 1. Cargar los datos genéticos
# ===================================================================

# Definimos la ruta a los datos genéticos de ADNI
base_dir = "../../data/raw/"
adni_genetics_path = os.path.join(base_dir, "ADNI/genetics")
oasis_genetics_path = os.path.join(base_dir, "OASIS/genetics")

# Verificamos que los directorios existan
print(f"Verificando acceso a datos genéticos...")
adni_exists = os.path.exists(adni_genetics_path)
oasis_exists = os.path.exists(oasis_genetics_path)

print(f"Datos genéticos ADNI disponibles: {adni_exists}")
print(f"Datos genéticos OASIS disponibles: {oasis_exists}")

# Función para cargar datos de diferentes fuentes
def load_genetic_data(dataset_name):
    """
    Cargar datos genéticos de un dataset específico
    
    Args:
        dataset_name: Nombre del dataset ('ADNI' o 'OASIS')
    
    Returns:
        DataFrame con datos genéticos o None si no está disponible
    """
    if dataset_name.upper() == 'ADNI' and adni_exists:
        # Ejemplo para ADNI (ajustar según la estructura real de archivos)
        try:
            # NOTA: Ajustar nombres de archivo según los reales en tu estructura
            apoe_file = os.path.join(adni_genetics_path, "APOERES.csv")
            snp_file = os.path.join(adni_genetics_path, "SNPS.csv")
            
            if os.path.exists(apoe_file):
                apoe_df = pd.read_csv(apoe_file)
                print(f"Datos APOE cargados: {apoe_df.shape[0]} sujetos, {apoe_df.shape[1]} variables")
                return apoe_df
            elif os.path.exists(snp_file):
                snp_df = pd.read_csv(snp_file)
                print(f"Datos SNP cargados: {snp_df.shape[0]} sujetos, {snp_df.shape[1]} SNPs")
                return snp_df
            else:
                print(f"No se encontraron archivos de datos genéticos en {adni_genetics_path}")
                # Crear datos de muestra para demostración si no hay datos reales
                return create_sample_genetic_data(dataset_name)
        except Exception as e:
            print(f"Error al cargar datos genéticos de ADNI: {e}")
            return create_sample_genetic_data(dataset_name)
            
    elif dataset_name.upper() == 'OASIS' and oasis_exists:
        # Ejemplo para OASIS (ajustar según la estructura real)
        try:
            # NOTA: Ajustar nombres de archivo según los reales en tu estructura
            genetics_file = os.path.join(oasis_genetics_path, "oasis_genetics.csv")
            
            if os.path.exists(genetics_file):
                genetics_df = pd.read_csv(genetics_file)
                print(f"Datos genéticos OASIS cargados: {genetics_df.shape[0]} sujetos, {genetics_df.shape[1]} variables")
                return genetics_df
            else:
                print(f"No se encontraron archivos de datos genéticos en {oasis_genetics_path}")
                # Crear datos de muestra para demostración
                return create_sample_genetic_data(dataset_name)
        except Exception as e:
            print(f"Error al cargar datos genéticos de OASIS: {e}")
            return create_sample_genetic_data(dataset_name)
    else:
        print(f"Datos de {dataset_name} no disponibles. Usando datos de muestra para demostración.")
        return create_sample_genetic_data(dataset_name)

# Función para crear datos genéticos de muestra para demostración
def create_sample_genetic_data(dataset_name, n_subjects=100):
    """
    Crea un conjunto de datos genéticos sintéticos para demostración
    
    Args:
        dataset_name: Nombre del dataset ('ADNI' o 'OASIS')
        n_subjects: Número de sujetos a generar
    
    Returns:
        DataFrame con datos genéticos sintéticos
    """
    print(f"Creando datos genéticos sintéticos para {dataset_name}...")
    
    np.random.seed(42)  # Para reproducibilidad
    
    # Crear IDs de sujeto
    subject_ids = [f"{dataset_name.lower()}_subject_{i+1:03d}" for i in range(n_subjects)]
    
    # Genotipos APOE (e2/e2, e2/e3, e2/e4, e3/e3, e3/e4, e4/e4)
    apoe_genotypes = np.random.choice(
        ['22', '23', '24', '33', '34', '44'], 
        size=n_subjects, 
        p=[0.01, 0.11, 0.02, 0.60, 0.23, 0.03]  # Distribución aproximada en la población
    )
    
    # Número de alelos APOE4 (0, 1, 2)
    apoe4_count = [genotype.count('4') for genotype in apoe_genotypes]
    
    # SNPs de riesgo comunes (simulados)
    common_risk_snps = ['rs429358', 'rs7412', 'rs75932628', 'rs143332484', 'rs63750847']
    snp_data = {}
    
    for snp in common_risk_snps:
        # Generar genotipos (0: AA, 1: AB, 2: BB) con distribución realista
        if snp in ['rs429358', 'rs7412']:  # SNPs que determinan APOE
            # Estos deben ser coherentes con el genotipo APOE
            snp_data[snp] = np.random.choice([0, 1, 2], size=n_subjects, p=[0.7, 0.25, 0.05])
        else:
            snp_data[snp] = np.random.choice([0, 1, 2], size=n_subjects, p=[0.85, 0.14, 0.01])
    
    # Diagnóstico (CN: Cognitivamente Normal, MCI: Deterioro Cognitivo Leve, AD: Alzheimer)
    # La distribución depende de APOE4
    diagnosis = []
    for count in apoe4_count:
        if count == 0:
            diag = np.random.choice(['CN', 'MCI', 'AD'], p=[0.7, 0.25, 0.05])
        elif count == 1:
            diag = np.random.choice(['CN', 'MCI', 'AD'], p=[0.5, 0.35, 0.15])
        else:  # count == 2
            diag = np.random.choice(['CN', 'MCI', 'AD'], p=[0.3, 0.4, 0.3])
        diagnosis.append(diag)
    
    # Crear DataFrame
    data = {
        'subject_id': subject_ids,
        'APOE_genotype': apoe_genotypes,
        'APOE4_count': apoe4_count,
        'diagnosis': diagnosis
    }
    
    # Añadir SNPs
    for snp, values in snp_data.items():
        data[snp] = values
    
    # Edad - ligeramente correlacionada con diagnóstico
    age = []
    for diag in diagnosis:
        if diag == 'CN':
            age.append(np.random.normal(68, 5))
        elif diag == 'MCI':
            age.append(np.random.normal(72, 5))
        else:  # AD
            age.append(np.random.normal(75, 5))
    
    data['age'] = [max(55, min(90, int(a))) for a in age]  # Rango realista 55-90
    
    # Sexo
    data['sex'] = np.random.choice(['M', 'F'], size=n_subjects)
    
    # Polygenic Risk Score (simulado)
    # Correlacionado con APOE4 y diagnóstico
    prs = []
    for i in range(n_subjects):
        base_prs = np.random.normal(0, 1)
        apoe_effect = apoe4_count[i] * 0.5  # APOE4 aumenta el PRS
        diag_effect = 0
        if diagnosis[i] == 'MCI':
            diag_effect = 0.3
        elif diagnosis[i] == 'AD':
            diag_effect = 0.7
        
        final_prs = base_prs + apoe_effect + diag_effect + np.random.normal(0, 0.2)
        prs.append(final_prs)
    
    data['PRS'] = prs
    
    # Crear DataFrame final
    df = pd.DataFrame(data)
    print(f"Datos genéticos sintéticos creados: {df.shape[0]} sujetos, {df.shape[1]} variables")
    
    return df

# Cargar datos de ambas fuentes
adni_genetics_df = load_genetic_data('ADNI')
oasis_genetics_df = load_genetic_data('OASIS')

# 2. Exploración inicial de los datos
# ===================================================================

print("\n--- Exploración de datos genéticos de ADNI ---")
if adni_genetics_df is not None:
    print("Dimensiones:", adni_genetics_df.shape)
    print("Primeras filas:")
    display(adni_genetics_df.head())
    print("\nInformación de las columnas:")
    display(adni_genetics_df.info())
    print("\nEstadísticas descriptivas:")
    display(adni_genetics_df.describe(include='all'))
    
    # Contar valores nulos
    null_counts = adni_genetics_df.isna().sum()
    print("\nValores nulos por columna:")
    display(null_counts[null_counts > 0])
    
    # Contar valores únicos para variables categóricas
    categorical_cols = adni_genetics_df.select_dtypes(include=['object', 'category']).columns
    print("\nRecuento de valores únicos en variables categóricas:")
    for col in categorical_cols:
        if len(adni_genetics_df[col].unique()) < 20:  # Solo mostrar si hay pocos valores únicos
            print(f"\n{col}:")
            display(adni_genetics_df[col].value_counts())

print("\n--- Exploración de datos genéticos de OASIS ---")
if oasis_genetics_df is not None:
    print("Dimensiones:", oasis_genetics_df.shape)
    print("Primeras filas:")
    display(oasis_genetics_df.head())
    print("\nInformación de las columnas:")
    display(oasis_genetics_df.info())
    print("\nEstadísticas descriptivas:")
    display(oasis_genetics_df.describe(include='all'))
    
    # Contar valores nulos
    null_counts = oasis_genetics_df.isna().sum()
    print("\nValores nulos por columna:")
    display(null_counts[null_counts > 0])
    
    # Contar valores únicos para variables categóricas
    categorical_cols = oasis_genetics_df.select_dtypes(include=['object', 'category']).columns
    print("\nRecuento de valores únicos en variables categóricas:")
    for col in categorical_cols:
        if len(oasis_genetics_df[col].unique()) < 20:  # Solo mostrar si hay pocos valores únicos
            print(f"\n{col}:")
            display(oasis_genetics_df[col].value_counts())

# 3. Análisis de APOE y su relación con el diagnóstico
# ===================================================================

print("\n--- Análisis de APOE y su relación con el diagnóstico ---")

def analyze_apoe_distribution(df, dataset_name):
    """Analiza la distribución de genotipos APOE y su relación con el diagnóstico"""
    if df is None or 'APOE_genotype' not in df.columns or 'diagnosis' not in df.columns:
        print(f"Datos insuficientes para análisis APOE en {dataset_name}")
        return
    
    print(f"\nDistribución de genotipos APOE en {dataset_name}:")
    apoe_counts = df['APOE_genotype'].value_counts()
    display(apoe_counts)
    
    # Visualización de distribución de genotipos APOE
    plt.figure(figsize=(10, 6))
    sns.countplot(x='APOE_genotype', data=df, order=apoe_counts.index)
    plt.title(f'Distribución de genotipos APOE en {dataset_name}')
    plt.xlabel('Genotipo APOE')
    plt.ylabel('Número de sujetos')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Distribución de APOE4 (número de alelos)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='APOE4_count', data=df)
    plt.title(f'Distribución de alelos APOE4 en {dataset_name}')
    plt.xlabel('Número de alelos APOE4')
    plt.ylabel('Número de sujetos')
    plt.xticks([0, 1, 2])
    plt.tight_layout()
    plt.show()
    
    # Relación entre APOE4 y diagnóstico
    plt.figure(figsize=(12, 7))
    sns.countplot(x='APOE4_count', hue='diagnosis', data=df)
    plt.title(f'Relación entre alelos APOE4 y diagnóstico en {dataset_name}')
    plt.xlabel('Número de alelos APOE4')
    plt.ylabel('Número de sujetos')
    plt.xticks([0, 1, 2])
    plt.legend(title='Diagnóstico')
    plt.tight_layout()
    plt.show()
    
    # Tabla de contingencia y chi-cuadrado
    contingency = pd.crosstab(df['APOE4_count'], df['diagnosis'])
    display(contingency)
    
    # Prueba chi-cuadrado para asociación entre APOE4 y diagnóstico
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"Prueba Chi-cuadrado para asociación APOE4-diagnóstico:")
    print(f"Chi2 = {chi2:.2f}, p-valor = {p:.4f}, grados de libertad = {dof}")
    if p < 0.05:
        print("Existe una asociación estadísticamente significativa entre APOE4 y diagnóstico")
    else:
        print("No hay asociación estadísticamente significativa entre APOE4 y diagnóstico")
    
    # Porcentaje de diagnóstico por número de alelos APOE4
    apoe_diag_pct = pd.crosstab(df['APOE4_count'], df['diagnosis'], normalize='index') * 100
    
    plt.figure(figsize=(12, 7))
    apoe_diag_pct.plot(kind='bar', stacked=True)
    plt.title(f'Distribución porcentual de diagnósticos por alelos APOE4 en {dataset_name}')
    plt.xlabel('Número de alelos APOE4')
    plt.ylabel('Porcentaje (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Diagnóstico')
    plt.tight_layout()
    plt.show()

# Analizar datos APOE en ADNI y OASIS
analyze_apoe_distribution(adni_genetics_df, 'ADNI')
analyze_apoe_distribution(oasis_genetics_df, 'OASIS')

# 4. Análisis de SNPs y riesgo genético
# ===================================================================

print("\n--- Análisis de SNPs y riesgo genético ---")

def analyze_snps(df, dataset_name):
    """Analiza la distribución y asociaciones de SNPs de riesgo"""
    if df is None:
        print(f"Datos insuficientes para análisis SNP en {dataset_name}")
        return
    
    # Identificar columnas que podrían ser SNPs (rs*)
    snp_cols = [col for col in df.columns if col.startswith('rs')]
    
    if len(snp_cols) == 0:
        print(f"No se encontraron columnas de SNPs en {dataset_name}")
        return
    
    print(f"\nSNPs encontrados en {dataset_name}: {len(snp_cols)}")
    print(f"Ejemplos: {snp_cols[:5]}")
    
    # Analizar los primeros 5 SNPs o todos si hay menos
    snps_to_analyze = snp_cols[:min(5, len(snp_cols))]
    
    for snp in snps_to_analyze:
        print(f"\nAnálisis del SNP {snp}:")
        
        # Distribución de genotipos
        snp_counts = df[snp].value_counts().sort_index()
        display(snp_counts)
        
        plt.figure(figsize=(8, 5))
        sns.countplot(x=snp, data=df)
        plt.title(f'Distribución de genotipos para {snp} en {dataset_name}')
        plt.xlabel('Genotipo (0: AA, 1: AB, 2: BB)')
        plt.ylabel('Número de sujetos')
        plt.xticks([0, 1, 2])
        plt.tight_layout()
        plt.show()
        
        # Relación con diagnóstico si está disponible
        if 'diagnosis' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=snp, hue='diagnosis', data=df)
            plt.title(f'Relación entre {snp} y diagnóstico en {dataset_name}')
            plt.xlabel('Genotipo (0: AA, 1: AB, 2: BB)')
            plt.ylabel('Número de sujetos')
            plt.xticks([0, 1, 2])
            plt.legend(title='Diagnóstico')
            plt.tight_layout()
            plt.show()
            
            # Tabla de contingencia y chi-cuadrado
            contingency = pd.crosstab(df[snp], df['diagnosis'])
            display(contingency)
            
            # Prueba chi-cuadrado para asociación entre SNP y diagnóstico
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            print(f"Prueba Chi-cuadrado para asociación {snp}-diagnóstico:")
            print(f"Chi2 = {chi2:.2f}, p-valor = {p:.4f}, grados de libertad = {dof}")
            if p < 0.05:
                print(f"Existe una asociación estadísticamente significativa entre {snp} y diagnóstico")
            else:
                print(f"No hay asociación estadísticamente significativa entre {snp} y diagnóstico")
    
    # Análisis de Polygenic Risk Score si está disponible
    if 'PRS' in df.columns:
        print("\nAnálisis del Polygenic Risk Score (PRS):")
        
        # Estadísticas descriptivas del PRS
        prs_stats = df['PRS'].describe()
        display(prs_stats)
        
        # Distribución del PRS
        plt.figure(figsize=(10, 6))
        sns.histplot(df['PRS'], kde=True)
        plt.title(f'Distribución del Polygenic Risk Score en {dataset_name}')
        plt.xlabel('PRS')
        plt.ylabel('Frecuencia')
        plt.axvline(x=prs_stats['mean'], color='red', linestyle='--', label='Media')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # PRS por diagnóstico si está disponible
        if 'diagnosis' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='diagnosis', y='PRS', data=df)
            plt.title(f'PRS por grupos diagnósticos en {dataset_name}')
            plt.xlabel('Diagnóstico')
            plt.ylabel('Polygenic Risk Score')
            plt.tight_layout()
            plt.show()
            
            # ANOVA para diferencias de PRS entre grupos diagnósticos
            groups = [df[df['diagnosis'] == diag]['PRS'] for diag in df['diagnosis'].unique()]
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"ANOVA para diferencias de PRS entre grupos diagnósticos:")
            print(f"F = {f_stat:.2f}, p-valor = {p_val:.4f}")
            if p_val < 0.05:
                print("Existen diferencias estadísticamente significativas en el PRS entre grupos diagnósticos")
            else:
                print("No hay diferencias estadísticamente significativas en el PRS entre grupos diagnósticos")
            
        # PRS por número de alelos APOE4 si está disponible
        if 'APOE4_count' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='APOE4_count', y='PRS', data=df)
            plt.title(f'PRS por número de alelos APOE4 en {dataset_name}')
            plt.xlabel('Número de alelos APOE4')
            plt.ylabel('Polygenic Risk Score')
            plt.xticks([0, 1, 2])
            plt.tight_layout()
            plt.show()
            
            # ANOVA para diferencias de PRS entre grupos de APOE4
            groups = [df[df['APOE4_count'] == count]['PRS'] for count in df['APOE4_count'].unique()]
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"ANOVA para diferencias de PRS entre grupos de APOE4:")
            print(f"F = {f_stat:.2f}, p-valor = {p_val:.4f}")
            if p_val < 0.05:
                print("Existen diferencias estadísticamente significativas en el PRS entre grupos de APOE4")
            else:
                print("No hay diferencias estadísticamente significativas en el PRS entre grupos de APOE4")

# Analizar SNPs en ADNI y OASIS
analyze_snps(adni_genetics_df, 'ADNI')
analyze_snps(oasis_genetics_df, 'OASIS')

# 5. Análisis demográfico y genético
# ===================================================================

print("\n--- Análisis demográfico y genético ---")

def analyze_demographics_genetics(df, dataset_name):
    """Analiza la relación entre variables demográficas y genéticas"""
    if df is None or not {'age', 'sex'}.issubset(df.columns):
        print(f"Datos demográficos insuficientes para análisis en {dataset_name}")
        return
    
    print(f"\nDistribución demográfica en {dataset_name}:")
    
    # Distribución de edad
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True, bins=20)
    plt.title(f'Distribución de edad en {dataset_name}')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.axvline(x=df['age'].mean(), color='red', linestyle='--', label='Media')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Distribución por sexo
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sex', data=df)
    plt.title(f'Distribución por sexo en {dataset_name}')
    plt.xlabel('Sexo')
    plt.ylabel('Número de sujetos')
    plt.tight_layout()
    plt.show()
    
    # Edad por diagnóstico si está disponible
    if 'diagnosis' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='diagnosis', y='age', data=df)
        plt.title(f'Edad por grupos diagnósticos en {dataset_name}')
        plt.xlabel('Diagnóstico')
        plt.ylabel('Edad')
        plt.tight_layout()
        plt.show()
        
        # ANOVA para diferencias de edad entre grupos diagnósticos
        groups = [df[df['diagnosis'] == diag]['age'] for diag in df['diagnosis'].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"ANOVA para diferencias de edad entre grupos diagnósticos:")
        print(f"F = {f_stat:.2f}, p-valor = {p_val:.4f}")
        if p_val < 0.05:
            print("Existen diferencias estadísticamente significativas en la edad entre grupos diagnósticos")
        else:
            print("No hay diferencias estadísticamente significativas en la edad entre grupos diagnósticos")
    
    # APOE4 por sexo si está disponible
    if 'APOE4_count' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='APOE4_count', hue='sex', data=df)
        plt.title(f'Distribución de alelos APOE4 por sexo en {dataset_name}')
        plt.xlabel('Número de alelos APOE4')
        plt.ylabel('Número de sujetos')
        plt.xticks([0, 1, 2])
        plt.legend(title='Sexo')
        plt.tight_layout()
        plt.show()
        
        # Tabla de contingencia y chi-cuadrado
        contingency = pd.crosstab(df['APOE4_count'], df['sex'])
        display(contingency)
        
        # Prueba chi-cuadrado para asociación entre APOE4 y sexo
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Prueba Chi-cuadrado para asociación APOE4-sexo:")
        print(f"Chi2 = {chi2:.2f}, p-valor = {p:.4f}, grados de libertad = {dof}")
        if p < 0.05:
            print("Existe una asociación estadísticamente significativa entre APOE4 y sexo")
        else:
            print("No hay asociación estadísticamente significativa entre APOE4 y sexo")
    
    # PRS por edad y sexo si está disponible
    if 'PRS' in df.columns:
        # Correlación entre PRS y edad
        corr, p_val = stats.pearsonr(df['age'], df['PRS'])
        print(f"Correlación entre PRS y edad: r = {corr:.3f}, p-valor = {p_val:.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='age', y='PRS', hue='sex', data=df)
        plt.title(f'Relación entre edad, PRS y sexo en {dataset_name}')
        plt.xlabel('Edad')
        plt.ylabel('Polygenic Risk Score')
        plt.legend(title='Sexo')
        plt.tight_layout()
        plt.show()
        
        # PRS por sexo
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='sex', y='PRS', data=df)
        plt.title(f'PRS por sexo en {dataset_name}')
        plt.xlabel('Sexo')
        plt.ylabel('Polygenic Risk Score')
        plt.tight_layout()
        plt.show()
        
        # T-test para diferencias de PRS entre sexos
        male_prs = df[df['sex'] == 'M']['PRS']
        female_prs = df[df['sex'] == 'F']['PRS']
        t_stat, p_val = stats.ttest_ind(male_prs, female_prs)
        print(f"T-test para diferencias de PRS entre sexos:")
        print(f"t = {t_stat:.2f}, p-valor = {p_val:.4f}")
        if p_val < 0.05:
            print("Existen diferencias estadísticamente significativas en el PRS entre sexos")
        else:
            print("No hay diferencias estadísticamente significativas en el PRS entre sexos")

# Analizar demografía y genética en ADNI y OASIS
analyze_demographics_genetics(adni_genetics_df, 'ADNI')
analyze_demographics_genetics(oasis_genetics_df, 'OASIS')

# 6. Correlaciones entre variables genéticas
# ===================================================================

print("\n--- Correlaciones entre variables genéticas ---")

def analyze_genetic_correlations(df, dataset_name):
    """Analiza las correlaciones entre variables genéticas"""
    if df is None:
        print(f"Datos insuficientes para análisis de correlaciones en {dataset_name}")
        return
    
    # Filtrar columnas numéricas que pueden representar SNPs o scores genéticos
    # Incluye APOE4_count si está disponible
    genetic_cols = [col for col in df.columns if col.startswith('rs') and df[col].dtype in ['int64', 'float64']]
    
    if 'APOE4_count' in df.columns:
        genetic_cols.append('AP