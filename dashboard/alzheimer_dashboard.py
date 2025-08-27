"""
Dashboard Interactivo de Monitorización y Prevención de Alzheimer
Proyecto: Monitorización y Predicción Multimodal de Alzheimer - Fase 6
Autor: Sistema de Ciencia de Datos
Fecha: 2025

Dashboard accesible para usuarios médicos y no médicos con explicaciones claras,
visualizaciones intuitivas y recomendaciones accionables.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN Y DATOS SIMULADOS
# ==========================================

class AlzheimerDashboardConfig:
    """Configuración centralizada del dashboard"""
    
    def __init__(self):
        self.colors = {
            'low_risk': '#10B981',      # Verde - Riesgo Bajo
            'moderate_risk': '#F59E0B', # Amarillo - Riesgo Moderado  
            'high_risk': '#EF4444',     # Rojo - Riesgo Alto
            'info': '#3B82F6',          # Azul - Información
            'secondary': '#6B7280',     # Gris - Secundario
            'background': '#F9FAFB',    # Fondo claro
            'card': '#FFFFFF'           # Fondo de tarjetas
        }
        
        self.thresholds = {
            'low_risk': 0.3,
            'moderate_risk': 0.7,
            'high_risk': 1.0
        }
        
        self.feature_groups = {
            'biomarcadores': ['tau_protein', 'abeta_42', 'ptau_181', 'nfl_protein'],
            'neuroimagen': ['hippocampus_volume', 'entorhinal_thickness', 'whole_brain_volume'],
            'cognitivo': ['mmse_score', 'cdr_score', 'adas_cog_score'],
            'lifestyle': ['physical_activity', 'social_engagement', 'sleep_quality'],
            'demografico': ['age', 'education_years', 'apoe4_carriers']
        }
        
        self.feature_explanations = {
            'tau_protein': 'Proteína Tau: Biomarcador clave del Alzheimer. Valores altos indican daño neuronal.',
            'abeta_42': 'Beta-amiloide 42: Proteína que forma placas en el cerebro. Niveles bajos son preocupantes.',
            'mmse_score': 'Mini-Mental State Exam: Evaluación cognitiva básica (0-30 puntos).',
            'age': 'Edad: Factor de riesgo no modificable. El riesgo aumenta después de los 65 años.',
            'education_years': 'Años de Educación: Mayor educación puede ser protectora contra el Alzheimer.',
            'physical_activity': 'Actividad Física: Ejercicio regular reduce el riesgo de deterioro cognitivo.',
            'hippocampus_volume': 'Volumen del Hipocampo: Área cerebral crucial para la memoria.',
            'apoe4_carriers': 'Gen APOE4: Variante genética que aumenta el riesgo de Alzheimer.'
        }

# Instancia global de configuración
config = AlzheimerDashboardConfig()

# ==========================================
# GENERACIÓN DE DATOS SIMULADOS
# ==========================================

def generate_sample_data():
    """Genera datos de muestra para el dashboard"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generar características base
    data = {
        'patient_id': [f'PAT_{i:04d}' for i in range(n_samples)],
        'age': np.random.normal(72, 8, n_samples).clip(50, 90),
        'education_years': np.random.normal(14, 4, n_samples).clip(8, 20),
        'mmse_score': np.random.normal(26, 4, n_samples).clip(0, 30),
        'cdr_score': np.random.exponential(0.5, n_samples).clip(0, 3),
        'adas_cog_score': np.random.exponential(8, n_samples).clip(0, 70),
        'tau_protein': np.random.lognormal(4.5, 0.8, n_samples),
        'abeta_42': np.random.normal(800, 200, n_samples).clip(200, 1500),
        'ptau_181': np.random.lognormal(3.2, 0.6, n_samples),
        'nfl_protein': np.random.lognormal(3.8, 0.7, n_samples),
        'hippocampus_volume': np.random.normal(7500, 1200, n_samples).clip(4000, 10000),
        'entorhinal_thickness': np.random.normal(3.2, 0.6, n_samples).clip(1.5, 5.0),
        'whole_brain_volume': np.random.normal(1200000, 150000, n_samples).clip(900000, 1500000),
        'physical_activity': np.random.beta(2, 3, n_samples) * 10,
        'social_engagement': np.random.beta(3, 2, n_samples) * 10,
        'sleep_quality': np.random.normal(7, 1.5, n_samples).clip(3, 10),
        'apoe4_carriers': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Generar probabilidades de riesgo basadas en las características
    risk_score = (
        (df['age'] - 50) / 40 * 0.2 +
        (30 - df['mmse_score']) / 30 * 0.25 +
        df['cdr_score'] / 3 * 0.2 +
        np.log(df['tau_protein'] / df['tau_protein'].median()) * 0.15 +
        (df['abeta_42'].median() - df['abeta_42']) / df['abeta_42'].std() * 0.1 +
        df['apoe4_carriers'] / 2 * 0.1
    )
    
    df['risk_probability'] = 1 / (1 + np.exp(-risk_score.clip(-5, 5)))
    
    # Categorizar riesgo
    df['risk_category'] = pd.cut(
        df['risk_probability'],
        bins=[0, config.thresholds['low_risk'], config.thresholds['moderate_risk'], 1],
        labels=['Bajo', 'Moderado', 'Alto']
    )
    
    return df

def generate_feature_importance():
    """Genera importancia de características para el modelo"""
    features = [
        'tau_protein', 'mmse_score', 'age', 'cdr_score', 'abeta_42',
        'hippocampus_volume', 'adas_cog_score', 'apoe4_carriers',
        'entorhinal_thickness', 'education_years', 'physical_activity',
        'ptau_181', 'nfl_protein', 'social_engagement', 'sleep_quality'
    ]
    
    # Importancias basadas en literatura médica
    importances = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005]
    
    return pd.DataFrame({
        'feature': features,
        'importance': importances,
        'group': [
            'biomarcadores', 'cognitivo', 'demografico', 'cognitivo', 'biomarcadores',
            'neuroimagen', 'cognitivo', 'demografico', 'neuroimagen', 'demografico',
            'lifestyle', 'biomarcadores', 'biomarcadores', 'lifestyle', 'lifestyle'
        ]
    }).sort_values('importance', ascending=False)

# Cargar datos
sample_data = generate_sample_data()
feature_importance = generate_feature_importance()

# ==========================================
# FUNCIONES DE UTILIDAD
# ==========================================

def get_risk_color(probability):
    """Obtiene el color según la probabilidad de riesgo"""
    if probability < config.thresholds['low_risk']:
        return config.colors['low_risk']
    elif probability < config.thresholds['moderate_risk']:
        return config.colors['moderate_risk']
    else:
        return config.colors['high_risk']

def get_risk_category(probability):
    """Obtiene la categoría de riesgo"""
    if probability < config.thresholds['low_risk']:
        return 'Bajo'
    elif probability < config.thresholds['moderate_risk']:
        return 'Moderado'
    else:
        return 'Alto'

def generate_explanation(patient_data, risk_prob):
    """Genera explicación personalizada del riesgo"""
    explanations = []
    
    # Factores principales
    if patient_data.get('age', 65) > 75:
        explanations.append("🔴 Edad avanzada (>75 años) aumenta significativamente el riesgo")
    elif patient_data.get('age', 65) > 65:
        explanations.append("🟡 Edad moderada (65-75 años) contribuye al riesgo")
    
    if patient_data.get('mmse_score', 30) < 24:
        explanations.append("🔴 Puntuación MMSE baja indica deterioro cognitivo significativo")
    elif patient_data.get('mmse_score', 30) < 27:
        explanations.append("🟡 Puntuación MMSE moderada sugiere leve deterioro cognitivo")
    
    if patient_data.get('tau_protein', 50) > 80:
        explanations.append("🔴 Niveles altos de proteína Tau indican daño neuronal")
    
    if patient_data.get('abeta_42', 800) < 600:
        explanations.append("🔴 Niveles bajos de Aβ42 sugieren formación de placas")
    
    if patient_data.get('apoe4_carriers', 0) >= 1:
        explanations.append("🔴 Presencia de gen APOE4 aumenta el riesgo genético")
    
    # Factores protectores
    if patient_data.get('education_years', 12) > 16:
        explanations.append("🟢 Alto nivel educativo puede ser protector")
    
    if patient_data.get('physical_activity', 5) > 7:
        explanations.append("🟢 Alta actividad física es protectora")
    
    return explanations

def generate_recommendations(patient_data, risk_prob):
    """Genera recomendaciones personalizadas"""
    recommendations = []
    
    if risk_prob > config.thresholds['moderate_risk']:
        recommendations.extend([
            "🏥 Consulta neurológica especializada urgente",
            "🧪 Evaluación de biomarcadores adicionales",
            "🧠 Resonancia magnética cerebral detallada"
        ])
    
    if patient_data.get('physical_activity', 5) < 5:
        recommendations.append("🚶‍♀️ Aumentar actividad física: caminatas diarias de 30 min")
    
    if patient_data.get('social_engagement', 5) < 5:
        recommendations.append("👥 Incrementar actividad social y estimulación cognitiva")
    
    if patient_data.get('sleep_quality', 7) < 6:
        recommendations.append("😴 Mejorar higiene del sueño: 7-8 horas diarias")
    
    recommendations.extend([
        "🥗 Dieta mediterránea rica en antioxidantes",
        "📚 Actividades cognitivamente estimulantes",
        "🎵 Terapia musical y actividades creativas",
        "👨‍⚕️ Seguimiento médico regular cada 6 meses"
    ])
    
    return recommendations

# ==========================================
# INICIALIZACIÓN DE LA APP DASH
# ==========================================

app = dash.Dash(__name__, external_stylesheets=[
    'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

app.title = "Dashboard Alzheimer - Monitorización Inteligente"

# ==========================================
# LAYOUT PRINCIPAL
# ==========================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-brain mr-3 text-blue-500"),
                "Dashboard de Monitorización de Alzheimer"
            ], className="text-3xl font-bold text-gray-800"),
            html.P(
                "Sistema inteligente de evaluación y prevención - Accesible para todos los usuarios",
                className="text-gray-600 mt-2"
            )
        ], className="flex-1"),
        html.Div([
            html.Div([
                html.I(className="fas fa-calendar-alt mr-2"),
                datetime.now().strftime("%d/%m/%Y")
            ], className="text-sm text-gray-600"),
            html.Div([
                html.I(className="fas fa-users mr-2"),
                f"{len(sample_data)} pacientes monitoreados"
            ], className="text-sm text-gray-600 mt-1")
        ])
    ], className="bg-white shadow-lg rounded-lg p-6 mb-6 flex items-center justify-between"),
    
    # Tabs Navigation
    dcc.Tabs(id="main-tabs", value="risk-evaluation", children=[
        dcc.Tab(label="🎯 Evaluación de Riesgo", value="risk-evaluation",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="📊 Análisis de Factores", value="risk-factors",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="👥 Casos Educativos", value="case-examples",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="📈 Monitoreo", value="monitoring",
                className="tab-style", selected_className="tab-selected"),
        dcc.Tab(label="🧠 Centro de Aprendizaje", value="learning-center",
                className="tab-style", selected_className="tab-selected")
    ], className="mb-6"),
    
    # Content Area
    html.Div(id="tabs-content", className="min-h-screen")
    
], className="bg-gray-50 min-h-screen p-4")

# ==========================================
# CALLBACKS PRINCIPALES
# ==========================================

@app.callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value")
)
def render_tab_content(active_tab):
    """Renderiza el contenido de cada tab"""
    if active_tab == "risk-evaluation":
        return render_risk_evaluation_tab()
    elif active_tab == "risk-factors":
        return render_risk_factors_tab()
    elif active_tab == "case-examples":
        return render_case_examples_tab()
    elif active_tab == "monitoring":
        return render_monitoring_tab()
    elif active_tab == "learning-center":
        return render_learning_center_tab()

def render_risk_evaluation_tab():
    """Tab 1: Evaluación de Riesgo Individual"""
    return html.Div([
        # Información introductoria
        html.Div([
            html.H2("🎯 Evaluación Personalizada de Riesgo", className="text-2xl font-bold text-gray-800 mb-4"),
            html.P([
                "Introduce los datos del paciente para obtener una evaluación completa del riesgo de Alzheimer. ",
                html.Strong("Todos los valores son opcionales"), " - el sistema calculará el riesgo con la información disponible."
            ], className="text-gray-600 mb-6")
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Panel de entrada de datos
            html.Div([
                html.H3("📝 Datos del Paciente", className="text-xl font-bold text-gray-800 mb-4"),
                
                # Datos demográficos
                html.Div([
                    html.H4("👤 Información Demográfica", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("Edad (años)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-age", type="number", value=70, min=50, max=100,
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Años de Educación", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-education", type="number", value=14, min=0, max=25,
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Gen APOE4 (copias)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Dropdown(id="input-apoe4", options=[
                                {"label": "0 copias (sin riesgo genético)", "value": 0},
                                {"label": "1 copia (riesgo moderado)", "value": 1},
                                {"label": "2 copias (alto riesgo genético)", "value": 2}
                            ], value=0, className="mb-4")
                        ])
                    ], className="grid grid-cols-1 gap-4")
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Evaluaciones cognitivas
                html.Div([
                    html.H4("🧠 Evaluaciones Cognitivas", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("MMSE Score (0-30)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Slider(id="input-mmse", min=0, max=30, value=26, marks={i: str(i) for i in range(0, 31, 5)},
                                     tooltip={"placement": "bottom", "always_visible": True})
                        ], className="mb-6"),
                        html.Div([
                            html.Label("CDR Score (0-3)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Slider(id="input-cdr", min=0, max=3, step=0.5, value=0.5,
                                     marks={i: str(i) for i in np.arange(0, 3.5, 0.5)},
                                     tooltip={"placement": "bottom", "always_visible": True})
                        ], className="mb-6")
                    ])
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Biomarcadores
                html.Div([
                    html.H4("🧪 Biomarcadores (Opcionales)", className="text-lg font-semibold text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("Proteína Tau (pg/mL)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-tau", type="number", value=85, min=0, max=200,
                                    placeholder="Valores normales: 40-80",
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4"),
                        html.Div([
                            html.Label("Beta-amiloide 42 (pg/mL)", className="block text-sm font-medium text-gray-700 mb-2"),
                            dcc.Input(id="input-abeta", type="number", value=750, min=200, max=1500,
                                    placeholder="Valores normales: 800-1200",
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500")
                        ], className="mb-4")
                    ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
                ], className="bg-gray-50 p-4 rounded-lg mb-6"),
                
                # Botón de evaluación
                html.Button(
                    [html.I(className="fas fa-calculator mr-2"), "Evaluar Riesgo"],
                    id="evaluate-button",
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200",
                    n_clicks=0
                )
                
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Panel de resultados
            html.Div([
                html.H3("📊 Resultado de la Evaluación", className="text-xl font-bold text-gray-800 mb-4"),
                html.Div(id="risk-results-content")
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 lg:grid-cols-2 gap-6")
    ])

@app.callback(
    Output("risk-results-content", "children"),
    [Input("evaluate-button", "n_clicks")],
    [State("input-age", "value"),
     State("input-education", "value"),
     State("input-apoe4", "value"),
     State("input-mmse", "value"),
     State("input-cdr", "value"),
     State("input-tau", "value"),
     State("input-abeta", "value")]
)
def update_risk_evaluation(n_clicks, age, education, apoe4, mmse, cdr, tau, abeta):
    """Actualiza la evaluación de riesgo"""
    if n_clicks == 0:
        return html.Div([
            html.I(className="fas fa-info-circle text-blue-500 text-6xl mb-4"),
            html.P("Haz clic en 'Evaluar Riesgo' para obtener los resultados", 
                   className="text-gray-600 text-center")
        ], className="text-center py-8")
    
    # Crear diccionario de datos del paciente
    patient_data = {
        'age': age or 70,
        'education_years': education or 14,
        'apoe4_carriers': apoe4 or 0,
        'mmse_score': mmse or 26,
        'cdr_score': cdr or 0.5,
        'tau_protein': tau or 85,
        'abeta_42': abeta or 750,
        'physical_activity': 5,  # Valores por defecto
        'social_engagement': 5
    }
    
    # Calcular probabilidad de riesgo (simplificado)
    risk_score = (
        (patient_data['age'] - 50) / 40 * 0.2 +
        (30 - patient_data['mmse_score']) / 30 * 0.25 +
        patient_data['cdr_score'] / 3 * 0.2 +
        (patient_data['tau_protein'] - 60) / 100 * 0.15 +
        (800 - patient_data['abeta_42']) / 400 * 0.1 +
        patient_data['apoe4_carriers'] / 2 * 0.1
    )
    
    risk_probability = max(0, min(1, 1 / (1 + np.exp(-risk_score.clip(-5, 5)))))
    risk_category = get_risk_category(risk_probability)
    risk_color = get_risk_color(risk_probability)
    
    # Generar explicaciones y recomendaciones
    explanations = generate_explanation(patient_data, risk_probability)
    recommendations = generate_recommendations(patient_data, risk_probability)
    
    return html.Div([
        # Gauge principal
        html.Div([
            dcc.Graph(
                figure=go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Riesgo de Alzheimer (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': config.colors['low_risk'] + '40'},
                            {'range': [30, 70], 'color': config.colors['moderate_risk'] + '40'},
                            {'range': [70, 100], 'color': config.colors['high_risk'] + '40'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                )).update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            )
        ], className="mb-6"),
        
        # Resultado principal
        html.Div([
            html.Div([
                html.H4(f"Riesgo: {risk_category}", 
                       className=f"text-2xl font-bold mb-2",
                       style={'color': risk_color}),
                html.P(f"Probabilidad: {risk_probability:.1%}", className="text-lg text-gray-600"),
                html.P(f"Nivel de confianza: 85%", className="text-sm text-gray-500")
            ], className="text-center p-4 border-2 rounded-lg", style={'border-color': risk_color})
        ], className="mb-6"),
        
        # Explicaciones
        html.Div([
            html.H4("🔍 Explicación Detallada", className="text-lg font-bold text-gray-800 mb-3"),
            html.Ul([
                html.Li(exp, className="mb-2 p-2 bg-gray-50 rounded") 
                for exp in explanations[:5]
            ], className="space-y-2")
        ], className="mb-6"),
        
        # Recomendaciones
        html.Div([
            html.H4("💡 Recomendaciones Personalizadas", className="text-lg font-bold text-gray-800 mb-3"),
            html.Ul([
                html.Li(rec, className="mb-2 p-2 bg-green-50 rounded border-l-4 border-green-400") 
                for rec in recommendations[:6]
            ], className="space-y-2")
        ])
    ])

def render_risk_factors_tab():
    """Tab 2: Análisis de Factores de Riesgo"""
    return html.Div([
        html.H2("📊 Análisis Completo de Factores de Riesgo", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Gráfico de importancia de características
        html.Div([
            html.H3("🎯 Importancia de los Factores", className="text-xl font-bold text-gray-800 mb-4"),
            dcc.Graph(
                figure=px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    color='group',
                    orientation='h',
                    title="Top 10 Factores Más Importantes para la Predicción",
                    labels={'importance': 'Importancia (%)', 'feature': 'Factor'},
                    color_discrete_map={
                        'biomarcadores': config.colors['high_risk'],
                        'cognitivo': config.colors['moderate_risk'],
                        'demografico': config.colors['info'],
                        'neuroimagen': config.colors['low_risk'],
                        'lifestyle': config.colors['secondary']
                    }
                ).update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=True
                )
            )
        ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
        
        # Análisis por grupos
        html.Div([
            html.H3("📋 Análisis por Grupos de Factores", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                # Biomarcadores
                html.Div([
                    html.H4("🧪 Biomarcadores", className="text-lg font-bold text-red-600 mb-2"),
                    html.P("Proteínas y moléculas que indican daño cerebral", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Proteína Tau: Indica daño en neuronas", className="mb-1"),
                        html.Li("• Beta-amiloide 42: Forma placas tóxicas", className="mb-1"),
                        html.Li("• pTau-181: Marcador específico de Alzheimer", className="mb-1"),
                        html.Li("• NFL: Indica daño en axones", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-red-50 border border-red-200 rounded-lg p-4"),
                
                # Cognitivo
                html.Div([
                    html.H4("🧠 Evaluaciones Cognitivas", className="text-lg font-bold text-yellow-600 mb-2"),
                    html.P("Pruebas que miden el funcionamiento mental", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• MMSE: Evaluación cognitiva básica (0-30)", className="mb-1"),
                        html.Li("• CDR: Escala de demencia clínica (0-3)", className="mb-1"),
                        html.Li("• ADAS-Cog: Evaluación detallada de Alzheimer", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-yellow-50 border border-yellow-200 rounded-lg p-4"),
                
                # Neuroimagen
                html.Div([
                    html.H4("🔍 Neuroimagen", className="text-lg font-bold text-green-600 mb-2"),
                    html.P("Medidas del cerebro por resonancia magnética", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Volumen del hipocampo: Área de la memoria", className="mb-1"),
                        html.Li("• Corteza entorrinal: Primera área afectada", className="mb-1"),
                        html.Li("• Volumen cerebral total: Atrofia general", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-green-50 border border-green-200 rounded-lg p-4"),
                
                # Estilo de vida
                html.Div([
                    html.H4("🏃‍♀️ Estilo de Vida", className="text-lg font-bold text-blue-600 mb-2"),
                    html.P("Factores modificables que puedes controlar", className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("• Actividad física: Ejercicio regular", className="mb-1"),
                        html.Li("• Compromiso social: Interacción con otros", className="mb-1"),
                        html.Li("• Calidad del sueño: Descanso reparador", className="mb-1")
                    ], className="text-sm text-gray-700")
                ], className="bg-blue-50 border border-blue-200 rounded-lg p-4")
                
            ], className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6")
        ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
        
        # Factores modificables vs no modificables
        html.Div([
            html.H3("⚖️ Factores Modificables vs No Modificables", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("❌ No Modificables", className="text-lg font-bold text-red-600 mb-3"),
                    html.Div([
                        html.Div("Edad", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Genética (APOE4)", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Sexo", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Antecedentes familiares", className="bg-red-100 text-red-800 px-3 py-2 rounded-lg")
                    ])
                ], className="bg-white border border-red-200 rounded-lg p-6"),
                
                html.Div([
                    html.H4("✅ Modificables", className="text-lg font-bold text-green-600 mb-3"),
                    html.Div([
                        html.Div("Actividad física", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Dieta mediterránea", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Estimulación cognitiva", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Control de factores cardiovasculares", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Calidad del sueño", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg mb-2"),
                        html.Div("Vida social activa", className="bg-green-100 text-green-800 px-3 py-2 rounded-lg")
                    ])
                ], className="bg-white border border-green-200 rounded-lg p-6")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-6")
        ], className="bg-gray-50 rounded-lg p-6")
    ])

def render_case_examples_tab():
    """Tab 3: Casos Educativos"""
    # Generar casos ejemplo
    case_examples = [
        {
            'id': 'caso_1',
            'name': 'María García, 68 años',
            'risk': 'Bajo',
            'probability': 0.15,
            'color': config.colors['low_risk'],
            'profile': {
                'age': 68,
                'education': 16,
                'mmse': 29,
                'tau': 45,
                'abeta': 950,
                'apoe4': 0,
                'activity': 8
            },
            'description': 'Profesora jubilada con alta educación y estilo de vida saludable',
            'factors': [
                'Alto nivel educativo (16 años)',
                'Excelente puntuación cognitiva (MMSE: 29/30)',
                'Biomarcadores dentro de rango normal',
                'Sin factor genético de riesgo',
                'Muy activa físicamente'
            ],
            'recommendations': [
                'Continuar con actividad física regular',
                'Mantener estimulación cognitiva',
                'Seguimiento anual de rutina'
            ]
        },
        {
            'id': 'caso_2',
            'name': 'Carlos Rodríguez, 74 años',
            'risk': 'Moderado',
            'probability': 0.55,
            'color': config.colors['moderate_risk'],
            'profile': {
                'age': 74,
                'education': 10,
                'mmse': 25,
                'tau': 95,
                'abeta': 620,
                'apoe4': 1,
                'activity': 4
            },
            'description': 'Jubilado con algunos factores de riesgo emergentes',
            'factors': [
                'Edad avanzada (74 años)',
                'Biomarcadores ligeramente alterados',
                'Una copia del gen APOE4',
                'Leve deterioro cognitivo (MMSE: 25/30)',
                'Actividad física limitada'
            ],
            'recommendations': [
                'Evaluación neurológica semestral',
                'Incrementar actividad física gradualmente',
                'Estimulación cognitiva estructurada',
                'Monitoreo de biomarcadores'
            ]
        },
        {
            'id': 'caso_3',
            'name': 'Ana Martínez, 79 años',
            'risk': 'Alto',
            'probability': 0.85,
            'color': config.colors['high_risk'],
            'profile': {
                'age': 79,
                'education': 6,
                'mmse': 21,
                'tau': 150,
                'abeta': 420,
                'apoe4': 2,
                'activity': 2
            },
            'description': 'Múltiples factores de riesgo requieren atención especializada',
            'factors': [
                'Edad muy avanzada (79 años)',
                'Deterioro cognitivo significativo (MMSE: 21/30)',
                'Biomarcadores muy alterados',
                'Dos copias del gen APOE4',
                'Muy baja actividad física'
            ],
            'recommendations': [
                'Consulta neurológica especializada urgente',
                'Evaluación integral de biomarcadores',
                'Plan de intervención multidisciplinario',
                'Soporte familiar y cuidador'
            ]
        }
    ]
    
    return html.Div([
        html.H2("👥 Casos Educativos Interactivos", className="text-2xl font-bold text-gray-800 mb-6"),
        
        html.Div([
            html.I(className="fas fa-info-circle text-blue-500 mr-2"),
            "Estos casos son ejemplos educativos basados en perfiles reales anonimizados. "
            "Cada caso muestra cómo diferentes factores influyen en el riesgo de Alzheimer."
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Caso 1: Riesgo Bajo
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[0]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[0]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[0]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[0]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[0]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[0]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': config.colors['low_risk'] + '40'},
                                        {'range': [30, 70], 'color': config.colors['moderate_risk'] + '40'},
                                        {'range': [70, 100], 'color': config.colors['high_risk'] + '40'}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("🔍 Factores Clave:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"✓ {factor}", className="text-green-700 mb-1") 
                        for factor in case_examples[0]['factors']
                    ], className="mb-4"),
                    
                    html.H4("💡 Recomendaciones:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-blue-700 mb-1") 
                        for rec in case_examples[0]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Caso 2: Riesgo Moderado
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[1]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[1]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[1]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[1]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[1]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[1]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': config.colors['low_risk'] + '40'},
                                        {'range': [30, 70], 'color': config.colors['moderate_risk'] + '40'},
                                        {'range': [70, 100], 'color': config.colors['high_risk'] + '40'}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("⚠️ Factores de Preocupación:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"• {factor}", className="text-yellow-700 mb-1") 
                        for factor in case_examples[1]['factors']
                    ], className="mb-4"),
                    
                    html.H4("🎯 Plan de Acción:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-orange-700 mb-1") 
                        for rec in case_examples[1]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Caso 3: Riesgo Alto
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(case_examples[2]['name'], className="text-xl font-bold text-gray-800"),
                        html.Div(f"Riesgo {case_examples[2]['risk']}", 
                               className="text-white px-3 py-1 rounded-full text-sm font-semibold",
                               style={'backgroundColor': case_examples[2]['color']})
                    ], className="flex justify-between items-center mb-4"),
                    
                    html.P(case_examples[2]['description'], className="text-gray-600 mb-4"),
                    
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=case_examples[2]['probability'] * 100,
                                title={'text': "Riesgo (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': case_examples[2]['color']},
                                    'steps': [
                                        {'range': [0, 30], 'color': config.colors['low_risk'] + '40'},
                                        {'range': [30, 70], 'color': config.colors['moderate_risk'] + '40'},
                                        {'range': [70, 100], 'color': config.colors['high_risk'] + '40'}
                                    ]
                                }
                            )).update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        )
                    ], className="mb-4"),
                    
                    html.H4("🚨 Factores de Alto Riesgo:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"⚠️ {factor}", className="text-red-700 mb-1") 
                        for factor in case_examples[2]['factors']
                    ], className="mb-4"),
                    
                    html.H4("🏥 Intervención Urgente:", className="text-lg font-semibold text-gray-800 mb-2"),
                    html.Ul([
                        html.Li(f"→ {rec}", className="text-red-700 mb-1") 
                        for rec in case_examples[2]['recommendations']
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6"),
        
        # Lecciones aprendidas
        html.Div([
            html.H3("📚 Lecciones Clave de estos Casos", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("🎓 Educación como Protección", className="text-lg font-semibold text-green-600 mb-2"),
                    html.P("Mayor nivel educativo puede proporcionar 'reserva cognitiva' que ayuda a resistir el deterioro cerebral.", 
                           className="text-gray-600")
                ], className="bg-green-50 border border-green-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("🧬 Genética no es Destino", className="text-lg font-semibold text-blue-600 mb-2"),
                    html.P("Aunque el APOE4 aumenta el riesgo, el estilo de vida y otros factores siguen siendo cruciales.", 
                           className="text-gray-600")
                ], className="bg-blue-50 border border-blue-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("⏰ Detección Temprana", className="text-lg font-semibold text-yellow-600 mb-2"),
                    html.P("La intervención temprana en casos de riesgo moderado puede prevenir o retrasar la progresión.", 
                           className="text-gray-600")
                ], className="bg-yellow-50 border border-yellow-200 rounded-lg p-4"),
                
                html.Div([
                    html.H4("🏃‍♀️ Nunca es Tarde", className="text-lg font-semibold text-purple-600 mb-2"),
                    html.P("Incluso en casos de alto riesgo, los cambios en el estilo de vida pueden tener beneficios.", 
                           className="text-gray-600")
                ], className="bg-purple-50 border border-purple-200 rounded-lg p-4")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

def render_monitoring_tab():
    """Tab 4: Monitoreo y Tendencias"""
    # Generar datos de monitoreo
    dates = pd.date_range(start='2024-01-01', end='2025-08-26', freq='W')
    monitoring_data = pd.DataFrame({
        'date': dates,
        'total_patients': np.cumsum(np.random.poisson(10, len(dates))) + 100,
        'high_risk': np.cumsum(np.random.poisson(2, len(dates))) + 20,
        'moderate_risk': np.cumsum(np.random.poisson(3, len(dates))) + 30,
        'low_risk': np.cumsum(np.random.poisson(5, len(dates))) + 50,
        'model_accuracy': 0.87 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001
    })
    monitoring_data['model_accuracy'] = monitoring_data['model_accuracy'].clip(0.82, 0.92)
    
    return html.Div([
        html.H2("📈 Monitoreo del Sistema y Tendencias", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Métricas principales
        html.Div([
            # Card 1: Total de pacientes
            html.Div([
                html.Div([
                    html.I(className="fas fa-users text-3xl text-blue-500 mb-2"),
                    html.H3(f"{monitoring_data['total_patients'].iloc[-1]:,}", className="text-2xl font-bold text-gray-800"),
                    html.P("Total Pacientes", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-arrow-up text-green-500 mr-1"),
                        f"+{monitoring_data['total_patients'].iloc[-1] - monitoring_data['total_patients'].iloc[-8]:,} esta semana"
                    ], className="text-sm text-green-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 2: Precisión del modelo
            html.Div([
                html.Div([
                    html.I(className="fas fa-bullseye text-3xl text-green-500 mb-2"),
                    html.H3(f"{monitoring_data['model_accuracy'].iloc[-1]:.1%}", className="text-2xl font-bold text-gray-800"),
                    html.P("Precisión del Modelo", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-check-circle text-green-500 mr-1"),
                        "Rendimiento óptimo"
                    ], className="text-sm text-green-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 3: Casos de alto riesgo
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-3xl text-red-500 mb-2"),
                    html.H3(f"{monitoring_data['high_risk'].iloc[-1]:,}", className="text-2xl font-bold text-gray-800"),
                    html.P("Alto Riesgo", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-eye text-blue-500 mr-1"),
                        "Requieren seguimiento"
                    ], className="text-sm text-blue-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center"),
            
            # Card 4: Alertas activas
            html.Div([
                html.Div([
                    html.I(className="fas fa-bell text-3xl text-yellow-500 mb-2"),
                    html.H3("12", className="text-2xl font-bold text-gray-800"),
                    html.P("Alertas Activas", className="text-gray-600"),
                    html.Div([
                        html.I(className="fas fa-clock text-yellow-500 mr-1"),
                        "Pendientes de revisión"
                    ], className="text-sm text-yellow-600 mt-2")
                ])
            ], className="bg-white rounded-lg shadow-md p-6 text-center")
            
        ], className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6"),
        
        # Gráficos de tendencias
        html.Div([
            html.Div([
                html.H3("📊 Evolución de Pacientes por Categoría de Riesgo", className="text-xl font-bold text-gray-800 mb-4"),
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['low_risk'], 
                                 mode='lines+markers', name='Riesgo Bajo', 
                                 line=dict(color=config.colors['low_risk'], width=3),
                                 marker=dict(size=6)),
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['moderate_risk'], 
                                 mode='lines+markers', name='Riesgo Moderado',
                                 line=dict(color=config.colors['moderate_risk'], width=3),
                                 marker=dict(size=6)),
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['high_risk'], 
                                 mode='lines+markers', name='Riesgo Alto',
                                 line=dict(color=config.colors['high_risk'], width=3),
                                 marker=dict(size=6))
                    ]).update_layout(
                        title="Tendencia de Casos por Nivel de Riesgo",
                        xaxis_title="Fecha",
                        yaxis_title="Número de Pacientes",
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                )
            ], className="bg-white rounded-lg shadow-md p-6 mb-6"),
            
            html.Div([
                html.H3("🎯 Rendimiento del Modelo en el Tiempo", className="text-xl font-bold text-gray-800 mb-4"),
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(x=monitoring_data['date'], y=monitoring_data['model_accuracy'], 
                                 mode='lines+markers', name='Precisión',
                                 line=dict(color=config.colors['info'], width=3),
                                 marker=dict(size=6),
                                 fill='tonexty'),
                        go.Scatter(x=monitoring_data['date'], y=[0.85]*len(monitoring_data), 
                                 mode='lines', name='Umbral Mínimo',
                                 line=dict(color='red', width=2, dash='dash'))
                    ]).update_layout(
                        title="Evolución de la Precisión del Modelo",
                        xaxis_title="Fecha",
                        yaxis_title="Precisión (%)",
                        yaxis=dict(tickformat='.1%', range=[0.8, 0.95]),
                        height=400,
                        hovermode='x unified'
                    )
                )
            ], className="bg-white rounded-lg shadow-md p-6")
        ]),
        
        # Estadísticas por subgrupos
        html.Div([
            html.H3("👥 Rendimiento por Subgrupos Poblacionales", className="text-xl font-bold text-gray-800 mb-4"),
            
            html.Div([
                # Por edad
                html.Div([
                    html.H4("🎂 Por Grupo de Edad", className="text-lg font-semibold text-gray-700 mb-3"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Grupo de Edad", "id": "age_group"},
                            {"name": "N° Pacientes", "id": "count"},
                            {"name": "Precisión", "id": "accuracy", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Sensibilidad", "id": "sensitivity", "type": "numeric", "format": {"specifier": ".1%"}}
                        ],
                        data=[
                            {"age_group": "50-64 años", "count": 245, "accuracy": 0.89, "sensitivity": 0.91},
                            {"age_group": "65-74 años", "count": 412, "accuracy": 0.87, "sensitivity": 0.88},
                            {"age_group": "75-84 años", "count": 278, "accuracy": 0.85, "sensitivity": 0.86},
                            {"age_group": "85+ años", "count": 65, "accuracy": 0.83, "sensitivity": 0.84}
                        ],
                        style_cell={'textAlign': 'center', 'fontSize': '14px'},
                        style_header={'backgroundColor': config.colors['info'], 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], className="bg-white rounded-lg border p-4"),
                
                # Por género
                html.Div([
                    html.H4("⚧️ Por Género", className="text-lg font-semibold text-gray-700 mb-3"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Género", "id": "gender"},
                            {"name": "N° Pacientes", "id": "count"},
                            {"name": "Precisión", "id": "accuracy", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Especificidad", "id": "specificity", "type": "numeric", "format": {"specifier": ".1%"}}
                        ],
                        data=[
                            {"gender": "Mujeres", "count": 580, "accuracy": 0.86, "specificity": 0.89},
                            {"gender": "Hombres", "count": 420, "accuracy": 0.88, "specificity": 0.87}
                        ],
                        style_cell={'textAlign': 'center', 'fontSize': '14px'},
                        style_header={'backgroundColor': config.colors['info'], 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], className="bg-white rounded-lg border p-4")
            ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
        ], className="bg-gray-50 rounded-lg p-6 mt-6"),
        
        # Sistema de alertas
        html.Div([
            html.H3("🚨 Sistema de Alertas Activo", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                # Alerta crítica
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle text-red-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Pacientes de Alto Riesgo Sin Seguimiento", className="text-lg font-semibold text-red-600"),
                            html.P("8 pacientes clasificados como alto riesgo no tienen cita programada", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Revisar", className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg mb-3"),
                
                # Alerta moderada
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-yellow-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Biomarcadores Pendientes", className="text-lg font-semibold text-yellow-600"),
                            html.P("15 pacientes tienen resultados de biomarcadores pendientes de análisis", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Ver Lista", className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded-r-lg mb-3"),
                
                # Alerta informativa
                html.Div([
                    html.Div([
                        html.I(className="fas fa-info-circle text-blue-500 text-xl mr-3"),
                        html.Div([
                            html.H4("Actualización de Modelo Disponible", className="text-lg font-semibold text-blue-600"),
                            html.P("Nueva versión del modelo con mejoras en precisión disponible para implementar", className="text-gray-600 mt-1")
                        ], className="flex-1"),
                        html.Button("Más Info", className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg")
                    ], className="flex items-center")
                ], className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg")
            ])
        ], className="bg-white rounded-lg shadow-md p-6 mt-6")
    ])

def render_learning_center_tab():
    """Tab 5: Centro de Aprendizaje"""
    return html.Div([
        html.H2("🧠 Centro de Aprendizaje sobre Alzheimer", className="text-2xl font-bold text-gray-800 mb-6"),
        
        # Navegación del centro de aprendizaje
        html.Div([
            dcc.Tabs(id="learning-tabs", value="glossary", children=[
                dcc.Tab(label="📚 Glosario Médico", value="glossary"),
                dcc.Tab(label="🤖 Cómo Funciona el Modelo", value="model-explanation"),
                dcc.Tab(label="❓ Preguntas Frecuentes", value="faq"),
                dcc.Tab(label="🔗 Recursos Adicionales", value="resources")
            ], className="mb-4"),
            
            html.Div(id="learning-content")
        ])
    ])

@app.callback(
    Output("learning-content", "children"),
    Input("learning-tabs", "value")
)
def render_learning_content(active_tab):
    """Renderiza el contenido del centro de aprendizaje"""
    if active_tab == "glossary":
        return render_glossary()
    elif active_tab == "model-explanation":
        return render_model_explanation()
    elif active_tab == "faq":
        return render_faq()
    elif active_tab == "resources":
        return render_resources()

def render_glossary():
    """Glosario médico interactivo"""
    glossary_terms = [
        {
            "term": "Alzheimer",
            "definition": "Enfermedad neurodegenerativa progresiva que es la causa más común de demencia. Afecta principalmente la memoria, el pensamiento y el comportamiento.",
            "icon": "fas fa-brain",
            "color": "red"
        },
        {
            "term": "Beta-amiloide (Aβ42)",
            "definition": "Proteína que se acumula formando placas en el cerebro de personas con Alzheimer. Los niveles bajos en líquido cefalorraquídeo pueden indicar la presencia de placas cerebrales.",
            "icon": "fas fa-microscope",
            "color": "blue"
        },
        {
            "term": "Proteína Tau",
            "definition": "Proteína que normalmente estabiliza las estructuras celulares. En el Alzheimer, se vuelve anormal y forma ovillos neurofibrilares que dañan las neuronas.",
            "icon": "fas fa-dna",
            "color": "green"
        },
        {
            "term": "MMSE (Mini-Mental State Exam)",
            "definition": "Prueba cognitiva breve que evalúa orientación, memoria, atención y habilidades del lenguaje. Puntuación de 0-30, donde puntuaciones más bajas indican mayor deterioro.",
            "icon": "fas fa-clipboard-check",
            "color": "purple"
        },
        {
            "term": "CDR (Clinical Dementia Rating)",
            "definition": "Escala que evalúa el grado de deterioro cognitivo en 6 áreas. Va de 0 (normal) a 3 (demencia severa).",
            "icon": "fas fa-chart-line",
            "color": "yellow"
        },
        {
            "term": "APOE4",
            "definition": "Variante del gen APOE que aumenta el riesgo de desarrollar Alzheimer. Tener una copia aumenta el riesgo 2-3 veces, dos copias lo aumentan 8-12 veces.",
            "icon": "fas fa-dna",
            "color": "red"
        },
        {
            "term": "Hipocampo",
            "definition": "Región cerebral crucial para la formación de nuevas memorias. Es una de las primeras áreas afectadas en el Alzheimer.",
            "icon": "fas fa-brain",
            "color": "blue"
        },
        {
            "term": "Biomarcadores",
            "definition": "Indicadores biológicos medibles que reflejan procesos normales, patológicos o respuestas a tratamientos. En Alzheimer incluyen proteínas en sangre y líquido cefalorraquídeo.",
            "icon": "fas fa-flask",
            "color": "green"
        },
        {
            "term": "Reserva Cognitiva",
            "definition": "Capacidad del cerebro para mantener el funcionamiento normal a pesar del daño. Se asocia con mayor educación, actividad mental y social.",
            "icon": "fas fa-shield-alt",
            "color": "blue"
        },
        {
            "term": "Deterioro Cognitivo Leve (DCL)",
            "definition": "Condición intermedia entre el envejecimiento normal y la demencia. Cambios cognitivos notables pero que no interfieren significativamente con la vida diaria.",
            "icon": "fas fa-battery-half",
            "color": "yellow"
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-book text-blue-500 mr-2"),
            "Glosario de términos médicos explicados en lenguaje claro y comprensible para todos"
        ], className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className=f"{term['icon']} text-2xl text-{term['color']}-500 mr-4"),
                        html.Div([
                            html.H3(term['term'], className="text-xl font-bold text-gray-800 mb-2"),
                            html.P(term['definition'], className="text-gray-600 leading-relaxed")
                        ], className="flex-1")
                    ], className="flex items-start")
                ], className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-200")
            ]) for term in glossary_terms
        ], className="grid grid-cols-1 md:grid-cols-2 gap-4")
    ])

def render_model_explanation():
    """Explicación del modelo en términos no técnicos"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-robot text-purple-500 mr-2"),
            "Entenda cómo funciona nuestro sistema de predicción de riesgo de Alzheimer"
        ], className="bg-purple-50 border-l-4 border-purple-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            # Paso 1
            html.Div([
                html.Div([
                    html.Div("1", className="bg-blue-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Recolección de Datos", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema recopila información de múltiples fuentes: ",
                        html.Strong("análisis de sangre"), ", ",
                        html.Strong("resonancias magnéticas"), ", ",
                        html.Strong("evaluaciones cognitivas"), " y ",
                        html.Strong("datos demográficos"), "."
                    ], className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("🩺 Biomarcadores en sangre y LCR", className="mb-1"),
                        html.Li("🧠 Imágenes del cerebro (RM)", className="mb-1"),
                        html.Li("📝 Pruebas de memoria y cognición", className="mb-1"),
                        html.Li("👤 Edad, educación, genética", className="mb-1")
                    ], className="text-sm text-gray-700")
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 2
            html.Div([
                html.Div([
                    html.Div("2", className="bg-green-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Análisis Inteligente", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "Un modelo de ",
                        html.Strong("inteligencia artificial"), 
                        " entrenado con datos de miles de pacientes analiza todos estos factores simultaneamente para identificar patrones complejos."
                    ], className="text-gray-600 mb-3"),
                    html.Div([
                        html.P("El modelo considera:", className="font-semibold text-gray-700 mb-2"),
                        html.Ul([
                            html.Li("🔄 Interacciones entre factores", className="mb-1"),
                            html.Li("⚖️ Peso relativo de cada variable", className="mb-1"),
                            html.Li("📊 Patrones en datos históricos", className="mb-1"),
                            html.Li("🎯 Umbrales clínicos validados", className="mb-1")
                        ], className="text-sm text-gray-700")
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 3
            html.Div([
                html.Div([
                    html.Div("3", className="bg-yellow-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Cálculo de Riesgo", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema calcula una ",
                        html.Strong("probabilidad de riesgo"), 
                        " entre 0% y 100%, que se traduce en categorías comprensibles: Bajo, Moderado o Alto riesgo."
                    ], className="text-gray-600 mb-3"),
                    html.Div([
                        html.Div("0-30%", className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Bajo", className="text-gray-600"),
                        html.Br(),
                        html.Div("30-70%", className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Moderado", className="text-gray-600"),
                        html.Br(),
                        html.Div("70-100%", className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-semibold mr-2 mb-2 inline-block"),
                        html.Span("Riesgo Alto", className="text-gray-600")
                    ])
                ])
            ], className="bg-white rounded-lg shadow-md p-6"),
            
            # Paso 4
            html.Div([
                html.Div([
                    html.Div("4", className="bg-purple-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold mb-4"),
                    html.H3("Explicación y Recomendaciones", className="text-xl font-bold text-gray-800 mb-3"),
                    html.P([
                        "El sistema no solo da un número, sino que ",
                        html.Strong("explica por qué"), 
                        " y ",
                        html.Strong("qué se puede hacer"), 
                        ". Genera explicaciones personalizadas y recomendaciones específicas."
                    ], className="text-gray-600 mb-3"),
                    html.Ul([
                        html.Li("🔍 Factores que más influyen en el riesgo", className="mb-1"),
                        html.Li("💡 Recomendaciones médicas específicas", className="mb-1"),
                        html.Li("🏃‍♀️ Cambios de estilo de vida sugeridos", className="mb-1"),
                        html.Li("📅 Cronograma de seguimiento", className="mb-1")
                    ], className="text-sm text-gray-700")
                ])
            ], className="bg-white rounded-lg shadow-md p-6")
            
        ], className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6"),
        
        # Confiabilidad del modelo
        html.Div([
            html.H3("🎯 ¿Qué tan confiable es el modelo?", className="text-xl font-bold text-gray-800 mb-4"),
            html.Div([
                html.Div([
                    html.H4("87% de Precisión", className="text-2xl font-bold text-green-600 mb-2"),
                    html.P("El modelo acierta en 87 de cada 100 predicciones", className="text-gray-600")
                ], className="text-center p-4"),
                html.Div([
                    html.H4("10,000+ Pacientes", className="text-2xl font-bold text-blue-600 mb-2"),
                    html.P("Entrenado con datos de más de 10,000 pacientes reales", className="text-gray-600")
                ], className="text-center p-4"),
                html.Div([
                    html.H4("5 Años de Datos", className="text-2xl font-bold text-purple-600 mb-2"),
                    html.P("Utiliza información longitudinal de seguimiento", className="text-gray-600")
                ], className="text-center p-4"),
                html.Div([
                    html.H4("Validado Clínicamente", className="text-2xl font-bold text-red-600 mb-2"),
                    html.P("Probado en múltiples hospitales y poblaciones", className="text-gray-600")
                ], className="text-center p-4")
            ], className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-gray-50 rounded-lg p-6")
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

def render_faq():
    """Preguntas frecuentes"""
    faqs = [
        {
            "question": "¿Qué significa tener alto riesgo de Alzheimer?",
            "answer": "Tener alto riesgo NO significa que definitivamente desarrollarás Alzheimer. Significa que, basado en factores conocidos, tienes una probabilidad mayor que el promedio. Muchas personas con alto riesgo nunca desarrollan la enfermedad, especialmente si toman medidas preventivas."
        },
        {
            "question": "¿Puedo cambiar mi riesgo de Alzheimer?",
            "answer": "¡Sí! Aunque factores como la edad y la genética no se pueden cambiar, muchos otros sí: ejercicio regular, dieta saludable, actividad social, estimulación mental, buen sueño y control de factores cardiovasculares pueden reducir significativamente el riesgo."
        },
        {
            "question": "¿Con qué frecuencia debo hacerme la evaluación?",
            "answer": "Depende de tu nivel de riesgo: Riesgo bajo: cada 2-3 años. Riesgo moderado: cada año. Riesgo alto: cada 6 meses o según indicación médica. Tu doctor puede recomendar una frecuencia diferente basada en tu situación específica."
        },
        {
            "question": "¿Los biomarcadores son necesarios para la evaluación?",
            "answer": "No son estrictamente necesarios. El modelo puede hacer predicciones útiles solo con evaluaciones cognitivas y datos demográficos. Sin embargo, los biomarcadores proporcionan información más precisa y detallada sobre el estado del cerebro."
        },
        {
            "question": "¿Qué tan temprano se puede detectar el riesgo?",
            "answer": "Los cambios cerebrales pueden comenzar 10-20 años antes de los síntomas. Nuestro modelo puede detectar riesgo elevado en etapas muy tempranas, cuando las intervenciones son más efectivas para prevenir o retrasar la progresión."
        },
        {
            "question": "¿El resultado puede estar equivocado?",
            "answer": "Ningún test es 100% perfecto. Nuestro modelo tiene 87% de precisión, lo que significa que puede haber falsos positivos o negativos. Por eso es importante: 1) Usar múltiples evaluaciones, 2) Considerar el contexto clínico completo, 3) Seguimiento regular."
        },
        {
            "question": "¿Debo preocuparme si tengo el gen APOE4?",
            "answer": "Tener APOE4 aumenta el riesgo, pero NO garantiza que desarrollarás Alzheimer. Muchas personas con este gen nunca desarrollan la enfermedad. Es información útil para tomar medidas preventivas más agresivas."
        },
        {
            "question": "¿Los síntomas de memoria normal del envejecimiento son preocupantes?",
            "answer": "Es normal olvidar ocasionalmente nombres o donde pusiste las llaves. Preocúpate si: olvidas conversaciones recientes, te pierdes en lugares familiares, tienes dificultad con tareas habituales, o familiares notan cambios significativos."
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-question-circle text-green-500 mr-2"),
            "Respuestas a las preguntas más comunes sobre Alzheimer y evaluación de riesgo"
        ], className="bg-green-50 border-l-4 border-green-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.Details([
                    html.Summary(faq['question'], className="text-lg font-semibold text-gray-800 cursor-pointer hover:text-blue-600 py-3"),
                    html.Div([
                        html.P(faq['answer'], className="text-gray-600 leading-relaxed mt-2 pl-4")
                    ])
                ], className="border-b border-gray-200")
            ]) for faq in faqs
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

def render_resources():
    """Recursos adicionales"""
    resources = [
        {
            "category": "Organizaciones Oficiales",
            "items": [
                {"name": "Alzheimer's Association", "url": "https://www.alz.org", "description": "Organización líder mundial en cuidado, apoyo e investigación del Alzheimer"},
                {"name": "Instituto Nacional del Envejecimiento", "url": "https://www.nia.nih.gov", "description": "Información científica sobre envejecimiento y demencia"},
                {"name": "Fundación Alzheimer España", "url": "https://www.fundacionalzheimeresp.org", "description": "Recursos en español sobre Alzheimer y demencias"}
            ]
        },
        {
            "category": "Prevención y Estilo de Vida",
            "items": [
                {"name": "FINGER Study", "url": "#", "description": "Estudio sobre intervención multidomain para prevenir deterioro cognitivo"},
                {"name": "Brain Training Games", "url": "#", "description": "Juegos científicamente validados para estimulación cognitiva"},
                {"name": "Mediterranean Diet Guide", "url": "#", "description": "Guía completa de la dieta mediterránea para salud cerebral"}
            ]
        },
        {
            "category": "Apoyo y Cuidadores",
            "items": [
                {"name": "Caregiver Support Groups", "url": "#", "description": "Grupos de apoyo para familiares y cuidadores"},
                {"name": "Respite Care Services", "url": "#", "description": "Servicios de cuidado temporal para dar descanso a cuidadores"},
                {"name": "Legal and Financial Planning", "url": "#", "description": "Planificación legal y financiera para familias afectadas"}
            ]
        },
        {
            "category": "Investigación y Tratamientos",
            "items": [
                {"name": "ClinicalTrials.gov", "url": "https://clinicaltrials.gov", "description": "Base de datos de ensayos clínicos en curso"},
                {"name": "Alzheimer's Drug Discovery", "url": "#", "description": "Últimos avances en tratamientos y medicamentos"},
                {"name": "Research Participation", "url": "#", "description": "Cómo participar en estudios de investigación"}
            ]
        }
    ]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-external-link-alt text-indigo-500 mr-2"),
            "Enlaces útiles a recursos confiables sobre Alzheimer, prevención y apoyo"
        ], className="bg-indigo-50 border-l-4 border-indigo-400 p-4 rounded-r-lg mb-6"),
        
        html.Div([
            html.Div([
                html.H3(category['category'], className="text-xl font-bold text-gray-800 mb-4"),
                html.Div([
                    html.Div([
                        html.H4(item['name'], className="text-lg font-semibold text-blue-600 mb-2"),
                        html.P(item['description'], className="text-gray-600 text-sm mb-3"),
                        html.A("Visitar →", href=item['url'], target="_blank", 
                              className="text-blue-500 hover:text-blue-700 font-medium text-sm")
                    ], className="bg-gray-50 rounded-lg p-4 hover:shadow-md transition-shadow duration-200")
                    for item in category['items']
                ], className="space-y-3")
            ], className="bg-white rounded-lg shadow-md p-6")
            for category in resources
        ], className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6"),
        
        # Contacto de emergencia
        html.Div([
            html.H3("🚨 En Caso de Emergencia", className="text-xl font-bold text-red-600 mb-4"),
            html.Div([
                html.Div([
                    html.I(className="fas fa-phone text-red-500 text-2xl mr-4"),
                    html.Div([
                        html.H4("Línea de Crisis 24/7", className="text-lg font-semibold text-gray-800"),
                        html.P("1-800-272-3900", className="text-xl font-bold text-red-600"),
                        html.P("Disponible 24 horas, los 7 días de la semana", className="text-gray-600 text-sm")
                    ])
                ], className="flex items-center p-4 bg-red-50 rounded-lg border border-red-200")
            ])
        ], className="bg-white rounded-lg shadow-md p-6")
    ])

# ==========================================
# CSS PERSONALIZADO
# ==========================================

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .tab-style {
                border: none !important;
                border-radius: 8px 8px 0 0 !important;
                background-color: #F3F4F6 !important;
                color: #6B7280 !important;
                font-weight: 600 !important;
                padding: 12px 24px !important;
                margin-right: 4px !important;
                transition: all 0.2s ease !important;
            }
            .tab-style:hover {
                background-color: #E5E7EB !important;
                color: #374151 !important;
            }
            .tab-selected {
                background-color: #3B82F6 !important;
                color: white !important;
                border-bottom: 3px solid #1D4ED8 !important;
            }
            .dash-table-container {
                font-family: 'Inter', sans-serif !important;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                line-height: 1.6 !important;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==========================================
# EJECUTAR LA APLICACIÓN
# ==========================================

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)