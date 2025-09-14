"""
Punto de entrada para el despliegue en Render
Dashboard Alzheimer - Fase 6 del Proyecto
"""

from alzheimer_dashboard import app, server

# Exponer el servidor para Gunicorn
application = server

if __name__ == '__main__':
    app.run_server()