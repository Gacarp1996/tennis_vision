# Tennis Vision - Análisis Automático de Video de Tenis 🎾

Sistema completo de visión por computadora para analizar partidos de tenis desde video, extrayendo métricas detalladas de rendimiento.

## Características

### Métricas Principales
- **Conteo de golpes**: Derecha, revés, saques, voleas
- **Clasificación de errores**: Red, largo, ancho
- **Análisis de profundidad**: Zonas corta/media/profunda
- **Errores en primeras 3 pelotas** de cada jugador
- **Mapas de calor** de posiciones y golpes
- **Detección automática** de inicio/fin de puntos

### Funcionalidades Técnicas
- Calibración de cancha (automática o manual)
- Tracking de pelota y jugadores
- Clasificación de poses con MediaPipe
- Vista optimizada para cámara trasera elevada (4m)
- UI de revisión y corrección manual
- Exportación a CSV, JSON, Excel
- Generación de video con overlays

## Instalación

### Requisitos
- Python 3.9+
- OpenCV
- CUDA (opcional, para GPU)

### Pasos de instalación

1. Clonar el repositorio:
```bash