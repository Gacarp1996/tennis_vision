"""
Configuración central del sistema de análisis de tenis.
Define umbrales, rutas y parámetros ajustables.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    """Configuración principal del sistema"""
    
    # === RUTAS ===
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path("data"))
    videos_dir: Path = field(default_factory=lambda: Path("data/videos"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    overlays_dir: Path = field(default_factory=lambda: Path("outputs/overlays"))
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    
    # === DIMENSIONES DE CANCHA (ITF estándar en metros) ===
    court_length: float = 23.77  # Largo total
    court_width: float = 10.97   # Ancho para dobles
    singles_width: float = 8.23  # Ancho para singles
    service_box_length: float = 6.40
    net_height_center: float = 0.914
    net_height_posts: float = 1.07
    baseline_to_serviceline: float = 5.49
    
    # === ZONAS DE PROFUNDIDAD (desde línea de fondo rival) ===
    depth_zones: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "corta": (0.0, 3.0),      # 0-3m desde la red
        "media": (3.0, 6.0),      # 3-6m desde la red  
        "profunda": (6.0, 11.89)  # 6m hasta línea de fondo
    })
    
    # === DETECCIÓN Y TRACKING ===
    yolo_model: str = "models/tennis_ball.pt"  # Modelo base, cambiaremos a uno específico
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    
    # Tracking
    max_distance_threshold: float = 50.0  # Píxeles máx entre frames
    track_buffer: int = 30  # Frames para mantener tracks perdidos
    min_track_length: int = 5  # Mínimo de frames para considerar válido
    
    # === CALIBRACIÓN DE CANCHA ===
    calibration_min_lines: int = 4  # Mínimo de líneas para calibrar
    hough_threshold: int = 100
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 50
    canny_low: int = 50
    canny_high: int = 150
    
    # === LÓGICA DE RALLY ===
    ball_speed_threshold: float = 5.0  # m/s mínimo para "en juego"
    max_frames_without_ball: int = 30  # Para declarar punto terminado
    min_rally_duration: int = 10  # Frames mínimos para rally válido
    
    # === CLASIFICACIÓN DE GOLPES ===
    # Umbrales para pose (MediaPipe)
    forehand_shoulder_angle_min: float = 45.0   # Grados
    forehand_shoulder_angle_max: float = 135.0
    backhand_shoulder_angle_min: float = -45.0
    backhand_shoulder_angle_max: float = 45.0
    
    # Distancia máxima jugador-pelota para asignar golpe
    max_player_ball_distance: float = 3.0  # metros
    
    # === CLASIFICACIÓN DE ERRORES ===
    # Tiempo de reacción para error forzado vs no forzado
    forced_error_reaction_time: float = 0.8  # segundos
    forced_error_speed_threshold: float = 25.0  # m/s
    
    # === PROCESAMIENTO DE VIDEO ===
    target_fps: int = 30
    frame_skip: int = 1  # Procesar cada N frames (1 = todos)
    resize_width: int = 1280  # Redimensionar para procesamiento
    resize_height: int = 720
    
    # === UI Y VISUALIZACIÓN ===
    overlay_trail_length: int = 10  # Frames de estela de pelota
    heatmap_resolution: Tuple[int, int] = (100, 200)  # Bins para heatmap
    heatmap_sigma: float = 1.5  # Suavizado gaussiano
    
    # === EXPORTACIÓN ===
    csv_decimal_places: int = 2
    video_codec: str = "mp4v"
    overlay_fps: int = 30
    
    def __post_init__(self):
        """Crear directorios si no existen"""
        for dir_path in [self.data_dir, self.videos_dir, self.outputs_dir, 
                         self.overlays_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convertir a diccionario para serialización"""
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Crear desde diccionario"""
        path_fields = ["project_root", "data_dir", "videos_dir", 
                      "outputs_dir", "overlays_dir", "figures_dir"]
        for field in path_fields:
            if field in data and isinstance(data[field], str):
                data[field] = Path(data[field])
        return cls(**data)


# Instancia global de configuración
config = Config()