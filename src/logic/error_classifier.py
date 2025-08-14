"""
Clasificador de errores en tenis.
Determina tipo de error (red/largo/ancho) y si es forzado o no forzado.
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

from ..config import config


class ErrorType(Enum):
    """Tipo específico de error"""
    NET = "net"           # Pelota en la red
    LONG = "long"         # Pelota larga
    WIDE = "wide"         # Pelota ancha
    DOUBLE_FAULT = "double_fault"  # Doble falta
    FOOT_FAULT = "foot_fault"      # Falta de pie
    UNKNOWN = "unknown"


class ErrorCategory(Enum):
    """Categoría del error"""
    UNFORCED = "unforced"  # Error no forzado
    FORCED = "forced"      # Error forzado
    WINNER = "winner"      # Ganador (no es error)


@dataclass
class ErrorAnalysis:
    """Resultado del análisis de error"""
    is_error: bool
    error_type: Optional[ErrorType]
    error_category: Optional[ErrorCategory]
    confidence: float
    
    # Detalles específicos
    ball_landing_position: Optional[Tuple[float, float]] = None
    distance_from_line: Optional[float] = None  # Metros fuera de línea
    net_clearance: Optional[float] = None       # Altura sobre la red
    
    # Contexto para determinar si es forzado
    opponent_position: Optional[Tuple[float, float]] = None
    time_pressure: Optional[float] = None  # Segundos disponibles
    shot_difficulty: Optional[float] = None  # 0-1 escala de dificultad


class ErrorClassifier:
    """
    Clasificador de errores en tenis.
    Analiza trayectorias para determinar errores y su tipo.
    """
    
    def __init__(self, court_calibration=None):
        self.court_calibration = court_calibration
        
        # Umbrales para clasificación
        self.net_height = config.net_height_center
        self.court_width = config.singles_width
        self.court_length = config.court_length
        
        # Margen para decisiones (líneas tienen 5cm de ancho)
        self.line_margin = 0.05  # 5cm
        
        # Historia para análisis temporal
        self.trajectory_buffer: List[Tuple[float, float, float]] = []  # (x, y, z)
        self.buffer_size = 30  # frames
    
    def analyze_shot(
        self,
        ball_trajectory: List[Tuple[float, float, float]],
        player_position: Tuple[float, float],
        opponent_position: Tuple[float, float],
        shot_speed: float,
        is_serve: bool = False
    ) -> ErrorAnalysis:
        """
        Analiza un golpe completo para determinar si es error.
        
        Args:
            ball_trajectory: Lista de posiciones (x, y, z) en metros
            player_position: Posición del jugador que golpeó
            opponent_position: Posición del oponente
            shot_speed: Velocidad del golpe en m/s
            is_serve: Si es un saque
            
        Returns:
            Análisis completo del error
        """
        if len(ball_trajectory) < 2:
            return ErrorAnalysis(
                is_error=False,
                error_type=None,
                error_category=None,
                confidence=0.0
            )
        
        # Predecir punto de aterrizaje
        landing_point = self._predict_landing_point(ball_trajectory)
        
        # Verificar si cruza la red
        crosses_net, net_clearance = self._check_net_clearance(ball_trajectory)
        
        if not crosses_net:
            # Error de red
            return ErrorAnalysis(
                is_error=True,
                error_type=ErrorType.NET,
                error_category=self._determine_error_category(
                    player_position, opponent_position, shot_speed
                ),
                confidence=0.9,
                net_clearance=net_clearance
            )
        
        # Verificar si cae dentro de la cancha
        is_in, distance_out = self._check_ball_in_court(landing_point, is_serve)
        
        if not is_in:
            # Determinar si es largo o ancho
            error_type = self._determine_out_type(landing_point, distance_out)
            
            return ErrorAnalysis(
                is_error=True,
                error_type=error_type,
                error_category=self._determine_error_category(
                    player_position, opponent_position, shot_speed
                ),
                confidence=0.85,
                ball_landing_position=landing_point,
                distance_from_line=distance_out,
                net_clearance=net_clearance
            )
        
        # No es error - posible winner
        difficulty = self._calculate_shot_difficulty(
            landing_point, opponent_position, shot_speed
        )
        
        if difficulty > 0.8:  # Muy difícil de alcanzar
            return ErrorAnalysis(
                is_error=False,
                error_type=None,
                error_category=ErrorCategory.WINNER,
                confidence=0.8,
                ball_landing_position=landing_point,
                shot_difficulty=difficulty
            )
        
        return ErrorAnalysis(
            is_error=False,
            error_type=None,
            error_category=None,
            confidence=0.7,
            ball_landing_position=landing_point
        )
    
    def _predict_landing_point(
        self,
        trajectory: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Predice dónde caerá la pelota basado en la trayectoria.
        Usa física simple con gravedad.
        """
        if len(trajectory) < 3:
            # No hay suficientes puntos, usar último conocido
            return (trajectory[-1][0], trajectory[-1][1])
        
        # Tomar últimos puntos para estimar velocidad y dirección
        recent_points = trajectory[-5:]
        
        # Estimar velocidad en cada componente
        dt = 1.0 / config.target_fps  # Tiempo entre frames
        
        vx_samples = []
        vy_samples = []
        vz_samples = []
        
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i-1][0]
            dy = recent_points[i][1] - recent_points[i-1][1]
            dz = recent_points[i][2] - recent_points[i-1][2]
            
            vx_samples.append(dx / dt)
            vy_samples.append(dy / dt)
            vz_samples.append(dz / dt)
        
        # Promediar velocidades
        vx = np.mean(vx_samples) if vx_samples else 0
        vy = np.mean(vy_samples) if vy_samples else 0
        vz = np.mean(vz_samples) if vz_samples else 0
        
        # Posición actual
        x0, y0, z0 = recent_points[-1]
        
        # Simular hasta que la pelota toque el suelo (z = 0)
        g = 9.81  # Gravedad
        t = 0
        max_t = 5.0  # Máximo 5 segundos de simulación
        
        while z0 > 0 and t < max_t:
            t += dt
            x0 += vx * dt
            y0 += vy * dt
            z0 += vz * dt
            vz -= g * dt  # Aplicar gravedad
        
        return (x0, y0)
    
    def _check_net_clearance(
        self,
        trajectory: List[Tuple[float, float, float]]
    ) -> Tuple[bool, float]:
        """
        Verifica si la pelota cruza la red y con qué margen.
        
        Returns:
            (cruza_la_red, altura_sobre_red)
        """
        net_y = self.court_length / 2
        
        # Encontrar puntos antes y después de la red
        before_net = None
        after_net = None
        
        for i in range(len(trajectory) - 1):
            y1 = trajectory[i][1]
            y2 = trajectory[i+1][1]
            
            # Detectar cruce de red
            if (y1 < net_y <= y2) or (y2 < net_y <= y1):
                before_net = trajectory[i]
                after_net = trajectory[i+1]
                break
        
        if before_net is None or after_net is None:
            # No cruza la red en la trayectoria
            return False, 0.0
        
        # Interpolar altura en la red
        t = (net_y - before_net[1]) / (after_net[1] - before_net[1])
        height_at_net = before_net[2] + t * (after_net[2] - before_net[2])
        
        # Verificar si pasa por encima
        if height_at_net > self.net_height:
            return True, height_at_net - self.net_height
        else:
            return False, height_at_net - self.net_height
    
    def _check_ball_in_court(
        self,
        landing_point: Tuple[float, float],
        is_serve: bool = False
    ) -> Tuple[bool, float]:
        """
        Verifica si el punto de aterrizaje está dentro de la cancha.
        
        Returns:
            (está_dentro, distancia_fuera)
        """
        x, y = landing_point
        
        # Límites laterales (singles)
        if x < -self.line_margin or x > self.court_width + self.line_margin:
            distance_out = max(
                -self.line_margin - x,
                x - (self.court_width + self.line_margin)
            )
            return False, distance_out
        
        if is_serve:
            # Para saque, debe caer en cuadro de servicio
            # Determinar lado del saque basado en posición Y
            if y < self.court_length / 2:
                # Sacando al lado cercano
                service_limit = config.baseline_to_serviceline
            else:
                # Sacando al lado lejano
                service_limit = self.court_length - config.baseline_to_serviceline
            
            if y < 0 or y > service_limit:
                distance_out = max(-y, y - service_limit)
                return False, distance_out
        else:
            # Límites de fondo
            if y < -self.line_margin or y > self.court_length + self.line_margin:
                distance_out = max(
                    -self.line_margin - y,
                    y - (self.court_length + self.line_margin)
                )
                return False, distance_out
        
        return True, 0.0
    
    def _determine_out_type(
        self,
        landing_point: Tuple[float, float],
        distance_out: float
    ) -> ErrorType:
        """
        Determina si el error fue largo o ancho.
        """
        x, y = landing_point
        
        # Verificar qué límite se excedió más
        lateral_excess = max(
            -x if x < 0 else 0,
            x - self.court_width if x > self.court_width else 0
        )
        
        longitudinal_excess = max(
            -y if y < 0 else 0,
            y - self.court_length if y > self.court_length else 0
        )
        
        if lateral_excess > longitudinal_excess:
            return ErrorType.WIDE
        else:
            return ErrorType.LONG
    
    def _determine_error_category(
        self,
        player_position: Tuple[float, float],
        opponent_position: Tuple[float, float],
        shot_speed: float
    ) -> ErrorCategory:
        """
        Determina si un error fue forzado o no forzado.
        Basado en presión del oponente y dificultad del golpe.
        """
        # Calcular distancia que tuvo que recorrer el jugador
        if hasattr(self, 'last_opponent_shot_position'):
            distance_to_ball = np.linalg.norm(
                np.array(player_position) - np.array(self.last_opponent_shot_position)
            )
        else:
            distance_to_ball = 0
        
        # Calcular tiempo disponible
        if shot_speed > 0:
            time_available = distance_to_ball / shot_speed
        else:
            time_available = float('inf')
        
        # Factores para determinar si es forzado
        factors = []
        
        # Factor 1: Tiempo de reacción
        if time_available < config.forced_error_reaction_time:
            factors.append(1.0)
        else:
            factors.append(0.3)
        
        # Factor 2: Velocidad del golpe previo
        if shot_speed > config.forced_error_speed_threshold:
            factors.append(0.8)
        else:
            factors.append(0.2)
        
        # Factor 3: Posición del jugador (fuera de posición)
        court_center = (self.court_width / 2, self.court_length / 2)
        distance_from_center = np.linalg.norm(
            np.array(player_position) - np.array(court_center)
        )
        
        if distance_from_center > self.court_width / 3:
            factors.append(0.7)
        else:
            factors.append(0.3)
        
        # Promedio ponderado de factores
        pressure_score = np.mean(factors)
        
        if pressure_score > 0.6:
            return ErrorCategory.FORCED
        else:
            return ErrorCategory.UNFORCED
    
    def _calculate_shot_difficulty(
        self,
        landing_point: Tuple[float, float],
        opponent_position: Tuple[float, float],
        shot_speed: float
    ) -> float:
        """
        Calcula la dificultad del golpe para el oponente (0-1).
        Usado para determinar winners.
        """
        # Distancia que debe recorrer el oponente
        distance = np.linalg.norm(
            np.array(landing_point) - np.array(opponent_position)
        )
        
        # Tiempo disponible
        if shot_speed > 0:
            time_available = distance / (shot_speed * 0.3)  # Factor de reducción
        else:
            time_available = float('inf')
        
        # Calcular dificultad
        if time_available < 0.5:
            difficulty = 1.0
        elif time_available < 1.0:
            difficulty = 0.8
        elif time_available < 1.5:
            difficulty = 0.6
        elif time_available < 2.0:
            difficulty = 0.4
        else:
            difficulty = 0.2
        
        # Ajustar por posición en la cancha
        # Golpes a las esquinas son más difíciles
        corner_distances = [
            np.linalg.norm(np.array(landing_point) - np.array([0, 0])),
            np.linalg.norm(np.array(landing_point) - np.array([0, self.court_length])),
            np.linalg.norm(np.array(landing_point) - np.array([self.court_width, 0])),
            np.linalg.norm(np.array(landing_point) - np.array([self.court_width, self.court_length]))
        ]
        
        min_corner_distance = min(corner_distances)
        if min_corner_distance < 2.0:
            difficulty += 0.2
        
        return min(1.0, difficulty)
    
    def update_trajectory_buffer(
        self,
        position: Tuple[float, float, float]
    ):
        """Actualiza el buffer de trayectoria"""
        self.trajectory_buffer.append(position)
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
    
    def classify_serve_fault(
        self,
        serve_trajectory: List[Tuple[float, float, float]],
        is_first_serve: bool = True
    ) -> ErrorAnalysis:
        """
        Clasifica específicamente faltas de saque.
        """
        # Verificar si cruza la red
        crosses_net, net_clearance = self._check_net_clearance(serve_trajectory)
        
        if not crosses_net:
            return ErrorAnalysis(
                is_error=True,
                error_type=ErrorType.NET,
                error_category=ErrorCategory.UNFORCED,  # Saques siempre no forzados
                confidence=0.95,
                net_clearance=net_clearance
            )
        
        # Verificar aterrizaje en cuadro de servicio
        landing_point = self._predict_landing_point(serve_trajectory)
        is_in, distance_out = self._check_ball_in_court(landing_point, is_serve=True)
        
        if not is_in:
            error_type = ErrorType.LONG if landing_point[1] > config.baseline_to_serviceline else ErrorType.WIDE
            
            # Si es segunda falta, es doble falta
            if not is_first_serve:
                error_type = ErrorType.DOUBLE_FAULT
            
            return ErrorAnalysis(
                is_error=True,
                error_type=error_type,
                error_category=ErrorCategory.UNFORCED,
                confidence=0.9,
                ball_landing_position=landing_point,
                distance_from_line=distance_out
            )
        
        # Saque bueno
        return ErrorAnalysis(
            is_error=False,
            error_type=None,
            error_category=None,
            confidence=0.9,
            ball_landing_position=landing_point
        )