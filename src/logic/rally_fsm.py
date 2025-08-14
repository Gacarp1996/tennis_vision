"""
Máquina de estados finitos (FSM) para gestionar el estado del rally.
Detecta inicio/fin de puntos, cuenta golpes, y mantiene el estado del juego.
"""
from enum import Enum
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

from ..config import config


class RallyState(Enum):
    """Estados posibles del rally"""
    IDLE = "idle"                    # Sin actividad
    SERVE_READY = "serve_ready"      # Preparando saque
    SERVE = "serve"                  # Sacando
    IN_PLAY = "in_play"             # Pelota en juego
    POINT_ENDED = "point_ended"     # Punto terminado
    BETWEEN_POINTS = "between_points"  # Entre puntos


class PointEndReason(Enum):
    """Razón por la que terminó el punto"""
    WINNER = "winner"
    UNFORCED_ERROR = "unforced_error"
    FORCED_ERROR = "forced_error"
    ACE = "ace"
    DOUBLE_FAULT = "double_fault"
    NET_ERROR = "net_error"
    OUT_LONG = "out_long"
    OUT_WIDE = "out_wide"
    UNKNOWN = "unknown"


@dataclass
class StrokeEvent:
    """Evento de golpe durante el rally"""
    frame_idx: int
    timestamp: float  # Segundos desde inicio del video
    player_id: int
    stroke_type: str  # "forehand", "backhand", "serve", etc.
    ball_position: Tuple[float, float]  # Posición en metros
    ball_speed: float  # Velocidad en m/s
    is_winner: bool = False
    is_error: bool = False
    error_type: Optional[PointEndReason] = None
    depth_zone: Optional[str] = None  # "corta", "media", "profunda"
    stroke_number: int = 0  # Número de golpe en el rally


@dataclass
class Point:
    """Información completa de un punto"""
    point_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    server_id: int
    winner_id: Optional[int] = None
    
    # Conteos
    total_strokes: int = 0
    strokes_player_0: int = 0
    strokes_player_1: int = 0
    
    # Eventos del punto
    strokes: List[StrokeEvent] = field(default_factory=list)
    end_reason: PointEndReason = PointEndReason.UNKNOWN
    
    # Métricas específicas
    first_serve: bool = True
    serve_speed: Optional[float] = None
    longest_rally_exchange: int = 0  # Golpes consecutivos sin error
    
    # Errores en primeras 3 pelotas
    errors_first_3_player_0: int = 0
    errors_first_3_player_1: int = 0
    
    def add_stroke(self, stroke: StrokeEvent):
        """Agrega un golpe al punto"""
        stroke.stroke_number = len(self.strokes) + 1
        self.strokes.append(stroke)
        self.total_strokes += 1
        
        if stroke.player_id == 0:
            self.strokes_player_0 += 1
            # Verificar si es error en primeras 3 pelotas del jugador
            if stroke.is_error and self.strokes_player_0 <= 3:
                self.errors_first_3_player_0 += 1
        else:
            self.strokes_player_1 += 1
            if stroke.is_error and self.strokes_player_1 <= 3:
                self.errors_first_3_player_1 += 1
    
    def get_player_strokes(self, player_id: int) -> List[StrokeEvent]:
        """Obtiene todos los golpes de un jugador"""
        return [s for s in self.strokes if s.player_id == player_id]
    
    def get_first_n_strokes(self, player_id: int, n: int = 3) -> List[StrokeEvent]:
        """Obtiene los primeros N golpes de un jugador"""
        player_strokes = self.get_player_strokes(player_id)
        return player_strokes[:n]


class RallyStateMachine:
    """
    Máquina de estados para gestionar rallies y puntos.
    Procesa eventos frame a frame y mantiene el estado del juego.
    """
    
    def __init__(self, fps: float = 30.0):
        self.state = RallyState.IDLE
        self.fps = fps
        self.frame_count = 0
        
        # Punto actual
        self.current_point: Optional[Point] = None
        self.points: List[Point] = []
        self.next_point_id = 0
        
        # Seguimiento del rally
        self.frames_without_ball = 0
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.last_ball_speed: float = 0.0
        self.current_server: int = 0  # Alternará entre 0 y 1
        
        # Buffer para detectar patrones
        self.position_buffer: List[Tuple[float, float]] = []
        self.speed_buffer: List[float] = []
        self.buffer_size = 10
        
        # Estadísticas globales
        self.total_strokes = 0
        self.total_errors = 0
        self.total_winners = 0
    
    def update(
        self,
        frame_idx: int,
        ball_detected: bool,
        ball_position: Optional[Tuple[float, float]] = None,
        ball_speed: Optional[float] = None,
        player_hitting: Optional[int] = None,
        stroke_type: Optional[str] = None
    ) -> RallyState:
        """
        Actualiza el estado del rally con información del frame actual.
        
        Args:
            frame_idx: Índice del frame actual
            ball_detected: Si se detectó la pelota
            ball_position: Posición de la pelota en metros
            ball_speed: Velocidad de la pelota en m/s
            player_hitting: ID del jugador golpeando (si aplica)
            stroke_type: Tipo de golpe detectado
            
        Returns:
            Estado actual del rally
        """
        self.frame_count = frame_idx
        
        # Actualizar buffers
        if ball_position is not None:
            self.position_buffer.append(ball_position)
            if len(self.position_buffer) > self.buffer_size:
                self.position_buffer.pop(0)
            self.last_ball_position = ball_position
        
        if ball_speed is not None:
            self.speed_buffer.append(ball_speed)
            if len(self.speed_buffer) > self.buffer_size:
                self.speed_buffer.pop(0)
            self.last_ball_speed = ball_speed
        
        # Procesar según estado actual
        if self.state == RallyState.IDLE:
            self._process_idle(ball_detected, ball_speed, stroke_type)
        
        elif self.state == RallyState.SERVE_READY:
            self._process_serve_ready(ball_detected, stroke_type)
        
        elif self.state == RallyState.SERVE:
            self._process_serve(ball_detected, ball_speed, player_hitting)
        
        elif self.state == RallyState.IN_PLAY:
            self._process_in_play(
                ball_detected, ball_position, ball_speed, 
                player_hitting, stroke_type
            )
        
        elif self.state == RallyState.POINT_ENDED:
            self._process_point_ended(ball_detected)
        
        elif self.state == RallyState.BETWEEN_POINTS:
            self._process_between_points(ball_detected)
        
        return self.state
    
    def _process_idle(
        self, 
        ball_detected: bool, 
        ball_speed: Optional[float],
        stroke_type: Optional[str]
    ):
        """Procesa estado IDLE - esperando inicio de punto"""
        if stroke_type == "serve" or self._detect_serve_motion():
            self.state = RallyState.SERVE_READY
            self._start_new_point()
    
    def _process_serve_ready(
        self,
        ball_detected: bool,
        stroke_type: Optional[str]
    ):
        """Procesa preparación de saque"""
        if stroke_type == "serve":
            self.state = RallyState.SERVE
            self.frames_without_ball = 0
        elif self.frames_without_ball > config.max_frames_without_ball * 2:
            # Timeout - volver a idle
            self.state = RallyState.IDLE
            self.current_point = None
    
    def _process_serve(
        self,
        ball_detected: bool,
        ball_speed: Optional[float],
        player_hitting: Optional[int]
    ):
        """Procesa el saque"""
        if ball_detected and ball_speed is not None:
            if ball_speed > config.ball_speed_threshold:
                self.state = RallyState.IN_PLAY
                
                # Registrar el saque como primer golpe
                if self.current_point and player_hitting is not None:
                    stroke = StrokeEvent(
                        frame_idx=self.frame_count,
                        timestamp=self.frame_count / self.fps,
                        player_id=self.current_server,
                        stroke_type="serve",
                        ball_position=self.last_ball_position or (0, 0),
                        ball_speed=ball_speed
                    )
                    self.current_point.add_stroke(stroke)
                    self.current_point.serve_speed = ball_speed
    
    def _process_in_play(
        self,
        ball_detected: bool,
        ball_position: Optional[Tuple[float, float]],
        ball_speed: Optional[float],
        player_hitting: Optional[int],
        stroke_type: Optional[str]
    ):
        """Procesa pelota en juego"""
        if not ball_detected:
            self.frames_without_ball += 1
            
            if self.frames_without_ball > config.max_frames_without_ball:
                # Punto terminado
                self._end_current_point()
                self.state = RallyState.POINT_ENDED
        else:
            self.frames_without_ball = 0
            
            # Detectar golpe
            if player_hitting is not None and self._detect_hit(ball_speed):
                if self.current_point:
                    stroke = StrokeEvent(
                        frame_idx=self.frame_count,
                        timestamp=self.frame_count / self.fps,
                        player_id=player_hitting,
                        stroke_type=stroke_type or "unknown",
                        ball_position=ball_position or (0, 0),
                        ball_speed=ball_speed or 0
                    )
                    
                    # Determinar zona de profundidad
                    if ball_position:
                        stroke.depth_zone = self._get_depth_zone(ball_position)
                    
                    self.current_point.add_stroke(stroke)
                    self.total_strokes += 1
            
            # Detectar si la pelota salió
            if self._detect_ball_out(ball_position):
                self._end_current_point(reason=PointEndReason.UNFORCED_ERROR)
                self.state = RallyState.POINT_ENDED
    
    def _process_point_ended(self, ball_detected: bool):
        """Procesa fin de punto"""
        if not ball_detected:
            self.frames_without_ball += 1
            
            if self.frames_without_ball > config.max_frames_without_ball:
                self.state = RallyState.BETWEEN_POINTS
                self.frames_without_ball = 0
    
    def _process_between_points(self, ball_detected: bool):
        """Procesa tiempo entre puntos"""
        # Esperar nueva actividad
        if self._detect_serve_motion():
            self.state = RallyState.SERVE_READY
            self._start_new_point()
            # Alternar servidor
            self.current_server = 1 - self.current_server
    
    def _start_new_point(self):
        """Inicia un nuevo punto"""
        self.current_point = Point(
            point_id=self.next_point_id,
            start_frame=self.frame_count,
            end_frame=self.frame_count,
            duration_frames=0,
            duration_seconds=0,
            server_id=self.current_server
        )
        self.next_point_id += 1
        self.frames_without_ball = 0
    
    def _end_current_point(
        self, 
        reason: PointEndReason = PointEndReason.UNKNOWN,
        winner_id: Optional[int] = None
    ):
        """Finaliza el punto actual"""
        if self.current_point:
            self.current_point.end_frame = self.frame_count
            self.current_point.duration_frames = (
                self.current_point.end_frame - self.current_point.start_frame
            )
            self.current_point.duration_seconds = (
                self.current_point.duration_frames / self.fps
            )
            self.current_point.end_reason = reason
            self.current_point.winner_id = winner_id
            
            # Determinar ganador si no se especificó
            if winner_id is None and len(self.current_point.strokes) > 0:
                last_stroke = self.current_point.strokes[-1]
                if last_stroke.is_error:
                    # El que erró pierde
                    self.current_point.winner_id = 1 - last_stroke.player_id
                else:
                    # El último que golpeó gana
                    self.current_point.winner_id = last_stroke.player_id
            
            self.points.append(self.current_point)
            self.current_point = None
    
    def _detect_serve_motion(self) -> bool:
        """
        Detecta movimiento de saque basado en patrones de velocidad.
        Un saque típicamente muestra aceleración súbita desde reposo.
        """
        if len(self.speed_buffer) < 3:
            return False
        
        # Buscar patrón: velocidad baja -> velocidad alta
        recent_speeds = self.speed_buffer[-3:]
        if recent_speeds[0] < 2.0 and recent_speeds[-1] > 15.0:
            return True
        
        return False
    
    def _detect_hit(self, ball_speed: Optional[float]) -> bool:
        """
        Detecta si hubo un golpe basado en cambios de velocidad/dirección.
        """
        if ball_speed is None or len(self.speed_buffer) < 2:
            return False
        
        # Detectar cambio brusco de velocidad
        if len(self.speed_buffer) >= 2:
            speed_change = abs(self.speed_buffer[-1] - self.speed_buffer[-2])
            if speed_change > 5.0:  # m/s
                return True
        
        # Detectar cambio de dirección
        if len(self.position_buffer) >= 3:
            p1, p2, p3 = self.position_buffer[-3:]
            
            # Vectores de movimiento
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Ángulo entre vectores
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                if angle > np.pi / 4:  # Cambio de más de 45 grados
                    return True
        
        return False
    
    def _detect_ball_out(self, ball_position: Optional[Tuple[float, float]]) -> bool:
        """
        Detecta si la pelota salió de la cancha.
        """
        if ball_position is None:
            return False
        
        x, y = ball_position
        
        # Verificar límites de cancha (singles)
        if x < -0.5 or x > config.singles_width + 0.5:
            return True
        
        if y < -0.5 or y > config.court_length + 0.5:
            return True
        
        return False
    
    def _get_depth_zone(self, ball_position: Tuple[float, float]) -> str:
        """
        Determina la zona de profundidad del golpe.
        """
        _, y = ball_position
        
        # Distancia desde la red
        dist_from_net = abs(y - config.court_length / 2)
        
        for zone_name, (zone_min, zone_max) in config.depth_zones.items():
            if zone_min <= dist_from_net < zone_max:
                return zone_name
        
        return "profunda"  # Por defecto
    
    def get_current_rally_length(self) -> int:
        """Obtiene la longitud del rally actual"""
        if self.current_point:
            return self.current_point.total_strokes
        return 0
    
    def get_match_stats(self) -> Dict:
        """
        Obtiene estadísticas del partido hasta el momento.
        """
        if not self.points:
            return {
                "total_points": 0,
                "total_strokes": 0,
                "avg_rally_length": 0,
                "longest_rally": 0
            }
        
        total_strokes = sum(p.total_strokes for p in self.points)
        rally_lengths = [p.total_strokes for p in self.points]
        
        return {
            "total_points": len(self.points),
            "total_strokes": total_strokes,
            "avg_rally_length": np.mean(rally_lengths) if rally_lengths else 0,
            "longest_rally": max(rally_lengths) if rally_lengths else 0,
            "points_won_player_0": sum(1 for p in self.points if p.winner_id == 0),
            "points_won_player_1": sum(1 for p in self.points if p.winner_id == 1),
            "avg_point_duration": np.mean([p.duration_seconds for p in self.points]),
            "total_aces": sum(1 for p in self.points if p.end_reason == PointEndReason.ACE),
            "total_double_faults": sum(
                1 for p in self.points if p.end_reason == PointEndReason.DOUBLE_FAULT
            )
        }
    
    def export_timeline(self) -> List[Dict]:
        """
        Exporta timeline de eventos para visualización.
        """
        timeline = []
        
        for point in self.points:
            # Evento de inicio de punto
            timeline.append({
                "frame": point.start_frame,
                "time": point.start_frame / self.fps,
                "event": "point_start",
                "point_id": point.point_id,
                "server": point.server_id
            })
            
            # Eventos de golpes
            for stroke in point.strokes:
                timeline.append({
                    "frame": stroke.frame_idx,
                    "time": stroke.timestamp,
                    "event": "stroke",
                    "player": stroke.player_id,
                    "type": stroke.stroke_type,
                    "speed": stroke.ball_speed,
                    "position": stroke.ball_position,
                    "depth": stroke.depth_zone
                })
            
            # Evento de fin de punto
            timeline.append({
                "frame": point.end_frame,
                "time": point.end_frame / self.fps,
                "event": "point_end",
                "point_id": point.point_id,
                "winner": point.winner_id,
                "reason": point.end_reason.value,
                "total_strokes": point.total_strokes
            })
        
        return timeline