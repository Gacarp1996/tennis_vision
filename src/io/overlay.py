"""
Módulo para generar overlays de video con anotaciones.
Dibuja trayectorias, etiquetas y métricas sobre el video.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path

from ..config import config


class VideoOverlay:
    """
    Generador de overlays para videos de tenis.
    Agrega visualizaciones de análisis sobre el video original.
    """
    
    def __init__(self, video_path: Path, output_path: Optional[Path] = None):
        self.video_path = video_path
        self.output_path = output_path or (
            config.overlays_dir / f"overlay_{video_path.stem}.mp4"
        )
        
        # Abrir video de entrada
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurar video de salida
        fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
        self.out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            config.overlay_fps,
            (self.width, self.height)
        )
        
        # Buffers para trails
        self.ball_trail = deque(maxlen=config.overlay_trail_length)
        self.player_trails = {0: deque(maxlen=30), 1: deque(maxlen=30)}
        
        # Colores para jugadores
        self.player_colors = {
            0: (255, 0, 0),    # Azul para jugador 0
            1: (0, 0, 255)     # Rojo para jugador 1
        }
        
        # Estado actual
        self.current_frame = 0
        self.current_point = None
        self.current_rally_length = 0
        
        # Fuentes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_data: Dict,
        calibration=None
    ) -> np.ndarray:
        """
        Procesa un frame agregando overlays.
        
        Args:
            frame: Frame original
            frame_data: Datos del análisis para este frame
            calibration: Calibración de cancha si está disponible
            
        Returns:
            Frame con overlays
        """
        overlay = frame.copy()
        
        # Dibujar cancha calibrada si está disponible
        if calibration:
            overlay = self._draw_court_grid(overlay, calibration)
        
        # Dibujar pelota
        if 'ball_position' in frame_data:
            overlay = self._draw_ball(
                overlay,
                frame_data['ball_position'],
                frame_data.get('ball_speed', 0),
                calibration
            )
        
        # Dibujar jugadores
        if 'players' in frame_data:
            overlay = self._draw_players(
                overlay,
                frame_data['players'],
                calibration
            )
        
        # Dibujar información del golpe actual
        if 'current_stroke' in frame_data:
            overlay = self._draw_stroke_info(
                overlay,
                frame_data['current_stroke']
            )
        
        # Panel de información
        overlay = self._draw_info_panel(overlay, frame_data)
        
        # Marcador si está disponible
        if 'score' in frame_data:
            overlay = self._draw_score(overlay, frame_data['score'])
        
        return overlay
    
    def _draw_court_grid(self, frame: np.ndarray, calibration) -> np.ndarray:
        """Dibuja grid de la cancha calibrada"""
        overlay = frame.copy()
        
        # Líneas principales con transparencia
        alpha = 0.3
        court_overlay = frame.copy()
        
        # Convertir líneas de cancha a píxeles y dibujar
        court_lines = [
            # Baselines
            ([0, 0], [config.singles_width, 0]),
            ([0, config.court_length], [config.singles_width, config.court_length]),
            # Laterales
            ([0, 0], [0, config.court_length]),
            ([config.singles_width, 0], [config.singles_width, config.court_length]),
            # Red
            ([0, config.court_length/2], [config.singles_width, config.court_length/2]),
        ]
        
        for pt1_world, pt2_world in court_lines:
            try:
                pt1_px = calibration.court_to_pixel(np.array(pt1_world))
                pt2_px = calibration.court_to_pixel(np.array(pt2_world))
                
                cv2.line(
                    court_overlay,
                    tuple(pt1_px.astype(int)),
                    tuple(pt2_px.astype(int)),
                    (0, 255, 0), 2
                )
            except:
                pass
        
        # Mezclar con transparencia
        cv2.addWeighted(overlay, 1-alpha, court_overlay, alpha, 0, overlay)
        
        return overlay
    
    def _draw_ball(
        self,
        frame: np.ndarray,
        ball_position: Tuple[float, float],
        ball_speed: float,
        calibration=None
    ) -> np.ndarray:
        """Dibuja la pelota y su trail"""
        overlay = frame.copy()
        
        # Convertir posición a píxeles si hay calibración
        if calibration and len(ball_position) == 2:
            try:
                ball_px = calibration.court_to_pixel(np.array(ball_position))
                ball_px = tuple(ball_px.astype(int))
            except:
                ball_px = ball_position
        else:
            ball_px = ball_position
        
        # Agregar a trail
        self.ball_trail.append(ball_px)
        
        # Dibujar trail con fade
        if len(self.ball_trail) > 1:
            for i in range(1, len(self.ball_trail)):
                alpha = i / len(self.ball_trail)
                color = (
                    int(255 * alpha),  # Amarillo que se desvanece
                    int(255 * alpha),
                    0
                )
                thickness = max(1, int(3 * alpha))
                
                cv2.line(
                    overlay,
                    self.ball_trail[i-1],
                    self.ball_trail[i],
                    color, thickness
                )
        
        # Dibujar pelota actual
        cv2.circle(overlay, ball_px, 8, (0, 255, 255), -1)
        cv2.circle(overlay, ball_px, 10, (0, 255, 255), 2)
        
        # Mostrar velocidad si es significativa
        if ball_speed > 5:
            speed_text = f"{ball_speed:.1f} m/s"
            cv2.putText(
                overlay, speed_text,
                (ball_px[0] + 15, ball_px[1] - 15),
                self.font, 0.4, (0, 255, 255), 1
            )
        
        return overlay
    
    def _draw_players(
        self,
        frame: np.ndarray,
        players_data: List[Dict],
        calibration=None
    ) -> np.ndarray:
        """Dibuja jugadores con sus bounding boxes y trails"""
        overlay = frame.copy()
        
        for player in players_data:
            player_id = player['id']
            bbox = player.get('bbox')
            position = player.get('position')
            stroke = player.get('current_stroke')
            
            color = self.player_colors.get(player_id, (0, 255, 0))
            
            # Dibujar bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Label con ID y golpe actual
                label = f"P{player_id}"
                if stroke:
                    label += f" - {stroke}"
                
                label_size = cv2.getTextSize(
                    label, self.font, self.font_scale, self.font_thickness
                )[0]
                
                # Fondo para el texto
                cv2.rectangle(
                    overlay,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    color, -1
                )
                
                cv2.putText(
                    overlay, label,
                    (x1 + 5, y1 - 5),
                    self.font, self.font_scale, (255, 255, 255), self.font_thickness
                )
            
            # Agregar posición al trail
            if position:
                if calibration:
                    try:
                        pos_px = calibration.court_to_pixel(np.array(position))
                        pos_px = tuple(pos_px.astype(int))
                    except:
                        pos_px = position
                else:
                    pos_px = position
                
                self.player_trails[player_id].append(pos_px)
            
            # Dibujar trail del jugador
            trail = self.player_trails.get(player_id, [])
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(2 * alpha))
                    
                    # Color con fade
                    faded_color = tuple(int(c * alpha) for c in color)
                    
                    cv2.line(
                        overlay,
                        trail[i-1],
                        trail[i],
                        faded_color, thickness
                    )
        
        return overlay
    
    def _draw_stroke_info(
        self,
        frame: np.ndarray,
        stroke_data: Dict
    ) -> np.ndarray:
        """Dibuja información del golpe actual"""
        overlay = frame.copy()
        
        # Extraer información
        stroke_type = stroke_data.get('type', 'unknown')
        depth_zone = stroke_data.get('depth_zone', '')
        is_winner = stroke_data.get('is_winner', False)
        is_error = stroke_data.get('is_error', False)
        
        # Determinar color y mensaje
        if is_winner:
            color = (0, 255, 0)  # Verde
            message = f"WINNER - {stroke_type}"
        elif is_error:
            color = (0, 0, 255)  # Rojo
            error_type = stroke_data.get('error_type', 'Error')
            message = f"ERROR ({error_type}) - {stroke_type}"
        else:
            color = (255, 255, 255)  # Blanco
            message = f"{stroke_type}"
        
        if depth_zone:
            message += f" - Zona {depth_zone}"
        
        # Posición del texto (centro superior)
        text_size = cv2.getTextSize(
            message, self.font, 0.8, 2
        )[0]
        
        x = (self.width - text_size[0]) // 2
        y = 100
        
        # Fondo semitransparente
        overlay_rect = overlay.copy()
        cv2.rectangle(
            overlay_rect,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, overlay_rect, 0.3, 0, overlay)
        
        # Texto
        cv2.putText(
            overlay, message,
            (x, y),
            self.font, 0.8, color, 2
        )
        
        return overlay
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        frame_data: Dict
    ) -> np.ndarray:
        """Dibuja panel de información en la esquina"""
        overlay = frame.copy()
        
        # Información a mostrar
        info_lines = []
        
        # Frame y tiempo
        frame_num = frame_data.get('frame_idx', self.current_frame)
        time_sec = frame_num / self.fps
        info_lines.append(f"Tiempo: {time_sec:.1f}s")
        
        # Estado del rally
        rally_state = frame_data.get('rally_state', 'IDLE')
        info_lines.append(f"Estado: {rally_state}")
        
        # Punto actual
        if 'current_point' in frame_data:
            point = frame_data['current_point']
            info_lines.append(f"Punto #{point['id']}")
            info_lines.append(f"Rally: {point.get('strokes', 0)} golpes")
        
        # Estadísticas rápidas
        if 'quick_stats' in frame_data:
            stats = frame_data['quick_stats']
            info_lines.append("---")
            info_lines.append(f"P0: {stats.get('strokes_p0', 0)} golpes")
            info_lines.append(f"P1: {stats.get('strokes_p1', 0)} golpes")
        
        # Dibujar panel
        panel_width = 250
        panel_height = len(info_lines) * 25 + 20
        panel_x = self.width - panel_width - 10
        panel_y = 10
        
        # Fondo semitransparente
        overlay_rect = overlay.copy()
        cv2.rectangle(
            overlay_rect,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.8, overlay_rect, 0.2, 0, overlay)
        
        # Texto
        for i, line in enumerate(info_lines):
            y_pos = panel_y + 25 + i * 25
            cv2.putText(
                overlay, line,
                (panel_x + 10, y_pos),
                self.font, 0.5, (255, 255, 255), 1
            )
        
        return overlay
    
    def _draw_score(
        self,
        frame: np.ndarray,
        score_data: Dict
    ) -> np.ndarray:
        """Dibuja el marcador del partido"""
        overlay = frame.copy()
        
        # Extraer puntuación
        p0_games = score_data.get('player_0_games', 0)
        p1_games = score_data.get('player_1_games', 0)
        p0_points = score_data.get('player_0_points', 0)
        p1_points = score_data.get('player_1_points', 0)
        
        # Convertir puntos a notación de tenis
        points_map = {0: "0", 1: "15", 2: "30", 3: "40"}
        p0_points_str = points_map.get(p0_points, str(p0_points))
        p1_points_str = points_map.get(p1_points, str(p1_points))
        
        # Texto del marcador
        score_text = f"P0: {p0_games} ({p0_points_str})  -  P1: {p1_games} ({p1_points_str})"
        
        # Posición (centro inferior)
        text_size = cv2.getTextSize(
            score_text, self.font, 0.7, 2
        )[0]
        
        x = (self.width - text_size[0]) // 2
        y = self.height - 30
        
        # Fondo
        cv2.rectangle(
            overlay,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0), -1
        )
        
        # Texto
        cv2.putText(
            overlay, score_text,
            (x, y),
            self.font, 0.7, (255, 255, 255), 2
        )
        
        return overlay
    
    def generate_overlay_video(
        self,
        frame_data_list: List[Dict],
        calibration=None,
        skip_frames: int = 1
    ):
        """
        Genera video completo con overlays.
        
        Args:
            frame_data_list: Lista con datos de análisis por frame
            calibration: Calibración de cancha
            skip_frames: Procesar cada N frames (para velocidad)
        """
        print(f"\nGenerando video con overlay...")
        print(f"Video entrada: {self.video_path}")
        print(f"Video salida: {self.output_path}")
        print(f"Total frames: {self.total_frames}")
        
        frame_idx = 0
        processed = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Procesar solo cada N frames
            if frame_idx % skip_frames == 0:
                # Obtener datos para este frame
                if frame_idx < len(frame_data_list):
                    frame_data = frame_data_list[frame_idx]
                else:
                    frame_data = {}
                
                # Agregar overlay
                frame_with_overlay = self.process_frame(
                    frame, frame_data, calibration
                )
                
                # Escribir frame
                self.out.write(frame_with_overlay)
                processed += 1
                
                # Progreso
                if processed % 100 == 0:
                    progress = (frame_idx / self.total_frames) * 100
                    print(f"Progreso: {progress:.1f}%")
            
            frame_idx += 1
            self.current_frame = frame_idx
        
        # Limpiar
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Video con overlay generado: {self.output_path}")
        print(f"  Frames procesados: {processed}")
    
    def generate_highlights_video(
        self,
        highlights_data: List[Dict],
        calibration=None
    ):
        """
        Genera video solo con los highlights (winners, aces, rallies largos).
        
        Args:
            highlights_data: Lista de momentos destacados con frame_start y frame_end
            calibration: Calibración de cancha
        """
        output_path = self.output_path.parent / f"highlights_{self.output_path.name}"
        
        print(f"\nGenerando video de highlights...")
        print(f"Momentos destacados: {len(highlights_data)}")
        
        # Configurar nuevo writer para highlights
        fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
        highlights_out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        for i, highlight in enumerate(highlights_data):
            start_frame = highlight['frame_start']
            end_frame = highlight['frame_end']
            highlight_type = highlight.get('type', 'highlight')
            
            print(f"  Procesando highlight {i+1}/{len(highlights_data)}: {highlight_type}")
            
            # Posicionar en frame inicial
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Procesar frames del highlight
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Agregar título del highlight
                title = f"{highlight_type.upper()}"
                if 'description' in highlight:
                    title += f" - {highlight['description']}"
                
                cv2.putText(
                    frame, title,
                    (50, 50),
                    self.font, 1.0, (0, 255, 255), 2
                )
                
                highlights_out.write(frame)
        
        highlights_out.release()
        print(f"✓ Video de highlights generado: {output_path}")