"""
Detector y tracker de jugadores de tenis.
Usa YOLO para detección y asociación por posición en cancha.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from ultralytics import YOLO

from ..config import config


class PlayerSide(Enum):
    """Lado de la cancha"""
    NEAR = "near"    # Lado cercano a la cámara
    FAR = "far"      # Lado lejano
    UNKNOWN = "unknown"


@dataclass
class PlayerDetection:
    """Detección de jugador en un frame"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float]
    frame_idx: int
    player_id: Optional[int] = None
    side: PlayerSide = PlayerSide.UNKNOWN
    
    @property
    def feet_position(self) -> Tuple[float, float]:
        """Posición estimada de los pies (base del bbox)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property 
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PlayerTrack:
    """Track de un jugador a través del tiempo"""
    
    def __init__(self, player_id: int, initial_detection: PlayerDetection):
        self.player_id = player_id
        self.detections: List[PlayerDetection] = [initial_detection]
        self.side = initial_detection.side
        self.color = None  # Color dominante del uniforme
        self.hits = 1
        self.time_since_update = 0
        
        # Estadísticas de posición
        self.positions_court: List[Tuple[float, float]] = []  # En metros
        self.avg_position = (0.0, 0.0)
        
    def update(self, detection: PlayerDetection):
        """Actualiza el track con nueva detección"""
        detection.player_id = self.player_id
        detection.side = self.side
        self.detections.append(detection)
        self.time_since_update = 0
        self.hits += 1
        
        # Mantener solo las últimas N detecciones
        if len(self.detections) > config.track_buffer:
            self.detections.pop(0)
    
    def predict_position(self, frame_idx: int) -> Tuple[float, float]:
        """Predice posición del jugador"""
        if not self.detections:
            return (0, 0)
        
        # Usar última posición conocida (jugadores se mueven menos que pelotas)
        last = self.detections[-1]
        return last.center
    
    @property
    def is_valid(self) -> bool:
        return self.hits >= config.min_track_length
    
    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Obtiene último bounding box"""
        if self.detections:
            return self.detections[-1].bbox
        return None


class PlayerDetector:
    """Detector y tracker de jugadores"""
    
    def __init__(self):
        self.yolo_model = None
        self.tracks: Dict[int, PlayerTrack] = {}
        self.frame_count = 0
        self.court_calibration = None
        
        self._init_yolo()
    
    def _init_yolo(self):
        """Inicializa modelo YOLO"""
        try:
            # Intentar modelo específico de personas/atletas
            model_path = config.project_root / "models" / "tennis_player.pt"
            if model_path.exists():
                self.yolo_model = YOLO(str(model_path))
            else:
                # Modelo general, clase persona
                self.yolo_model = YOLO(config.yolo_model)
        except Exception as e:
            print(f"Error cargando YOLO para jugadores: {e}")
            raise
    
    def set_court_calibration(self, calibration):
        """Establece calibración de cancha para determinar lados"""
        self.court_calibration = calibration
    
    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detecta jugadores en el frame"""
        if not self.yolo_model:
            return []
        
        results = self.yolo_model(
            frame,
            conf=config.yolo_conf_threshold,
            iou=config.yolo_iou_threshold,
            classes=[0]  # Clase 'person' en COCO
        )
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Centro del bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Crear detección
                detection = PlayerDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    center=(cx, cy),
                    frame_idx=self.frame_count
                )
                
                # Determinar lado de la cancha si hay calibración
                if self.court_calibration:
                    detection.side = self._determine_side(detection)
                
                detections.append(detection)
        
        # Filtrar detecciones fuera de la cancha o muy pequeñas
        detections = self._filter_detections(detections, frame.shape)
        
        return detections
    
    def _determine_side(self, detection: PlayerDetection) -> PlayerSide:
        """
        Determina en qué lado de la cancha está el jugador.
        Usa la posición de los pies y la calibración.
        """
        if not self.court_calibration:
            return PlayerSide.UNKNOWN
        
        try:
            # Convertir posición de pies a coordenadas de cancha
            feet_px = np.array(detection.feet_position)
            feet_court = self.court_calibration.pixel_to_court(feet_px)
            
            # Determinar lado basado en coordenada Y (largo de cancha)
            mid_court = config.court_length / 2
            
            if feet_court[1] < mid_court - 1:  # Margen de 1 metro
                return PlayerSide.NEAR
            elif feet_court[1] > mid_court + 1:
                return PlayerSide.FAR
            else:
                return PlayerSide.UNKNOWN
                
        except:
            return PlayerSide.UNKNOWN
    
    def _filter_detections(
        self, 
        detections: List[PlayerDetection],
        frame_shape: Tuple[int, ...]
    ) -> List[PlayerDetection]:
        """Filtra detecciones no válidas"""
        h, w = frame_shape[:2]
        filtered = []
        
        for det in detections:
            # Verificar tamaño mínimo (evitar detecciones parciales)
            if det.width < 30 or det.height < 60:
                continue
            
            # Verificar que esté dentro del frame
            x1, y1, x2, y2 = det.bbox
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            
            # Verificar proporción altura/ancho (personas de pie)
            aspect_ratio = det.height / det.width
            if aspect_ratio < 1.5 or aspect_ratio > 4.0:
                continue
            
            filtered.append(det)
        
        # Mantener máximo 2 jugadores (singles) o 4 (dobles)
        # Ordenar por confianza y tomar los mejores
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        return filtered[:4]
    
    def update_tracks(self, detections: List[PlayerDetection]) -> Dict[int, PlayerTrack]:
        """Actualiza tracks con nuevas detecciones"""
        # Incrementar tiempo sin actualización
        for track in self.tracks.values():
            track.time_since_update += 1
        
        if len(detections) == 0:
            return self.tracks
        
        # Si no hay tracks, crear nuevos
        if len(self.tracks) == 0:
            for i, det in enumerate(detections[:2]):  # Max 2 jugadores para singles
                track = PlayerTrack(i, det)
                self.tracks[i] = track
            return self.tracks
        
        # Asociar detecciones a tracks existentes
        # Usamos asociación por distancia y lado de cancha
        unmatched_detections = list(detections)
        
        for track_id, track in self.tracks.items():
            best_det = None
            best_distance = float('inf')
            
            for det in unmatched_detections:
                # Priorizar mismo lado de cancha
                if track.side != PlayerSide.UNKNOWN and det.side != PlayerSide.UNKNOWN:
                    if track.side != det.side:
                        continue
                
                # Calcular distancia
                pred_pos = track.predict_position(self.frame_count)
                dist = np.sqrt(
                    (det.center[0] - pred_pos[0])**2 + 
                    (det.center[1] - pred_pos[1])**2
                )
                
                if dist < best_distance and dist < config.max_distance_threshold * 2:
                    best_distance = dist
                    best_det = det
            
            if best_det is not None:
                track.update(best_det)
                unmatched_detections.remove(best_det)
        
        # Crear nuevos tracks si hay detecciones no asociadas y tenemos menos de 2
        if len(self.tracks) < 2:
            for det in unmatched_detections[:2 - len(self.tracks)]:
                new_id = max(self.tracks.keys()) + 1 if self.tracks else 0
                track = PlayerTrack(new_id, det)
                self.tracks[new_id] = track
        
        # Eliminar tracks muy antiguos
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > config.track_buffer:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        return self.tracks
    
    def process_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[List[PlayerDetection], Dict[int, PlayerTrack]]:
        """Procesa frame completo"""
        # Detectar
        detections = self.detect(frame)
        
        # Actualizar tracks
        self.update_tracks(detections)
        
        # Incrementar contador
        self.frame_count += 1
        
        return detections, self.tracks
    
    def identify_players_by_color(self, frame: np.ndarray):
        """
        Identifica jugadores por color de uniforme.
        Útil para distinguir jugadores consistentemente.
        """
        for track in self.tracks.values():
            if not track.detections:
                continue
            
            last_det = track.detections[-1]
            x1, y1, x2, y2 = last_det.bbox
            
            # Extraer región del jugador
            player_roi = frame[y1:y2, x1:x2]
            
            if player_roi.size == 0:
                continue
            
            # Calcular color dominante (histograma en HSV)
            hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
            
            # Tomar parte central del torso (evitar piernas y cabeza)
            h, w = hsv_roi.shape[:2]
            torso = hsv_roi[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            
            if torso.size == 0:
                continue
            
            # Histograma del canal H (tono)
            hist = cv2.calcHist([torso], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist)
            
            # Guardar color dominante
            track.color = dominant_hue
    
    def get_player_positions_court(
        self, 
        calibration
    ) -> Dict[int, Tuple[float, float]]:
        """
        Obtiene posiciones de jugadores en coordenadas de cancha (metros).
        
        Returns:
            Dict mapping player_id a (x, y) en metros
        """
        positions = {}
        
        for track_id, track in self.tracks.items():
            if not track.is_valid or not track.detections:
                continue
            
            last_det = track.detections[-1]
            feet_px = np.array(last_det.feet_position)
            
            try:
                feet_court = calibration.pixel_to_court(feet_px)
                positions[track_id] = tuple(feet_court)
                
                # Actualizar en el track también
                track.positions_court.append(tuple(feet_court))
                
                # Calcular promedio móvil
                if len(track.positions_court) > 0:
                    recent = track.positions_court[-30:]  # Últimos 30 frames
                    track.avg_position = (
                        np.mean([p[0] for p in recent]),
                        np.mean([p[1] for p in recent])
                    )
            except:
                pass
        
        return positions
    
    def draw_detections(
        self, 
        frame: np.ndarray,
        detections: List[PlayerDetection],
        tracks: Optional[Dict[int, PlayerTrack]] = None
    ) -> np.ndarray:
        """Dibuja detecciones y tracks en el frame"""
        overlay = frame.copy()
        
        if tracks is None:
            tracks = self.tracks
        
        # Dibujar bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color según lado de cancha
            if det.side == PlayerSide.NEAR:
                color = (255, 0, 0)  # Azul
            elif det.side == PlayerSide.FAR:
                color = (0, 0, 255)  # Rojo
            else:
                color = (0, 255, 0)  # Verde
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # ID y confianza
            if det.player_id is not None:
                text = f"P{det.player_id} ({det.confidence:.2f})"
            else:
                text = f"{det.confidence:.2f}"
            
            cv2.putText(
                overlay, text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            
            # Marcar posición de pies
            feet_x, feet_y = det.feet_position
            cv2.circle(overlay, (int(feet_x), int(feet_y)), 3, color, -1)
        
        # Información de tracks
        y_offset = 30
        for track_id, track in tracks.items():
            if not track.is_valid:
                continue
            
            text = f"Player {track_id}: {track.side.value}"
            if track.color is not None:
                text += f" (Hue: {track.color}°)"
            
            cv2.putText(
                overlay, text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            y_offset += 25
        
        return overlay