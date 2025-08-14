"""
Detector y tracker de pelota de tenis.
Usa YOLOv8 con modelo específico o detección por color/movimiento como fallback.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import deque
import torch
from ultralytics import YOLO

from ..config import config


@dataclass
class BallDetection:
    """Detección de pelota en un frame"""
    x: float  # Centro x en píxeles
    y: float  # Centro y en píxeles
    confidence: float
    radius: float  # Radio estimado en píxeles
    frame_idx: int
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Convertir a bounding box [x1, y1, x2, y2]"""
        x1 = int(self.x - self.radius)
        y1 = int(self.y - self.radius)
        x2 = int(self.x + self.radius)
        y2 = int(self.y + self.radius)
        return (x1, y1, x2, y2)


class BallTrack:
    """Track de pelota a través de múltiples frames"""
    
    def __init__(self, track_id: int, initial_detection: BallDetection):
        self.track_id = track_id
        self.detections: deque = deque(maxlen=config.track_buffer)
        self.detections.append(initial_detection)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.velocity = np.array([0.0, 0.0])  # Velocidad en px/frame
        
    def update(self, detection: BallDetection):
        """Actualiza el track con nueva detección"""
        if len(self.detections) > 0:
            last = self.detections[-1]
            dt = detection.frame_idx - last.frame_idx
            if dt > 0:
                self.velocity = np.array([
                    (detection.x - last.x) / dt,
                    (detection.y - last.y) / dt
                ])
        
        self.detections.append(detection)
        self.time_since_update = 0
        self.hits += 1
    
    def predict(self, frame_idx: int) -> Tuple[float, float]:
        """Predice posición en frame dado"""
        if len(self.detections) == 0:
            return (0, 0)
        
        last = self.detections[-1]
        dt = frame_idx - last.frame_idx
        
        # Predicción lineal simple (mejorar con Kalman si es necesario)
        pred_x = last.x + self.velocity[0] * dt
        pred_y = last.y + self.velocity[1] * dt
        
        return (pred_x, pred_y)
    
    @property
    def is_valid(self) -> bool:
        """Track válido si tiene suficientes detecciones"""
        return self.hits >= config.min_track_length
    
    def get_trajectory(self, last_n: Optional[int] = None) -> List[Tuple[float, float]]:
        """Obtiene trayectoria de últimos N puntos"""
        if last_n is None:
            points = list(self.detections)
        else:
            points = list(self.detections)[-last_n:]
        
        return [(p.x, p.y) for p in points]


class BallDetector:
    """Detector de pelota de tenis"""
    
    def __init__(self, use_yolo: bool = True):
        self.use_yolo = use_yolo
        self.yolo_model = None
        self.tracks: Dict[int, BallTrack] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        if use_yolo:
            self._init_yolo()
        
        # Parámetros para detección por color (fallback)
        self.hsv_lower = np.array([25, 50, 50])   # Amarillo-verde
        self.hsv_upper = np.array([35, 255, 255])
        
        # Background subtractor para movimiento
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )
    
    def _init_yolo(self):
        """Inicializa modelo YOLO"""
        try:
            # Intentar cargar modelo específico de pelota de tenis
            model_path = config.project_root / "models" / "tennis_ball.pt"
            if model_path.exists():
                self.yolo_model = YOLO(str(model_path))
            else:
                # Usar modelo general y filtrar por clase 'sports ball'
                self.yolo_model = YOLO(config.yolo_model)
                print(f"Advertencia: Usando modelo general {config.yolo_model}")
                print("Para mejor precisión, entrena un modelo específico de pelota")
        except Exception as e:
            print(f"Error cargando YOLO: {e}")
            self.use_yolo = False
    
    def detect_yolo(self, frame: np.ndarray) -> List[BallDetection]:
        """Detecta pelotas usando YOLO"""
        if not self.yolo_model:
            return []
        
        results = self.yolo_model(
            frame, 
            conf=config.yolo_conf_threshold,
            iou=config.yolo_iou_threshold,
            classes=[32]  # Clase 'sports ball' en COCO
        )
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
                
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Centro y radio
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                radius = max(x2 - x1, y2 - y1) / 2
                
                # Filtrar por tamaño (pelota de tenis es pequeña)
                if 5 < radius < 30:  # Ajustar según resolución
                    detections.append(BallDetection(
                        x=cx, y=cy,
                        confidence=float(conf),
                        radius=radius,
                        frame_idx=self.frame_count
                    ))
        
        return detections
    
    def detect_color_motion(self, frame: np.ndarray) -> List[BallDetection]:
        """
        Detección por color y movimiento como fallback.
        Busca objetos pequeños amarillos en movimiento.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara de color
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Máscara de movimiento
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Combinar máscaras
        combined_mask = cv2.bitwise_and(color_mask, fg_mask)
        
        # Operaciones morfológicas para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            combined_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por tamaño
            if 20 < area < 500:  # Ajustar según resolución
                # Calcular círculo mínimo
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                # Verificar circularidad
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.7:  # Bastante circular
                        detections.append(BallDetection(
                            x=cx, y=cy,
                            confidence=circularity,  # Usar circularidad como confianza
                            radius=radius,
                            frame_idx=self.frame_count
                        ))
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[BallDetection]:
        """
        Detecta pelotas en el frame.
        Intenta YOLO primero, luego fallback a color/movimiento.
        """
        detections = []
        
        if self.use_yolo:
            detections = self.detect_yolo(frame)
        
        # Si YOLO no encuentra nada o no está disponible, usar fallback
        if len(detections) == 0:
            detections = self.detect_color_motion(frame)
        
        # Filtrar detecciones duplicadas (NMS simple)
        if len(detections) > 1:
            detections = self._non_max_suppression(detections)
        
        return detections
    
    def _non_max_suppression(
        self, 
        detections: List[BallDetection], 
        iou_threshold: float = 0.3
    ) -> List[BallDetection]:
        """Supresión no máxima para eliminar duplicados"""
        if len(detections) == 0:
            return []
        
        # Ordenar por confianza
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        keep = []
        for i, det1 in enumerate(detections):
            should_keep = True
            
            for det2 in keep:
                # Calcular distancia entre centros
                dist = np.sqrt((det1.x - det2.x)**2 + (det1.y - det2.y)**2)
                
                # Si están muy cerca, es duplicado
                if dist < (det1.radius + det2.radius) * 0.5:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det1)
        
        return keep
    
    def update_tracks(self, detections: List[BallDetection]) -> Dict[int, BallTrack]:
        """
        Actualiza tracks con nuevas detecciones.
        Asociación simple por distancia.
        """
        # Incrementar edad de tracks
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1
        
        if len(detections) == 0:
            return self.tracks
        
        # Asociar detecciones a tracks existentes
        unmatched_detections = list(detections)
        matched_tracks = set()
        
        for detection in detections:
            best_track = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # Predecir posición del track
                pred_x, pred_y = track.predict(self.frame_count)
                
                # Calcular distancia
                dist = np.sqrt((detection.x - pred_x)**2 + 
                             (detection.y - pred_y)**2)
                
                if dist < config.max_distance_threshold and dist < best_distance:
                    best_distance = dist
                    best_track = track_id
            
            if best_track is not None:
                # Actualizar track existente
                self.tracks[best_track].update(detection)
                matched_tracks.add(best_track)
                unmatched_detections.remove(detection)
        
        # Crear nuevos tracks para detecciones no asociadas
        for detection in unmatched_detections:
            new_track = BallTrack(self.next_track_id, detection)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Eliminar tracks antiguos sin actualización
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > config.track_buffer:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        return self.tracks
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[BallDetection], Dict[int, BallTrack]]:
        """
        Procesa un frame completo: detecta y trackea.
        
        Returns:
            Tupla de (detecciones, tracks actualizados)
        """
        # Detectar
        detections = self.detect(frame)
        
        # Trackear
        self.update_tracks(detections)
        
        # Incrementar contador
        self.frame_count += 1
        
        return detections, self.tracks
    
    def get_main_ball_track(self) -> Optional[BallTrack]:
        """
        Obtiene el track principal de la pelota.
        Generalmente el más largo y reciente.
        """
        valid_tracks = [t for t in self.tracks.values() if t.is_valid]
        
        if not valid_tracks:
            return None
        
        # Ordenar por hits (detecciones) y tiempo desde última actualización
        valid_tracks.sort(
            key=lambda t: (t.hits, -t.time_since_update), 
            reverse=True
        )
        
        return valid_tracks[0]
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[BallDetection],
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """Dibuja detecciones en el frame"""
        overlay = frame.copy()
        
        for det in detections:
            # Círculo alrededor de la pelota
            cv2.circle(
                overlay, 
                (int(det.x), int(det.y)), 
                int(det.radius),
                color, 2
            )
            
            # Confianza
            text = f"{det.confidence:.2f}"
            cv2.putText(
                overlay, text,
                (int(det.x - 20), int(det.y - det.radius - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        return overlay
    
    def draw_tracks(
        self, 
        frame: np.ndarray,
        tracks: Optional[Dict[int, BallTrack]] = None,
        trail_length: int = 10
    ) -> np.ndarray:
        """Dibuja tracks con estela"""
        overlay = frame.copy()
        
        if tracks is None:
            tracks = self.tracks
        
        for track_id, track in tracks.items():
            if not track.is_valid:
                continue
            
            # Obtener trayectoria
            trajectory = track.get_trajectory(trail_length)
            
            if len(trajectory) > 1:
                # Dibujar estela
                for i in range(1, len(trajectory)):
                    # Fade del color según antigüedad
                    alpha = i / len(trajectory)
                    color = (
                        int(255 * alpha),  # Azul  
                        int(255 * alpha),  # Verde
                        0                  # Rojo
                    )
                    
                    pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    
                    cv2.line(overlay, pt1, pt2, color, 2)
            
            # Marcar posición actual
            if len(track.detections) > 0:
                last = track.detections[-1]
                cv2.circle(
                    overlay,
                    (int(last.x), int(last.y)),
                    int(last.radius),
                    (0, 255, 0), 2
                )
                
                # ID del track
                text = f"Ball #{track_id}"
                cv2.putText(
                    overlay, text,
                    (int(last.x + 10), int(last.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
        
        return overlay