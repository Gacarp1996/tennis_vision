"""
Módulo de calibración de cancha para vista desde atrás.
Vista elevada a ~4m de altura, detrás de la línea de fondo.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ..config import config


@dataclass
class CalibrationResult:
    """Resultado de calibración de cancha"""
    homography_matrix: np.ndarray
    court_corners_pixels: np.ndarray  # 4 esquinas en píxeles
    court_corners_world: np.ndarray   # 4 esquinas en metros
    image_width: int
    image_height: int
    calibration_error: float  # Error RMS en píxeles
    view_type: str = "back_elevated"  # Tipo de vista
    
    def to_dict(self) -> dict:
        """Serializar para guardar en JSON"""
        return {
            "homography_matrix": self.homography_matrix.tolist(),
            "court_corners_pixels": self.court_corners_pixels.tolist(),
            "court_corners_world": self.court_corners_world.tolist(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "calibration_error": self.calibration_error,
            "view_type": self.view_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        """Cargar desde diccionario"""
        return cls(
            homography_matrix=np.array(data["homography_matrix"]),
            court_corners_pixels=np.array(data["court_corners_pixels"]),
            court_corners_world=np.array(data["court_corners_world"]),
            image_width=data["image_width"],
            image_height=data["image_height"],
            calibration_error=data["calibration_error"],
            view_type=data.get("view_type", "back_elevated")
        )


class CourtCalibratorBackView:
    """
    Calibrador de cancha para vista desde atrás elevada.
    Optimizado para cámara a ~4m de altura detrás de línea de fondo.
    """
    
    def __init__(self):
        self.config = config
        
        # Puntos de referencia para vista desde atrás
        # Origen (0,0) en esquina inferior izquierda del campo CERCANO
        # X positivo hacia la derecha, Y positivo hacia el fondo opuesto
        
        # Para singles (más fácil de detectar)
        self.court_singles_corners = np.array([
            [0, 0],                                      # Esquina inferior izq cerca
            [config.singles_width, 0],                  # Esquina inferior der cerca
            [config.singles_width, config.court_length], # Esquina superior der lejos
            [0, config.court_length],                    # Esquina superior izq lejos
        ], dtype=np.float32)
        
        # Puntos clave adicionales para calibración precisa
        self.key_points_world = {
            "baseline_near_left": [0, 0],
            "baseline_near_right": [config.singles_width, 0],
            "baseline_far_left": [0, config.court_length],
            "baseline_far_right": [config.singles_width, config.court_length],
            "service_near_left": [0, config.baseline_to_serviceline],
            "service_near_right": [config.singles_width, config.baseline_to_serviceline],
            "service_far_left": [0, config.court_length - config.baseline_to_serviceline],
            "service_far_right": [config.singles_width, config.court_length - config.baseline_to_serviceline],
            "net_left": [0, config.court_length / 2],
            "net_right": [config.singles_width, config.court_length / 2],
            "center_service_near": [config.singles_width / 2, config.baseline_to_serviceline],
            "center_service_far": [config.singles_width / 2, config.court_length - config.baseline_to_serviceline],
        }
        
        self.calibration: Optional[CalibrationResult] = None
        
        # Parámetros específicos para vista desde atrás
        self.perspective_scale = 1.5  # Las líneas lejanas se ven más pequeñas
    
    def detect_court_lines(self, frame: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Detecta líneas de la cancha agrupadas por orientación.
        Para vista desde atrás, esperamos líneas horizontales y convergentes.
        
        Returns:
            Dict con 'horizontal', 'vertical_left', 'vertical_right'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste para líneas blancas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Desenfoque adaptativo (menos en la parte inferior, más arriba)
        h, w = enhanced.shape
        blurred = enhanced.copy()
        for i in range(3):
            kernel_size = 3 + i * 2
            region_height = h // 3
            region = enhanced[i*region_height:(i+1)*region_height]
            blurred[i*region_height:(i+1)*region_height] = cv2.GaussianBlur(
                region, (kernel_size, kernel_size), 0
            )
        
        # Detección de bordes
        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
            apertureSize=3
        )
        
        # Aplicar máscara para zona de interés (la cancha está en el centro)
        mask = np.zeros_like(edges)
        roi_points = np.array([
            [w * 0.1, h * 0.3],   # Superior izquierda
            [w * 0.9, h * 0.3],   # Superior derecha
            [w * 0.95, h],        # Inferior derecha (más ancho abajo)
            [w * 0.05, h]         # Inferior izquierda
        ], dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        edges_masked = cv2.bitwise_and(edges, mask)
        
        # Detección de líneas con Hough probabilístico
        lines = cv2.HoughLinesP(
            edges_masked,
            rho=1,
            theta=np.pi/180,
            threshold=50,  # Más bajo para detectar líneas lejanas
            minLineLength=40,  # Más corto para líneas lejanas
            maxLineGap=30
        )
        
        if lines is None:
            return {"horizontal": [], "vertical_left": [], "vertical_right": []}
        
        lines = lines.reshape(-1, 4)
        
        # Clasificar líneas por orientación
        classified = {
            "horizontal": [],      # Líneas de fondo y servicio
            "vertical_left": [],   # Líneas laterales izquierdas
            "vertical_right": [],  # Líneas laterales derechas
            "net": []             # Línea de red
        }
        
        center_x = w / 2
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calcular ángulo
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Líneas horizontales (paralelas a líneas de fondo)
            if angle < 15 or angle > 165:
                classified["horizontal"].append(line)
                
                # Detectar si es la red (en el medio de la imagen)
                mid_y = (y1 + y2) / 2
                if h * 0.4 < mid_y < h * 0.6:
                    classified["net"].append(line)
            
            # Líneas laterales (convergen hacia el horizonte)
            elif 60 < angle < 120:
                mid_x = (x1 + x2) / 2
                if mid_x < center_x:
                    classified["vertical_left"].append(line)
                else:
                    classified["vertical_right"].append(line)
        
        return classified
    
    def find_court_keypoints(
        self, 
        lines_dict: Dict[str, List[np.ndarray]],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Encuentra puntos clave de la cancha desde las líneas detectadas.
        
        Returns:
            Dict con puntos clave detectados
        """
        h, w = frame_shape[:2]
        keypoints = {}
        
        # Encontrar línea de fondo cercana (más baja en imagen)
        baseline_near = self._find_baseline_near(lines_dict["horizontal"], h)
        if baseline_near is not None:
            # Intersecciones con laterales
            for line_left in lines_dict["vertical_left"]:
                pt = self._line_intersection(baseline_near, line_left)
                if pt is not None and self._is_valid_point(pt, w, h):
                    keypoints["baseline_near_left"] = pt
                    break
            
            for line_right in lines_dict["vertical_right"]:
                pt = self._line_intersection(baseline_near, line_right)
                if pt is not None and self._is_valid_point(pt, w, h):
                    keypoints["baseline_near_right"] = pt
                    break
        
        # Encontrar línea de fondo lejana (más alta en imagen)
        baseline_far = self._find_baseline_far(lines_dict["horizontal"], h)
        if baseline_far is not None:
            for line_left in lines_dict["vertical_left"]:
                pt = self._line_intersection(baseline_far, line_left)
                if pt is not None and self._is_valid_point(pt, w, h):
                    keypoints["baseline_far_left"] = pt
                    break
            
            for line_right in lines_dict["vertical_right"]:
                pt = self._line_intersection(baseline_far, line_right)
                if pt is not None and self._is_valid_point(pt, w, h):
                    keypoints["baseline_far_right"] = pt
                    break
        
        # Encontrar líneas de servicio
        service_lines = self._find_service_lines(lines_dict["horizontal"], h)
        for service_line in service_lines:
            y_avg = (service_line[1] + service_line[3]) / 2
            
            # Determinar si es cercana o lejana
            if y_avg > h * 0.6:  # Línea de servicio cercana
                for line_left in lines_dict["vertical_left"]:
                    pt = self._line_intersection(service_line, line_left)
                    if pt is not None and self._is_valid_point(pt, w, h):
                        keypoints["service_near_left"] = pt
                        break
                
                for line_right in lines_dict["vertical_right"]:
                    pt = self._line_intersection(service_line, line_right)
                    if pt is not None and self._is_valid_point(pt, w, h):
                        keypoints["service_near_right"] = pt
                        break
            
            elif y_avg < h * 0.4:  # Línea de servicio lejana
                for line_left in lines_dict["vertical_left"]:
                    pt = self._line_intersection(service_line, line_left)
                    if pt is not None and self._is_valid_point(pt, w, h):
                        keypoints["service_far_left"] = pt
                        break
                
                for line_right in lines_dict["vertical_right"]:
                    pt = self._line_intersection(service_line, line_right)
                    if pt is not None and self._is_valid_point(pt, w, h):
                        keypoints["service_far_right"] = pt
                        break
        
        return keypoints
    
    def _find_baseline_near(self, horizontal_lines: List[np.ndarray], height: int) -> Optional[np.ndarray]:
        """Encuentra la línea de fondo cercana (más baja en la imagen)"""
        if not horizontal_lines:
            return None
        
        # Buscar línea horizontal más baja
        lowest_line = None
        max_y = 0
        
        for line in horizontal_lines:
            y_avg = (line[1] + line[3]) / 2
            if y_avg > max_y and y_avg > height * 0.7:  # En el tercio inferior
                max_y = y_avg
                lowest_line = line
        
        return lowest_line
    
    def _find_baseline_far(self, horizontal_lines: List[np.ndarray], height: int) -> Optional[np.ndarray]:
        """Encuentra la línea de fondo lejana (más alta en la imagen)"""
        if not horizontal_lines:
            return None
        
        # Buscar línea horizontal más alta
        highest_line = None
        min_y = height
        
        for line in horizontal_lines:
            y_avg = (line[1] + line[3]) / 2
            if y_avg < min_y and y_avg < height * 0.3:  # En el tercio superior
                min_y = y_avg
                highest_line = line
        
        return highest_line
    
    def _find_service_lines(self, horizontal_lines: List[np.ndarray], height: int) -> List[np.ndarray]:
        """Encuentra las líneas de servicio (entre baselines y red)"""
        service_lines = []
        
        for line in horizontal_lines:
            y_avg = (line[1] + line[3]) / 2
            
            # Línea de servicio cercana (entre 60-75% de altura)
            if height * 0.6 < y_avg < height * 0.75:
                service_lines.append(line)
            
            # Línea de servicio lejana (entre 25-40% de altura)
            elif height * 0.25 < y_avg < height * 0.4:
                service_lines.append(line)
        
        return service_lines
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """Calcula intersección entre dos líneas"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        if -0.2 <= t <= 1.2:  # Margen más amplio para perspectiva
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return np.array([x, y])
        
        return None
    
    def _is_valid_point(self, point: np.ndarray, width: int, height: int) -> bool:
        """Verifica si un punto está dentro de límites razonables"""
        x, y = point
        margin = 50  # Margen para puntos cerca del borde
        return -margin <= x <= width + margin and -margin <= y <= height + margin
    
    def calibrate_from_keypoints(
        self,
        keypoints_px: Dict[str, np.ndarray],
        frame_shape: Tuple[int, int]
    ) -> CalibrationResult:
        """
        Calcula homografía desde puntos clave detectados.
        
        Args:
            keypoints_px: Puntos detectados en píxeles
            frame_shape: (height, width) del frame
        """
        h, w = frame_shape[:2]
        
        # Necesitamos al menos 4 puntos para homografía
        required_points = [
            "baseline_near_left", "baseline_near_right",
            "baseline_far_left", "baseline_far_right"
        ]
        
        # Verificar que tenemos los puntos mínimos
        missing = [p for p in required_points if p not in keypoints_px]
        if missing:
            # Intentar con puntos de servicio como alternativa
            alternative_points = [
                "service_near_left", "service_near_right",
                "service_far_left", "service_far_right"
            ]
            
            if all(p in keypoints_px for p in alternative_points):
                # Usar líneas de servicio en lugar de baselines
                src_points = np.array([
                    keypoints_px["service_near_left"],
                    keypoints_px["service_near_right"],
                    keypoints_px["service_far_right"],
                    keypoints_px["service_far_left"]
                ], dtype=np.float32)
                
                dst_points = np.array([
                    self.key_points_world["service_near_left"],
                    self.key_points_world["service_near_right"],
                    self.key_points_world["service_far_right"],
                    self.key_points_world["service_far_left"]
                ], dtype=np.float32)
            else:
                raise ValueError(f"Faltan puntos clave para calibración: {missing}")
        else:
            # Usar corners de baseline
            src_points = np.array([
                keypoints_px["baseline_near_left"],
                keypoints_px["baseline_near_right"],
                keypoints_px["baseline_far_right"],
                keypoints_px["baseline_far_left"]
            ], dtype=np.float32)
            
            dst_points = self.court_singles_corners
        
        # Calcular homografía
        homography, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            5.0
        )
        
        # Calcular error de reproyección
        src_reproj = cv2.perspectiveTransform(
            src_points.reshape(1, -1, 2),
            homography
        ).reshape(-1, 2)
        
        error = np.sqrt(np.mean((src_reproj - dst_points)**2))
        
        self.calibration = CalibrationResult(
            homography_matrix=homography,
            court_corners_pixels=src_points,
            court_corners_world=dst_points,
            image_width=w,
            image_height=h,
            calibration_error=error,
            view_type="back_elevated"
        )
        
        return self.calibration
    
    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        manual_points: Optional[Dict[str, np.ndarray]] = None
    ) -> CalibrationResult:
        """
        Calibra desde un frame con detección automática o puntos manuales.
        
        Args:
            frame: Frame del video
            manual_points: Puntos manuales si la detección falla
        """
        if manual_points is not None:
            return self.calibrate_from_keypoints(manual_points, frame.shape)
        
        # Detección automática
        lines_dict = self.detect_court_lines(frame)
        
        # Verificar que tenemos suficientes líneas
        total_lines = sum(len(lines) for lines in lines_dict.values())
        if total_lines < 4:
            raise ValueError(
                f"Solo se detectaron {total_lines} líneas. "
                "Se necesitan al menos 4. Intenta calibración manual."
            )
        
        # Encontrar puntos clave
        keypoints = self.find_court_keypoints(lines_dict, frame.shape)
        
        if len(keypoints) < 4:
            raise ValueError(
                f"Solo se encontraron {len(keypoints)} puntos clave. "
                "Se necesitan al menos 4. Intenta calibración manual."
            )
        
        return self.calibrate_from_keypoints(keypoints, frame.shape)
    
    def interactive_calibration(self, frame: np.ndarray) -> CalibrationResult:
        """
        Calibración interactiva donde el usuario hace clic en puntos clave.
        Útil cuando la detección automática falla.
        """
        print("\n=== CALIBRACIÓN INTERACTIVA ===")
        print("Haz clic en los siguientes puntos en orden:")
        print("1. Esquina inferior izquierda (baseline cercana)")
        print("2. Esquina inferior derecha (baseline cercana)")
        print("3. Esquina superior derecha (baseline lejana)")
        print("4. Esquina superior izquierda (baseline lejana)")
        print("\nPresiona 'r' para reiniciar, 'q' para cancelar")
        
        points = []
        point_names = [
            "baseline_near_left",
            "baseline_near_right",
            "baseline_far_right",
            "baseline_far_left"
        ]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append([x, y])
                    print(f"Punto {len(points)}/4: ({x}, {y})")
        
        cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibracion", mouse_callback)
        
        while True:
            display = frame.copy()
            
            # Dibujar puntos seleccionados
            for i, pt in enumerate(points):
                cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(
                    display, str(i+1),
                    (pt[0]+10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )
            
            # Dibujar líneas si hay suficientes puntos
            if len(points) >= 2:
                cv2.line(display, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
            if len(points) >= 3:
                cv2.line(display, tuple(points[1]), tuple(points[2]), (0, 255, 0), 2)
            if len(points) >= 4:
                cv2.line(display, tuple(points[2]), tuple(points[3]), (0, 255, 0), 2)
                cv2.line(display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
            
            cv2.imshow("Calibracion", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                raise ValueError("Calibración cancelada por usuario")
            elif key == ord('r'):
                points = []
                print("Puntos reiniciados")
            elif len(points) == 4 and key == ord(' '):
                break
        
        cv2.destroyAllWindows()
        
        # Crear diccionario de puntos
        manual_points = {}
        for i, name in enumerate(point_names):
            manual_points[name] = np.array(points[i], dtype=np.float32)
        
        return self.calibrate_from_keypoints(manual_points, frame.shape)
    
    def pixel_to_court(self, point_px: np.ndarray) -> np.ndarray:
        """Convierte punto en píxeles a coordenadas de cancha (metros)"""
        if self.calibration is None:
            raise ValueError("Cancha no calibrada")
        
        point_px = np.array(point_px, dtype=np.float32).reshape(1, 1, 2)
        point_court = cv2.perspectiveTransform(
            point_px,
            self.calibration.homography_matrix
        )
        return point_court.reshape(2)
    
    def court_to_pixel(self, point_court: np.ndarray) -> np.ndarray:
        """Convierte punto en coordenadas de cancha a píxeles"""
        if self.calibration is None:
            raise ValueError("Cancha no calibrada")
        
        H_inv = np.linalg.inv(self.calibration.homography_matrix)
        point_court = np.array(point_court, dtype=np.float32).reshape(1, 1, 2)
        point_px = cv2.perspectiveTransform(point_court, H_inv)
        return point_px.reshape(2)
    
    def draw_court_overlay(
        self,
        frame: np.ndarray,
        show_grid: bool = True,
        show_zones: bool = True
    ) -> np.ndarray:
        """
        Dibuja overlay de la cancha calibrada.
        
        Args:
            frame: Frame del video
            show_grid: Mostrar grid de cancha
            show_zones: Mostrar zonas de profundidad
        """
        if self.calibration is None:
            return frame
        
        overlay = frame.copy()
        
        # Dibujar contorno de cancha
        corners = self.calibration.court_corners_pixels.astype(int)
        cv2.polylines(overlay, [corners], True, (0, 255, 0), 2)
        
        if show_grid:
            # Dibujar líneas de cancha principales
            court_lines = [
                # Líneas de servicio
                ([0, config.baseline_to_serviceline],
                 [config.singles_width, config.baseline_to_serviceline]),
                ([0, config.court_length - config.baseline_to_serviceline],
                 [config.singles_width, config.court_length - config.baseline_to_serviceline]),
                # Red
                ([0, config.court_length/2],
                 [config.singles_width, config.court_length/2]),
                # Línea central de servicio
                ([config.singles_width/2, 0],
                 [config.singles_width/2, config.baseline_to_serviceline]),
                ([config.singles_width/2, config.court_length - config.baseline_to_serviceline],
                 [config.singles_width/2, config.court_length]),
            ]
            
            for pt1_world, pt2_world in court_lines:
                pt1_px = self.court_to_pixel(np.array(pt1_world))
                pt2_px = self.court_to_pixel(np.array(pt2_world))
                
                if (0 <= pt1_px[0] < frame.shape[1] and 0 <= pt1_px[1] < frame.shape[0] and
                    0 <= pt2_px[0] < frame.shape[1] and 0 <= pt2_px[1] < frame.shape[0]):
                    cv2.line(
                        overlay,
                        tuple(pt1_px.astype(int)),
                        tuple(pt2_px.astype(int)),
                        (0, 255, 255), 1
                    )
        
        if show_zones:
            # Dibujar zonas de profundidad con transparencia
            alpha = 0.3
            colors = {
                "corta": (255, 0, 0),    # Azul
                "media": (0, 255, 0),    # Verde
                "profunda": (0, 0, 255)  # Rojo
            }
            
            # Para cada lado de la cancha
            for side_y_start, side_y_end in [(0, config.court_length/2), 
                                            (config.court_length/2, config.court_length)]:
                for zone_name, (zone_start, zone_end) in config.depth_zones.items():
                    # Calcular posición de la zona desde la red
                    if side_y_start == 0:  # Lado cercano
                        y1 = config.court_length/2 - zone_end
                        y2 = config.court_length/2 - zone_start
                    else:  # Lado lejano
                        y1 = config.court_length/2 + zone_start
                        y2 = config.court_length/2 + zone_end
                    
                    # Convertir esquinas a píxeles
                    corners_world = np.array([
                        [0, y1],
                        [config.singles_width, y1],
                        [config.singles_width, y2],
                        [0, y2]
                    ], dtype=np.float32)
                    
                    corners_px = np.array([
                        self.court_to_pixel(corner) 
                        for corner in corners_world
                    ], dtype=np.int32)
                    
                    # Crear overlay con transparencia
                    zone_overlay = overlay.copy()
                    cv2.fillPoly(zone_overlay, [corners_px], colors[zone_name])
                    cv2.addWeighted(overlay, 1-alpha, zone_overlay, alpha, 0, overlay)
        
        # Mostrar información de calibración
        text = f"Calibration Error: {self.calibration.calibration_error:.2f}px"
        cv2.putText(
            overlay, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        text2 = "View: Back Elevated (4m)"
        cv2.putText(
            overlay, text2, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return overlay
    
    def save_calibration(self, filepath: Path):
        """Guarda calibración en archivo JSON"""
        if self.calibration is None:
            raise ValueError("No hay calibración para guardar")
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
    
    def load_calibration(self, filepath: Path):
        """Carga calibración desde archivo JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.calibration = CalibrationResult.from_dict(data)