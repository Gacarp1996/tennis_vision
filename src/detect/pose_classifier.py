"""
Clasificador de pose para determinar tipo de golpe (derecha/revés).
Usa MediaPipe para detectar keypoints y clasificar.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

from ..config import config


class StrokeType(Enum):
    """Tipo de golpe"""
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE = "serve"
    VOLLEY = "volley"
    OVERHEAD = "overhead"
    UNKNOWN = "unknown"


class HandDominance(Enum):
    """Mano dominante del jugador"""
    RIGHT = "right"
    LEFT = "left"
    UNKNOWN = "unknown"


@dataclass
class PoseResult:
    """Resultado del análisis de pose"""
    keypoints: Dict[str, Tuple[float, float]]  # Nombre -> (x, y) normalizado
    stroke_type: StrokeType
    confidence: float
    shoulder_angle: float  # Ángulo de hombros respecto a la red
    hip_angle: float      # Ángulo de caderas
    hand_position: str    # "left", "right", "center"
    is_hitting: bool      # Si está en posición de golpe
    
    def get_hitting_hand(self) -> str:
        """Determina qué mano está golpeando"""
        if self.stroke_type == StrokeType.FOREHAND:
            return "right" if self.hand_position == "right" else "left"
        elif self.stroke_type == StrokeType.BACKHAND:
            return "left" if self.hand_position == "right" else "right"
        return "unknown"


class PoseClassifier:
    """Clasificador de pose para golpes de tenis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cache de dominancia por jugador
        self.player_dominance: Dict[int, HandDominance] = {}
        
        # Keypoints importantes para tenis
        self.tennis_keypoints = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]
    
    def process_player_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        player_id: Optional[int] = None
    ) -> Optional[PoseResult]:
        """
        Procesa región de un jugador para extraer pose y clasificar golpe.
        
        Args:
            frame: Frame completo
            bbox: Bounding box del jugador (x1, y1, x2, y2)
            player_id: ID del jugador para cache de dominancia
            
        Returns:
            PoseResult o None si no se detecta pose
        """
        x1, y1, x2, y2 = bbox
        
        # Extraer región del jugador con margen
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
        
        # Convertir a RGB para MediaPipe
        rgb_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(rgb_region)
        
        if not results.pose_landmarks:
            return None
        
        # Extraer keypoints
        keypoints = self._extract_keypoints(results.pose_landmarks)
        
        # Calcular ángulos corporales
        shoulder_angle = self._calculate_shoulder_angle(keypoints)
        hip_angle = self._calculate_hip_angle(keypoints)
        
        # Determinar posición de manos
        hand_position = self._determine_hand_position(keypoints)
        
        # Detectar si está golpeando
        is_hitting = self._detect_hitting_motion(keypoints)
        
        # Clasificar tipo de golpe
        stroke_type, confidence = self._classify_stroke(
            keypoints, 
            shoulder_angle,
            hand_position,
            is_hitting,
            player_id
        )
        
        return PoseResult(
            keypoints=keypoints,
            stroke_type=stroke_type,
            confidence=confidence,
            shoulder_angle=shoulder_angle,
            hip_angle=hip_angle,
            hand_position=hand_position,
            is_hitting=is_hitting
        )
    
    def _extract_keypoints(
        self, 
        landmarks
    ) -> Dict[str, Tuple[float, float]]:
        """Extrae keypoints relevantes de MediaPipe landmarks"""
        keypoints = {}
        
        for idx, landmark in enumerate(landmarks.landmark):
            if idx < len(self.mp_pose.PoseLandmark):
                name = self.mp_pose.PoseLandmark(idx).name
                if name in self.tennis_keypoints:
                    keypoints[name] = (landmark.x, landmark.y)
        
        return keypoints
    
    def _calculate_shoulder_angle(
        self, 
        keypoints: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Calcula ángulo de línea de hombros respecto a horizontal.
        0° = paralelo a la red, 90° = perpendicular
        """
        if 'LEFT_SHOULDER' not in keypoints or 'RIGHT_SHOULDER' not in keypoints:
            return 0.0
        
        left = keypoints['LEFT_SHOULDER']
        right = keypoints['RIGHT_SHOULDER']
        
        # Ángulo en grados
        angle = np.arctan2(right[1] - left[1], right[0] - left[0])
        return np.degrees(angle)
    
    def _calculate_hip_angle(
        self,
        keypoints: Dict[str, Tuple[float, float]]
    ) -> float:
        """Calcula ángulo de caderas similar a hombros"""
        if 'LEFT_HIP' not in keypoints or 'RIGHT_HIP' not in keypoints:
            return 0.0
        
        left = keypoints['LEFT_HIP']
        right = keypoints['RIGHT_HIP']
        
        angle = np.arctan2(right[1] - left[1], right[0] - left[0])
        return np.degrees(angle)
    
    def _determine_hand_position(
        self,
        keypoints: Dict[str, Tuple[float, float]]
    ) -> str:
        """
        Determina posición relativa de las manos.
        Importante para distinguir derecha de revés.
        """
        if 'LEFT_WRIST' not in keypoints or 'RIGHT_WRIST' not in keypoints:
            return "unknown"
        
        left_wrist = keypoints['LEFT_WRIST']
        right_wrist = keypoints['RIGHT_WRIST']
        
        # Centro del cuerpo (promedio de hombros)
        if 'LEFT_SHOULDER' in keypoints and 'RIGHT_SHOULDER' in keypoints:
            center_x = (keypoints['LEFT_SHOULDER'][0] + 
                       keypoints['RIGHT_SHOULDER'][0]) / 2
        else:
            center_x = 0.5
        
        # Ver qué mano está más adelante (menor x = más a la izquierda)
        if right_wrist[0] < center_x - 0.1 and left_wrist[0] > center_x:
            return "left"  # Mano derecha cruzada al lado izquierdo (backhand derecho)
        elif left_wrist[0] > center_x + 0.1 and right_wrist[0] < center_x:
            return "right"  # Mano izquierda cruzada al lado derecho (backhand zurdo)
        elif abs(right_wrist[0] - left_wrist[0]) < 0.1:
            return "center"  # Ambas manos juntas (revés a dos manos)
        elif right_wrist[0] > left_wrist[0]:
            return "right"  # Mano derecha adelante
        else:
            return "left"  # Mano izquierda adelante
    
    def _detect_hitting_motion(
        self,
        keypoints: Dict[str, Tuple[float, float]]
    ) -> bool:
        """
        Detecta si el jugador está en movimiento de golpe.
        Busca extensión de brazo y rotación de torso.
        """
        # Verificar extensión de al menos un brazo
        for side in ['LEFT', 'RIGHT']:
            shoulder_key = f'{side}_SHOULDER'
            elbow_key = f'{side}_ELBOW'
            wrist_key = f'{side}_WRIST'
            
            if all(k in keypoints for k in [shoulder_key, elbow_key, wrist_key]):
                shoulder = keypoints[shoulder_key]
                elbow = keypoints[elbow_key]
                wrist = keypoints[wrist_key]
                
                # Calcular ángulo del codo
                v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
                v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # Brazo extendido si ángulo > 120°
                    if angle > 120:
                        return True
        
        return False
    
    def _classify_stroke(
        self,
        keypoints: Dict[str, Tuple[float, float]],
        shoulder_angle: float,
        hand_position: str,
        is_hitting: bool,
        player_id: Optional[int] = None
    ) -> Tuple[StrokeType, float]:
        """
        Clasifica el tipo de golpe basado en la pose.
        
        Returns:
            Tupla de (tipo_golpe, confianza)
        """
        if not is_hitting:
            return (StrokeType.UNKNOWN, 0.0)
        
        # Detectar saque (ambos brazos arriba)
        if self._is_serve_position(keypoints):
            return (StrokeType.SERVE, 0.9)
        
        # Detectar volea (cerca de la red, movimiento corto)
        if self._is_volley_position(keypoints):
            return (StrokeType.VOLLEY, 0.7)
        
        # Clasificar derecha vs revés basado en rotación de hombros
        confidence = 0.8
        
        # Obtener dominancia del jugador si la conocemos
        dominance = self.player_dominance.get(
            player_id, 
            HandDominance.RIGHT  # Asumir diestro por defecto
        )
        
        # Para diestros
        if dominance == HandDominance.RIGHT:
            if config.forehand_shoulder_angle_min <= shoulder_angle <= config.forehand_shoulder_angle_max:
                if hand_position in ["right", "center"]:
                    return (StrokeType.FOREHAND, confidence)
                else:
                    return (StrokeType.BACKHAND, confidence)
            else:
                if hand_position in ["left", "center"]:
                    return (StrokeType.BACKHAND, confidence)
                else:
                    return (StrokeType.FOREHAND, confidence)
        
        # Para zurdos (invertir lógica)
        else:
            if config.forehand_shoulder_angle_min <= shoulder_angle <= config.forehand_shoulder_angle_max:
                if hand_position in ["left", "center"]:
                    return (StrokeType.FOREHAND, confidence)
                else:
                    return (StrokeType.BACKHAND, confidence)
            else:
                if hand_position in ["right", "center"]:
                    return (StrokeType.BACKHAND, confidence)
                else:
                    return (StrokeType.FOREHAND, confidence)
    
    def _is_serve_position(
        self,
        keypoints: Dict[str, Tuple[float, float]]
    ) -> bool:
        """Detecta posición de saque"""
        # En saque, al menos una mano está muy arriba
        for wrist_key in ['LEFT_WRIST', 'RIGHT_WRIST']:
            if wrist_key in keypoints:
                shoulder_key = wrist_key.replace('WRIST', 'SHOULDER')
                if shoulder_key in keypoints:
                    wrist_y = keypoints[wrist_key][1]
                    shoulder_y = keypoints[shoulder_key][1]
                    
                    # Mano por encima del hombro
                    if wrist_y < shoulder_y - 0.1:  # Y invertida (0 arriba)
                        return True
        return False
    
    def _is_volley_position(
        self,
        keypoints: Dict[str, Tuple[float, float]]
    ) -> bool:
        """
        Detecta posición de volea.
        Típicamente brazos adelante, cuerpo erguido.
        """
        # Verificar que ambas manos estén adelante
        if 'LEFT_WRIST' in keypoints and 'RIGHT_WRIST' in keypoints:
            if 'LEFT_SHOULDER' in keypoints and 'RIGHT_SHOULDER' in keypoints:
                avg_wrist_x = (keypoints['LEFT_WRIST'][0] + 
                              keypoints['RIGHT_WRIST'][0]) / 2
                avg_shoulder_x = (keypoints['LEFT_SHOULDER'][0] + 
                                 keypoints['RIGHT_SHOULDER'][0]) / 2
                
                # Manos significativamente adelante de hombros
                # (esto dependería de la orientación del jugador)
                if abs(avg_wrist_x - avg_shoulder_x) > 0.15:
                    return True
        
        return False
    
    def update_player_dominance(
        self, 
        player_id: int,
        dominance: HandDominance
    ):
        """Actualiza la mano dominante conocida de un jugador"""
        self.player_dominance[player_id] = dominance
    
    def infer_dominance_from_serves(
        self,
        serve_poses: List[PoseResult]
    ) -> HandDominance:
        """
        Infiere mano dominante analizando saques.
        En el saque, la mano dominante lanza la pelota.
        """
        if not serve_poses:
            return HandDominance.UNKNOWN
        
        right_count = 0
        left_count = 0
        
        for pose in serve_poses:
            if pose.stroke_type == StrokeType.SERVE:
                # Ver qué mano está más alta (lanzando la pelota)
                if 'LEFT_WRIST' in pose.keypoints and 'RIGHT_WRIST' in pose.keypoints:
                    left_y = pose.keypoints['LEFT_WRIST'][1]
                    right_y = pose.keypoints['RIGHT_WRIST'][1]
                    
                    if right_y < left_y:  # Y invertida
                        right_count += 1
                    else:
                        left_count += 1
        
        if right_count > left_count:
            return HandDominance.RIGHT
        elif left_count > right_count:
            return HandDominance.LEFT
        else:
            return HandDominance.UNKNOWN
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Dibuja skeleton y clasificación sobre el frame"""
        overlay = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Dibujar keypoints
        for name, (norm_x, norm_y) in pose_result.keypoints.items():
            # Convertir coordenadas normalizadas a píxeles
            x = int(x1 + norm_x * (x2 - x1))
            y = int(y1 + norm_y * (y2 - y1))
            
            # Color según parte del cuerpo
            if 'WRIST' in name or 'ELBOW' in name:
                color = (0, 255, 0)  # Verde para brazos
            elif 'SHOULDER' in name or 'HIP' in name:
                color = (255, 0, 0)  # Azul para torso
            else:
                color = (0, 0, 255)  # Rojo para piernas
            
            cv2.circle(overlay, (x, y), 3, color, -1)
        
        # Dibujar conexiones (skeleton)
        connections = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), 
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE')
        ]
        
        for conn in connections:
            if conn[0] in pose_result.keypoints and conn[1] in pose_result.keypoints:
                pt1 = pose_result.keypoints[conn[0]]
                pt2 = pose_result.keypoints[conn[1]]
                
                x1_pt = int(x1 + pt1[0] * (x2 - x1))
                y1_pt = int(y1 + pt1[1] * (y2 - y1))
                x2_pt = int(x1 + pt2[0] * (x2 - x1))
                y2_pt = int(y1 + pt2[1] * (y2 - y1))
                
                cv2.line(overlay, (x1_pt, y1_pt), (x2_pt, y2_pt), (255, 255, 0), 2)
        
        # Mostrar clasificación
        if pose_result.stroke_type != StrokeType.UNKNOWN:
            text = f"{pose_result.stroke_type.value} ({pose_result.confidence:.2f})"
            color = (0, 255, 0) if pose_result.is_hitting else (255, 255, 255)
            
            cv2.putText(
                overlay, text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )
            
            # Ángulos
            angle_text = f"Shoulder: {pose_result.shoulder_angle:.0f}°"
            cv2.putText(
                overlay, angle_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )
        
        return overlay