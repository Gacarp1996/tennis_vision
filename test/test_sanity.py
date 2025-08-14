"""
Tests básicos para verificar que el sistema funciona.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.court.calibrate import CourtCalibratorBackView
from src.detect.ball_detector import BallDetector
from src.detect.player_detector import PlayerDetector
from src.logic.rally_fsm import RallyStateMachine


class TestBasicFunctionality(unittest.TestCase):
    """Tests básicos de funcionalidad"""
    
    def test_config_loads(self):
        """Verifica que la configuración carga correctamente"""
        self.assertIsNotNone(config)
        self.assertEqual(config.court_length, 23.77)
        self.assertEqual(config.singles_width, 8.23)
    
    def test_calibrator_init(self):
        """Verifica que el calibrador se inicializa"""
        calibrator = CourtCalibratorBackView()
        self.assertIsNotNone(calibrator)
        self.assertEqual(len(calibrator.court_singles_corners), 4)
    
    def test_ball_detector_init(self):
        """Verifica que el detector de pelota se inicializa"""
        detector = BallDetector(use_yolo=False)  # Sin YOLO para test rápido
        self.assertIsNotNone(detector)
        
        # Test con frame dummy
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)
    
    def test_rally_fsm(self):
        """Verifica que la máquina de estados funciona"""
        fsm = RallyStateMachine(fps=30)
        self.assertIsNotNone(fsm)
        
        # Test actualización
        state = fsm.update(
            frame_idx=0,
            ball_detected=False,
            ball_position=None,
            ball_speed=None
        )
        self.assertEqual(state.value, "idle")
    
    def test_court_conversion(self):
        """Test de conversión de coordenadas"""
        calibrator = CourtCalibratorBackView()
        
        # Crear calibración dummy
        test_corners_px = np.array([
            [100, 600],
            [700, 600],
            [600, 100],
            [200, 100]
        ], dtype=np.float32)
        
        calibration = calibrator.calibrate_from_keypoints(
            {
                "baseline_near_left": test_corners_px[0],
                "baseline_near_right": test_corners_px[1],
                "baseline_far_right": test_corners_px[2],
                "baseline_far_left": test_corners_px[3]
            },
            (720, 1280)
        )
        
        self.assertIsNotNone(calibration)
        
        # Test conversión
        point_px = np.array([400, 350])
        point_court = calibrator.pixel_to_court(point_px)
        self.assertEqual(len(point_court), 2)


class TestVideoProcessing(unittest.TestCase):
    """Tests de procesamiento de video"""
    
    def setUp(self):
        """Crear video de prueba"""
        self.test_video_path = Path("test_video.mp4")
        
        # Crear video dummy si no existe
        if not self.test_video_path.exists():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(self.test_video_path),
                fourcc, 30.0, (1280, 720)
            )
            
            # 30 frames negros
            for _ in range(30):
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Agregar algunas líneas blancas (simular cancha)
                cv2.line(frame, (100, 600), (1180, 600), (255, 255, 255), 2)
                cv2.line(frame, (100, 120), (1180, 120), (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
    
    def test_video_loads(self):
        """Verifica que el video se carga"""
        cap = cv2.VideoCapture(str(self.test_video_path))
        self.assertTrue(cap.isOpened())
        
        ret, frame = cap.read()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (720, 1280, 3))
        
        cap.release()
    
    def tearDown(self):
        """Limpiar archivos de prueba"""
        if self.test_video_path.exists():
            self.test_video_path.unlink()


if __name__ == '__main__':
    unittest.main()