"""
Script principal para análisis de video de tenis.
Coordina todos los módulos y ejecuta el pipeline completo.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Optional
import json
import time

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from court.calibrate import CourtCalibratorBackView
from detect.ball_detector import BallDetector
from detect.player_detector import PlayerDetector
from detect.pose_classifier import PoseClassifier
from logic.rally_fsm import RallyStateMachine
from logic.error_classifier import ErrorClassifier
from logic.depth_metrics import DepthAnalyzer
from io.writer import ResultsWriter
from io.overlay import VideoOverlay


class TennisAnalyzer:
    """
    Analizador principal de video de tenis.
    Coordina todos los componentes del sistema.
    """
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        
        # Inicializar componentes
        print("Inicializando componentes...")
        
        # Detectores
        self.ball_detector = BallDetector(use_yolo=True)
        self.player_detector = PlayerDetector()
        self.pose_classifier = PoseClassifier()
        
        # Calibración
        self.court_calibrator = CourtCalibratorBackView()
        self.calibration = None
        
        # Lógica
        self.rally_fsm = RallyStateMachine(fps=config.target_fps)
        self.error_classifier = ErrorClassifier()
        self.depth_analyzer = DepthAnalyzer()
        
        # I/O
        self.writer = ResultsWriter()
        
        # Video
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Datos acumulados
        self.frame_data_list = []
        self.all_events = []
        self.player_stats = {0: {}, 1: {}}
        
        print(f"✓ Sistema inicializado")
        print(f"  Video: {video_path.name}")
        print(f"  FPS: {self.fps}")
        print(f"  Frames totales: {self.total_frames}")
    
    def calibrate_court(self, frame: Optional[np.ndarray] = None):
        """
        Calibra la cancha usando el primer frame o uno proporcionado.
        """
        print("\nCalibrando cancha...")
        
        if frame is None:
            # Usar primer frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("No se pudo leer el primer frame")
        
        # Intentar calibración automática
        try:
            self.calibration = self.court_calibrator.calibrate_from_frame(frame)
            print("✓ Calibración automática exitosa")
            print(f"  Error de calibración: {self.calibration.calibration_error:.2f}px")
        except Exception as e:
            print(f"⚠ Calibración automática falló: {e}")
            print("Iniciando calibración manual...")
            
            # Calibración interactiva
            self.calibration = self.court_calibrator.interactive_calibration(frame)
            print("✓ Calibración manual completada")
        
        # Guardar calibración
        calib_file = config.outputs_dir / f"calibration_{self.video_path.stem}.json"
        self.court_calibrator.save_calibration(calib_file)
        print(f"✓ Calibración guardada en: {calib_file}")
        
        # Configurar calibración en otros módulos
        self.player_detector.set_court_calibration(self.calibration)
        self.error_classifier.court_calibration = self.calibration
    
    def process_video(self, max_frames: Optional[int] = None):
        """
        Procesa el video completo o hasta max_frames.
        """
        print(f"\nProcesando video...")
        
        # Reset a inicio
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        frames_to_process = min(max_frames or self.total_frames, self.total_frames)
        
        start_time = time.time()
        
        while frame_idx < frames_to_process:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Procesar cada N frames según config
            if frame_idx % config.frame_skip == 0:
                frame_data = self.process_frame(frame, frame_idx)
                self.frame_data_list.append(frame_data)
            
            frame_idx += 1
            
            # Mostrar progreso
            if frame_idx % 100 == 0:
                progress = (frame_idx / frames_to_process) * 100
                elapsed = time.time() - start_time
                fps_processing = frame_idx / elapsed
                eta = (frames_to_process - frame_idx) / fps_processing
                
                print(f"  Progreso: {progress:.1f}% | "
                      f"FPS: {fps_processing:.1f} | "
                      f"ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n✓ Procesamiento completado")
        print(f"  Frames procesados: {frame_idx}")
        print(f"  Tiempo total: {total_time:.1f}s")
        print(f"  FPS promedio: {frame_idx/total_time:.1f}")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Procesa un frame individual.
        """
        frame_data = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / self.fps
        }
        
        # 1. Detectar pelota
        ball_detections, ball_tracks = self.ball_detector.process_frame(frame)
        main_ball_track = self.ball_detector.get_main_ball_track()
        
        ball_detected = main_ball_track is not None
        ball_position = None
        ball_speed = None
        
        if main_ball_track and len(main_ball_track.detections) > 0:
            last_detection = main_ball_track.detections[-1]
            ball_position_px = (last_detection.x, last_detection.y)
            
            # Convertir a coordenadas de cancha
            if self.calibration:
                try:
                    ball_position = self.court_calibrator.pixel_to_court(
                        np.array(ball_position_px)
                    )
                    ball_position = tuple(ball_position)
                except:
                    pass
            
            # Calcular velocidad
            if len(main_ball_track.detections) > 1:
                prev = main_ball_track.detections[-2]
                dt = (last_detection.frame_idx - prev.frame_idx) / self.fps
                if dt > 0:
                    dx = last_detection.x - prev.x
                    dy = last_detection.y - prev.y
                    ball_speed = np.sqrt(dx**2 + dy**2) / dt
                    
                    # Convertir a m/s si hay calibración
                    if self.calibration:
                        # Aproximar conversión de píxeles a metros
                        pixels_per_meter = frame.shape[1] / config.singles_width
                        ball_speed = ball_speed / pixels_per_meter
        
        frame_data['ball_detected'] = ball_detected
        frame_data['ball_position'] = ball_position
        frame_data['ball_speed'] = ball_speed
        
        # 2. Detectar jugadores
        player_detections, player_tracks = self.player_detector.process_frame(frame)
        
        players_data = []
        for track_id, track in player_tracks.items():
            if track.is_valid and track.detections:
                last_det = track.detections[-1]
                player_info = {
                    'id': track_id,
                    'bbox': last_det.bbox,
                    'position': None,
                    'current_stroke': None
                }
                
                # Posición en cancha
                if self.calibration:
                    try:
                        feet_pos = self.court_calibrator.pixel_to_court(
                            np.array(last_det.feet_position)
                        )
                        player_info['position'] = tuple(feet_pos)
                    except:
                        pass
                
                # Clasificar pose/golpe
                pose_result = self.pose_classifier.process_player_region(
                    frame, last_det.bbox, track_id
                )
                
                if pose_result and pose_result.is_hitting:
                    player_info['current_stroke'] = pose_result.stroke_type.value
                
                players_data.append(player_info)
        
        frame_data['players'] = players_data
        
        # 3. Actualizar máquina de estados del rally
        player_hitting = None
        stroke_type = None
        
        # Determinar quién está golpeando
        if ball_position and players_data:
            min_distance = float('inf')
            
            for player in players_data:
                if player['position']:
                    dist = np.linalg.norm(
                        np.array(ball_position) - np.array(player['position'])
                    )
                    if dist < min_distance and dist < config.max_player_ball_distance:
                        min_distance = dist
                        player_hitting = player['id']
                        stroke_type = player.get('current_stroke')
        
        rally_state = self.rally_fsm.update(
            frame_idx=frame_idx,
            ball_detected=ball_detected,
            ball_position=ball_position,
            ball_speed=ball_speed,
            player_hitting=player_hitting,
            stroke_type=stroke_type
        )
        
        frame_data['rally_state'] = rally_state.value
        frame_data['current_point'] = None
        
        if self.rally_fsm.current_point:
            frame_data['current_point'] = {
                'id': self.rally_fsm.current_point.point_id,
                'strokes': self.rally_fsm.current_point.total_strokes
            }
        
        # 4. Análisis de profundidad
        if ball_position and player_hitting is not None:
            impact = self.depth_analyzer.add_impact(
                position=ball_position,
                frame_idx=frame_idx,
                player_id=player_hitting,
                stroke_type=stroke_type or "unknown",
                ball_speed=ball_speed or 0
            )
            
            frame_data['depth_zone'] = impact.zone_name
        
        # 5. Clasificación de errores (si la pelota salió)
        # Esto requiere acumular trayectoria, lo hacemos en post-procesamiento
        
        return frame_data
    
    def post_process_analysis(self):
        """
        Post-procesamiento después de analizar todo el video.
        Genera estadísticas finales y métricas agregadas.
        """
        print("\nPost-procesamiento...")
        
        # 1. Procesar puntos completados
        points_data = []
        
        for point in self.rally_fsm.points:
            point_dict = {
                'point_id': point.point_id,
                'start_time': point.start_frame / self.fps,
                'end_time': point.end_frame / self.fps,
                'duration_seconds': point.duration_seconds,
                'server_id': point.server_id,
                'winner_id': point.winner_id,
                'total_strokes': point.total_strokes,
                'strokes_player_0': point.strokes_player_0,
                'strokes_player_1': point.strokes_player_1,
                'end_reason': point.end_reason.value,
                'errors_first_3_p0': point.errors_first_3_player_0,
                'errors_first_3_p1': point.errors_first_3_player_1
            }
            
            # Contar tipos de golpes por jugador
            for player_id in [0, 1]:
                player_strokes = point.get_player_strokes(player_id)
                
                forehand_count = sum(1 for s in player_strokes 
                                   if s.stroke_type == "forehand")
                backhand_count = sum(1 for s in player_strokes 
                                   if s.stroke_type == "backhand")
                
                point_dict[f'forehand_count_p{player_id}'] = forehand_count
                point_dict[f'backhand_count_p{player_id}'] = backhand_count
            
            points_data.append(point_dict)
        
        # 2. Generar eventos
        events_data = []
        
        for point in self.rally_fsm.points:
            for stroke in point.strokes:
                event = {
                    'timestamp': stroke.timestamp,
                    'frame_idx': stroke.frame_idx,
                    'point_id': point.point_id,
                    'event_type': 'stroke',
                    'player_id': stroke.player_id,
                    'stroke_type': stroke.stroke_type,
                    'stroke_number': stroke.stroke_number,
                    'ball_x': stroke.ball_position[0] if stroke.ball_position else None,
                    'ball_y': stroke.ball_position[1] if stroke.ball_position else None,
                    'ball_speed': stroke.ball_speed,
                    'depth_zone': stroke.depth_zone,
                    'is_error': stroke.is_error,
                    'is_winner': stroke.is_winner
                }
                events_data.append(event)
        
        # 3. Calcular estadísticas por jugador
        for player_id in [0, 1]:
            stats = self.calculate_player_stats(player_id)
            self.player_stats[player_id] = stats
        
        # 4. Generar resumen del partido
        match_summary = {
            'video_file': str(self.video_path),
            'total_frames': self.total_frames,
            'frames_analyzed': len(self.frame_data_list),
            'total_points': len(self.rally_fsm.points),
            'match_stats': self.rally_fsm.get_match_stats(),
            'depth_analysis': self.depth_analyzer.export_depth_report(),
            'player_comparison': self.depth_analyzer.get_comparison_stats()
        }
        
        # 5. Exportar todo
        self.writer.export_all(
            events=events_data,
            points=points_data,
            player_stats=self.player_stats,
            summary_data=match_summary,
            heatmap_data=self.depth_analyzer.heatmap_data
        )
        
        print("✓ Post-procesamiento completado")
    
    def calculate_player_stats(self, player_id: int) -> Dict:
        """
        Calcula estadísticas completas para un jugador.
        """
        stats = {
            'player_id': player_id,
            'total_strokes': 0,
            'forehand_total': 0,
            'backhand_total': 0,
            'serve_total': 0,
            'volley_total': 0,
            'total_errors': 0,
            'unforced_errors': 0,
            'forced_errors': 0,
            'net_errors': 0,
            'long_errors': 0,
            'wide_errors': 0,
            'winners': 0,
            'aces': 0,
            'points_won': 0,
            'points_played': len(self.rally_fsm.points)
        }
        
        # Procesar puntos
        for point in self.rally_fsm.points:
            if point.winner_id == player_id:
                stats['points_won'] += 1
            
            # Procesar golpes del jugador
            player_strokes = point.get_player_strokes(player_id)
            
            for stroke in player_strokes:
                stats['total_strokes'] += 1
                
                # Tipo de golpe
                if stroke.stroke_type == "forehand":
                    stats['forehand_total'] += 1
                elif stroke.stroke_type == "backhand":
                    stats['backhand_total'] += 1
                elif stroke.stroke_type == "serve":
                    stats['serve_total'] += 1
                elif stroke.stroke_type == "volley":
                    stats['volley_total'] += 1
                
                # Errores y winners
                if stroke.is_error:
                    stats['total_errors'] += 1
                    
                    if stroke.error_type:
                        # Clasificar tipo de error
                        # (Esto requeriría más lógica del ErrorClassifier)
                        pass
                
                if stroke.is_winner:
                    stats['winners'] += 1
        
        # Agregar estadísticas de profundidad
        depth_stats = self.depth_analyzer.get_player_depth_stats(player_id)
        stats['avg_depth'] = depth_stats.avg_depth
        stats['percent_deep'] = depth_stats.percent_deep
        stats['percent_medium'] = depth_stats.percent_medium
        stats['percent_short'] = depth_stats.percent_short
        stats['consistency'] = depth_stats.consistency
        
        # Calcular porcentaje de victoria
        if stats['points_played'] > 0:
            stats['win_percentage'] = (stats['points_won'] / stats['points_played']) * 100
        
        return stats
    
    def generate_overlay_video(self, skip_frames: int = 1):
        """
        Genera video con overlays de análisis.
        """
        print("\nGenerando video con overlay...")
        
        overlay_gen = VideoOverlay(self.video_path)
        overlay_gen.generate_overlay_video(
            self.frame_data_list,
            self.calibration,
            skip_frames=skip_frames
        )
    
    def run_full_analysis(
        self,
        max_frames: Optional[int] = None,
        generate_overlay: bool = True
    ):
        """
        Ejecuta el análisis completo del video.
        """
        print("\n" + "="*60)
        print("INICIANDO ANÁLISIS COMPLETO DE VIDEO DE TENIS")
        print("="*60)
        
        # 1. Calibrar cancha
        self.calibrate_court()
        
        # 2. Procesar video
        self.process_video(max_frames)
        
        # 3. Post-procesamiento
        self.post_process_analysis()
        
        # 4. Generar overlay (opcional)
        if generate_overlay:
            self.generate_overlay_video(skip_frames=2)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        
        # Mostrar resumen
        print("\nRESUMEN DEL ANÁLISIS:")
        print(f"  Total de puntos: {len(self.rally_fsm.points)}")
        print(f"  Total de golpes: {self.rally_fsm.total_strokes}")
        
        for player_id, stats in self.player_stats.items():
            print(f"\n  Jugador {player_id}:")
            print(f"    - Golpes totales: {stats.get('total_strokes', 0)}")
            print(f"    - Derechas: {stats.get('forehand_total', 0)}")
            print(f"    - Reveses: {stats.get('backhand_total', 0)}")
            print(f"    - Winners: {stats.get('winners', 0)}")
            print(f"    - Errores: {stats.get('total_errors', 0)}")
            print(f"    - Profundidad promedio: {stats.get('avg_depth', 0):.2f}m")
            print(f"    - % Pelotas profundas: {stats.get('percent_deep', 0):.1f}%")
        
        print(f"\n📁 Resultados guardados en: {config.outputs_dir}")
        
        return True


def main():
    """
    Función principal - CLI.
    """
    parser = argparse.ArgumentParser(
        description="Análisis automático de video de tenis"
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Ruta al archivo de video"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Número máximo de frames a procesar (para pruebas)"
    )
    
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="No generar video con overlay"
    )
    
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Exportar resultados a CSV"
    )
    
    parser.add_argument(
        "--run-ui",
        action="store_true",
        help="Ejecutar UI de revisión después del análisis"
    )
    
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Usar archivo de calibración existente"
    )
    
    args = parser.parse_args()
    
    # Verificar que el video existe
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: No se encuentra el video: {video_path}")
        sys.exit(1)
    
    # Crear analizador
    analyzer = TennisAnalyzer(video_path)
    
    # Cargar calibración si se proporciona
    if args.calibration_file:
        calib_path = Path(args.calibration_file)
        if calib_path.exists():
            print(f"Cargando calibración desde: {calib_path}")
            analyzer.court_calibrator.load_calibration(calib_path)
            analyzer.calibration = analyzer.court_calibrator.calibration
    
    # Ejecutar análisis
    try:
        analyzer.run_full_analysis(
            max_frames=args.max_frames,
            generate_overlay=not args.no_overlay
        )
        
        # Lanzar UI si se solicita
        if args.run_ui:
            print("\nLanzando UI de revisión...")
            import subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                str(Path(__file__).parent / "ui" / "review_app.py"),
                "--", 
                "--results-dir", str(config.outputs_dir)
            ])
        
    except KeyboardInterrupt:
        print("\n⚠ Análisis interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()