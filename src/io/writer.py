"""
Módulo para escribir resultados en diferentes formatos.
Genera CSVs, JSONs y reportes con todas las métricas.
"""
import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import config


class ResultsWriter:
    """
    Escritor de resultados en múltiples formatos.
    Genera archivos estructurados con todas las métricas.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.outputs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para nombres únicos
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Paths de salida
        self.events_csv = self.output_dir / f"events_{self.timestamp}.csv"
        self.points_csv = self.output_dir / f"points_{self.timestamp}.csv"
        self.players_csv = self.output_dir / f"players_stats_{self.timestamp}.csv"
        self.summary_json = self.output_dir / f"match_summary_{self.timestamp}.json"
    
    def write_events(self, events: List[Dict[str, Any]]):
        """
        Escribe eventos (cada toque de pelota) a CSV.
        
        Columnas:
        - timestamp: Tiempo en segundos
        - frame_idx: Número de frame
        - event_type: Tipo de evento (stroke, error, winner)
        - player_id: ID del jugador
        - stroke_type: Tipo de golpe (forehand, backhand, serve)
        - ball_x, ball_y: Posición de la pelota en metros
        - ball_speed: Velocidad en m/s
        - depth_zone: Zona de profundidad
        - is_error: Si fue error
        - error_type: Tipo de error si aplica
        """
        with open(self.events_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'frame_idx', 'point_id', 'event_type',
                'player_id', 'stroke_type', 'stroke_number',
                'ball_x', 'ball_y', 'ball_speed',
                'depth_zone', 'is_error', 'error_type',
                'is_winner', 'is_forced_error'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                # Asegurar que todos los campos existen
                row = {field: event.get(field, '') for field in fieldnames}
                
                # Formatear números
                for num_field in ['timestamp', 'ball_x', 'ball_y', 'ball_speed']:
                    if row[num_field] != '':
                        row[num_field] = round(float(row[num_field]), 2)
                
                writer.writerow(row)
        
        print(f"✓ Eventos exportados a: {self.events_csv}")
    
    def write_points(self, points: List[Dict[str, Any]]):
        """
        Escribe resumen de puntos a CSV.
        
        Columnas:
        - point_id: ID del punto
        - start_time, end_time: Tiempos
        - duration_seconds: Duración
        - server_id: Quién sacó
        - winner_id: Quién ganó
        - total_strokes: Total de golpes
        - strokes_player_0, strokes_player_1: Golpes por jugador
        - end_reason: Razón del fin (winner, error, etc)
        - errors_first_3_p0, errors_first_3_p1: Errores en primeras 3 pelotas
        - forehand_count_p0, backhand_count_p0: Conteo por tipo de golpe
        """
        with open(self.points_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'point_id', 'start_time', 'end_time', 'duration_seconds',
                'server_id', 'winner_id', 'total_strokes',
                'strokes_player_0', 'strokes_player_1',
                'end_reason', 'serve_speed',
                'errors_first_3_p0', 'errors_first_3_p1',
                'forehand_count_p0', 'backhand_count_p0',
                'forehand_count_p1', 'backhand_count_p1',
                'net_errors_p0', 'long_errors_p0', 'wide_errors_p0',
                'net_errors_p1', 'long_errors_p1', 'wide_errors_p1',
                'forced_errors_p0', 'unforced_errors_p0',
                'forced_errors_p1', 'unforced_errors_p1'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for point in points:
                row = {field: point.get(field, 0) for field in fieldnames}
                
                # Formatear tiempos
                for time_field in ['start_time', 'end_time', 'duration_seconds']:
                    if row[time_field]:
                        row[time_field] = round(float(row[time_field]), 2)
                
                # Formatear velocidad de saque
                if row['serve_speed']:
                    row['serve_speed'] = round(float(row['serve_speed']), 1)
                
                writer.writerow(row)
        
        print(f"✓ Puntos exportados a: {self.points_csv}")
    
    def write_player_stats(self, player_stats: Dict[int, Dict[str, Any]]):
        """
        Escribe estadísticas agregadas por jugador.
        
        Incluye:
        - Totales de golpes por tipo
        - Errores por categoría
        - Promedios de profundidad
        - Efectividad por zona
        - Ubicación promedio
        """
        rows = []
        
        for player_id, stats in player_stats.items():
            row = {
                'player_id': player_id,
                
                # Golpes totales
                'total_strokes': stats.get('total_strokes', 0),
                'forehand_total': stats.get('forehand_total', 0),
                'backhand_total': stats.get('backhand_total', 0),
                'serve_total': stats.get('serve_total', 0),
                'volley_total': stats.get('volley_total', 0),
                
                # Errores
                'total_errors': stats.get('total_errors', 0),
                'unforced_errors': stats.get('unforced_errors', 0),
                'forced_errors': stats.get('forced_errors', 0),
                'net_errors': stats.get('net_errors', 0),
                'long_errors': stats.get('long_errors', 0),
                'wide_errors': stats.get('wide_errors', 0),
                
                # Errores en primeras 3 pelotas
                'errors_first_3_total': stats.get('errors_first_3', 0),
                
                # Winners
                'total_winners': stats.get('winners', 0),
                'aces': stats.get('aces', 0),
                
                # Profundidad
                'avg_depth_meters': round(stats.get('avg_depth', 0), 2),
                'percent_deep': round(stats.get('percent_deep', 0), 1),
                'percent_medium': round(stats.get('percent_medium', 0), 1),
                'percent_short': round(stats.get('percent_short', 0), 1),
                
                # Consistencia
                'consistency_score': round(stats.get('consistency', 0), 2),
                
                # Ubicación promedio
                'avg_position_x': round(stats.get('avg_position_x', 0), 2),
                'avg_position_y': round(stats.get('avg_position_y', 0), 2),
                
                # Efectividad
                'points_won': stats.get('points_won', 0),
                'points_played': stats.get('points_played', 0),
                'win_percentage': round(stats.get('win_percentage', 0), 1)
            }
            
            rows.append(row)
        
        # Escribir a CSV
        df = pd.DataFrame(rows)
        df.to_csv(self.players_csv, index=False)
        
        print(f"✓ Estadísticas de jugadores exportadas a: {self.players_csv}")
    
    def write_match_summary(self, summary_data: Dict[str, Any]):
        """
        Escribe resumen completo del partido en JSON.
        Incluye todas las métricas y metadatos.
        """
        # Asegurar que los arrays NumPy sean serializables
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        summary_data = convert_numpy(summary_data)
        
        # Agregar metadata
        summary_data['metadata'] = {
            'timestamp': self.timestamp,
            'version': '1.0',
            'config': {
                'fps': config.target_fps,
                'court_length': config.court_length,
                'court_width': config.singles_width,
                'depth_zones': config.depth_zones
            }
        }
        
        with open(self.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Resumen del partido exportado a: {self.summary_json}")
    
    def generate_heatmaps(
        self, 
        heatmap_data: Dict[int, np.ndarray],
        save_path: Optional[Path] = None
    ):
        """
        Genera y guarda mapas de calor de impactos.
        
        Args:
            heatmap_data: Dict con datos de heatmap por jugador
            save_path: Ruta donde guardar las imágenes
        """
        save_path = save_path or (config.figures_dir / f"heatmaps_{self.timestamp}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        for player_id, data in heatmap_data.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Heatmap de impactos
            im1 = ax1.imshow(
                data, 
                cmap='YlOrRd',
                aspect='equal',
                extent=[0, config.singles_width, 0, config.court_length]
            )
            ax1.set_title(f'Jugador {player_id} - Mapa de Impactos')
            ax1.set_xlabel('Ancho de cancha (m)')
            ax1.set_ylabel('Largo de cancha (m)')
            
            # Agregar líneas de cancha
            self._draw_court_lines(ax1)
            
            plt.colorbar(im1, ax=ax1, label='Número de impactos')
            
            # Heatmap suavizado
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(data, sigma=config.heatmap_sigma)
            
            im2 = ax2.imshow(
                smoothed,
                cmap='coolwarm',
                aspect='equal',
                extent=[0, config.singles_width, 0, config.court_length]
            )
            ax2.set_title(f'Jugador {player_id} - Densidad de Juego')
            ax2.set_xlabel('Ancho de cancha (m)')
            ax2.set_ylabel('Largo de cancha (m)')
            
            self._draw_court_lines(ax2)
            
            plt.colorbar(im2, ax=ax2, label='Densidad')
            
            plt.tight_layout()
            
            # Guardar
            heatmap_file = save_path / f"heatmap_player_{player_id}.png"
            plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Heatmap jugador {player_id} guardado en: {heatmap_file}")
    
    def _draw_court_lines(self, ax):
        """Dibuja líneas de cancha en el plot"""
        # Color de líneas
        line_color = 'white'
        line_width = 1
        
        # Líneas de fondo
        ax.axhline(y=0, color=line_color, linewidth=line_width)
        ax.axhline(y=config.court_length, color=line_color, linewidth=line_width)
        
        # Líneas laterales
        ax.axvline(x=0, color=line_color, linewidth=line_width)
        ax.axvline(x=config.singles_width, color=line_color, linewidth=line_width)
        
        # Red
        ax.axhline(y=config.court_length/2, color=line_color, 
                  linewidth=line_width*2, linestyle='--')
        
        # Líneas de servicio
        ax.axhline(y=config.baseline_to_serviceline, 
                  color=line_color, linewidth=line_width)
        ax.axhline(y=config.court_length - config.baseline_to_serviceline,
                  color=line_color, linewidth=line_width)
        
        # Línea central de servicio
        ax.plot([config.singles_width/2, config.singles_width/2],
               [0, config.baseline_to_serviceline],
               color=line_color, linewidth=line_width)
        ax.plot([config.singles_width/2, config.singles_width/2],
               [config.court_length - config.baseline_to_serviceline, config.court_length],
               color=line_color, linewidth=line_width)
    
    def generate_statistics_plots(
        self,
        player_stats: Dict[int, Dict[str, Any]],
        save_path: Optional[Path] = None
    ):
        """
        Genera gráficos de estadísticas comparativas.
        """
        save_path = save_path or (config.figures_dir / f"stats_{self.timestamp}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Preparar datos para plots
        players = list(player_stats.keys())
        if len(players) < 2:
            print("⚠ Se necesitan al menos 2 jugadores para comparación")
            return
        
        # 1. Comparación de tipos de golpes
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Golpes por tipo
        ax = axes[0, 0]
        stroke_types = ['forehand_total', 'backhand_total', 'serve_total']
        x = np.arange(len(stroke_types))
        width = 0.35
        
        for i, player_id in enumerate(players[:2]):
            values = [player_stats[player_id].get(s, 0) for s in stroke_types]
            ax.bar(x + i*width, values, width, label=f'Jugador {player_id}')
        
        ax.set_xlabel('Tipo de Golpe')
        ax.set_ylabel('Cantidad')
        ax.set_title('Distribución de Golpes')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(['Derecha', 'Revés', 'Saque'])
        ax.legend()
        
        # Errores por tipo
        ax = axes[0, 1]
        error_types = ['net_errors', 'long_errors', 'wide_errors']
        x = np.arange(len(error_types))
        
        for i, player_id in enumerate(players[:2]):
            values = [player_stats[player_id].get(e, 0) for e in error_types]
            ax.bar(x + i*width, values, width, label=f'Jugador {player_id}')
        
        ax.set_xlabel('Tipo de Error')
        ax.set_ylabel('Cantidad')
        ax.set_title('Distribución de Errores')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(['Red', 'Largo', 'Ancho'])
        ax.legend()
        
        # Profundidad
        ax = axes[1, 0]
        depth_data = []
        labels = []
        
        for player_id in players[:2]:
            stats = player_stats[player_id]
            depth_data.append([
                stats.get('percent_short', 0),
                stats.get('percent_medium', 0),
                stats.get('percent_deep', 0)
            ])
            labels.append(f'Jugador {player_id}')
        
        x = np.arange(3)
        bottom = np.zeros(3)
        
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        zone_names = ['Corta', 'Media', 'Profunda']
        
        for i, data in enumerate(depth_data):
            ax.bar(labels[i], data, bottom=bottom[i] if i < len(bottom) else 0,
                  color=colors, label=zone_names if i == 0 else None)
        
        ax.set_ylabel('Porcentaje')
        ax.set_title('Distribución de Profundidad')
        ax.legend()
        
        # Resumen de efectividad
        ax = axes[1, 1]
        categories = ['Winners', 'Errores\nNo Forzados', 'Errores\nForzados']
        
        for i, player_id in enumerate(players[:2]):
            stats = player_stats[player_id]
            values = [
                stats.get('winners', 0),
                stats.get('unforced_errors', 0),
                stats.get('forced_errors', 0)
            ]
            x_pos = np.arange(len(categories)) + i*width
            ax.bar(x_pos, values, width, label=f'Jugador {player_id}')
        
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Cantidad')
        ax.set_title('Winners vs Errores')
        ax.set_xticks(np.arange(len(categories)) + width/2)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.suptitle('Análisis Comparativo del Partido', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Guardar
        stats_file = save_path / "comparative_stats.png"
        plt.savefig(stats_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráficos de estadísticas guardados en: {stats_file}")
    
    def export_all(
        self,
        events: List[Dict],
        points: List[Dict],
        player_stats: Dict[int, Dict],
        summary_data: Dict,
        heatmap_data: Optional[Dict[int, np.ndarray]] = None
    ):
        """
        Exporta todos los resultados de una vez.
        """
        print("\n" + "="*50)
        print("EXPORTANDO RESULTADOS")
        print("="*50)
        
        # CSVs principales
        self.write_events(events)
        self.write_points(points)
        self.write_player_stats(player_stats)
        
        # JSON de resumen
        self.write_match_summary(summary_data)
        
        # Visualizaciones
        if heatmap_data:
            self.generate_heatmaps(heatmap_data)
        
        self.generate_statistics_plots(player_stats)
        
        print("\n✅ Exportación completa")
        print(f"📁 Directorio de salida: {self.output_dir}")