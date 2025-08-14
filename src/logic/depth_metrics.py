"""
Cálculo de métricas de profundidad y zonas de impacto.
Analiza dónde caen las pelotas y genera estadísticas por zona.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..config import config


@dataclass
class DepthStats:
    """Estadísticas de profundidad para un jugador"""
    total_shots: int = 0
    
    # Conteos por zona
    shots_short: int = 0      # Zona corta (0-3m desde red)
    shots_medium: int = 0     # Zona media (3-6m desde red)
    shots_deep: int = 0       # Zona profunda (6m+ desde red)
    
    # Porcentajes
    percent_short: float = 0.0
    percent_medium: float = 0.0
    percent_deep: float = 0.0
    
    # Métricas adicionales
    avg_depth: float = 0.0    # Profundidad promedio en metros
    consistency: float = 0.0   # Consistencia (1 - coef. variación)
    
    # Por tipo de golpe
    forehand_depths: List[float] = field(default_factory=list)
    backhand_depths: List[float] = field(default_factory=list)
    serve_depths: List[float] = field(default_factory=list)
    
    def update_percentages(self):
        """Actualiza los porcentajes basado en conteos"""
        if self.total_shots > 0:
            self.percent_short = (self.shots_short / self.total_shots) * 100
            self.percent_medium = (self.shots_medium / self.total_shots) * 100
            self.percent_deep = (self.shots_deep / self.total_shots) * 100


@dataclass
class ZoneImpact:
    """Información de impacto en una zona específica"""
    zone_name: str
    position: Tuple[float, float]  # Posición en metros
    frame_idx: int
    player_id: int
    stroke_type: str
    ball_speed: float
    resulted_in_error: bool = False
    resulted_in_winner: bool = False


class DepthAnalyzer:
    """
    Analizador de profundidad y zonas de impacto.
    Calcula métricas detalladas sobre dónde caen las pelotas.
    """
    
    def __init__(self):
        # Estadísticas por jugador
        self.player_stats: Dict[int, DepthStats] = defaultdict(DepthStats)
        
        # Registro de todos los impactos
        self.all_impacts: List[ZoneImpact] = []
        
        # Mapa de calor (grid de la cancha)
        self.heatmap_resolution = config.heatmap_resolution
        self.heatmap_data: Dict[int, np.ndarray] = {}  # Por jugador
        
        # Zonas personalizadas además de las estándar
        self.custom_zones = {
            "corners": self._define_corner_zones(),
            "service_boxes": self._define_service_boxes(),
            "alleys": self._define_alleys()
        }
        
        # Cache de cálculos
        self._depth_cache: Dict[Tuple[float, float], str] = {}
    
    def _define_corner_zones(self) -> List[Dict]:
        """Define las 4 esquinas de la cancha"""
        corner_size = 2.0  # 2 metros cuadrados
        
        return [
            {
                "name": "corner_near_left",
                "bounds": (0, 0, corner_size, corner_size)
            },
            {
                "name": "corner_near_right",
                "bounds": (config.singles_width - corner_size, 0,
                          config.singles_width, corner_size)
            },
            {
                "name": "corner_far_left",
                "bounds": (0, config.court_length - corner_size,
                          corner_size, config.court_length)
            },
            {
                "name": "corner_far_right",
                "bounds": (config.singles_width - corner_size,
                          config.court_length - corner_size,
                          config.singles_width, config.court_length)
            }
        ]
    
    def _define_service_boxes(self) -> List[Dict]:
        """Define los cuadros de servicio"""
        service_width = config.singles_width / 2
        service_length = config.baseline_to_serviceline
        
        return [
            {
                "name": "service_near_left",
                "bounds": (0, 0, service_width, service_length)
            },
            {
                "name": "service_near_right",
                "bounds": (service_width, 0, config.singles_width, service_length)
            },
            {
                "name": "service_far_left",
                "bounds": (0, config.court_length - service_length,
                          service_width, config.court_length)
            },
            {
                "name": "service_far_right",
                "bounds": (service_width, config.court_length - service_length,
                          config.singles_width, config.court_length)
            }
        ]
    
    def _define_alleys(self) -> List[Dict]:
        """Define los pasillos laterales (para dobles)"""
        alley_width = (config.court_width - config.singles_width) / 2
        
        return [
            {
                "name": "alley_left",
                "bounds": (-alley_width, 0, 0, config.court_length)
            },
            {
                "name": "alley_right",
                "bounds": (config.singles_width, 0,
                          config.singles_width + alley_width, config.court_length)
            }
        ]
    
    def add_impact(
        self,
        position: Tuple[float, float],
        frame_idx: int,
        player_id: int,
        stroke_type: str,
        ball_speed: float,
        resulted_in_error: bool = False,
        resulted_in_winner: bool = False
    ) -> ZoneImpact:
        """
        Registra un impacto de pelota.
        
        Args:
            position: Posición del impacto en metros
            frame_idx: Frame donde ocurrió
            player_id: ID del jugador que golpeó
            stroke_type: Tipo de golpe
            ball_speed: Velocidad de la pelota
            resulted_in_error: Si resultó en error
            resulted_in_winner: Si resultó en winner
            
        Returns:
            Objeto ZoneImpact con la información
        """
        # Determinar zona de profundidad
        zone_name = self.get_depth_zone(position)
        
        # Crear registro de impacto
        impact = ZoneImpact(
            zone_name=zone_name,
            position=position,
            frame_idx=frame_idx,
            player_id=player_id,
            stroke_type=stroke_type,
            ball_speed=ball_speed,
            resulted_in_error=resulted_in_error,
            resulted_in_winner=resulted_in_winner
        )
        
        # Guardar impacto
        self.all_impacts.append(impact)
        
        # Actualizar estadísticas del jugador
        self._update_player_stats(player_id, position, zone_name, stroke_type)
        
        # Actualizar heatmap
        self._update_heatmap(player_id, position)
        
        return impact
    
    def get_depth_zone(self, position: Tuple[float, float]) -> str:
        """
        Determina la zona de profundidad para una posición.
        
        Args:
            position: (x, y) en metros
            
        Returns:
            Nombre de la zona ("corta", "media", "profunda")
        """
        # Verificar cache
        if position in self._depth_cache:
            return self._depth_cache[position]
        
        x, y = position
        
        # Determinar lado de la cancha
        if y < config.court_length / 2:
            # Lado cercano - medir desde su línea de fondo
            distance_from_baseline = y
        else:
            # Lado lejano - medir desde su línea de fondo
            distance_from_baseline = config.court_length - y
        
        # Clasificar por distancia
        zone = "profunda"  # Por defecto
        
        for zone_name, (min_dist, max_dist) in config.depth_zones.items():
            if min_dist <= distance_from_baseline < max_dist:
                zone = zone_name
                break
        
        # Guardar en cache
        self._depth_cache[position] = zone
        
        return zone
    
    def _update_player_stats(
        self,
        player_id: int,
        position: Tuple[float, float],
        zone_name: str,
        stroke_type: str
    ):
        """Actualiza estadísticas del jugador"""
        stats = self.player_stats[player_id]
        
        # Incrementar contador total
        stats.total_shots += 1
        
        # Incrementar contador de zona
        if zone_name == "corta":
            stats.shots_short += 1
        elif zone_name == "media":
            stats.shots_medium += 1
        else:
            stats.shots_deep += 1
        
        # Calcular profundidad
        _, y = position
        if y < config.court_length / 2:
            depth = y
        else:
            depth = config.court_length - y
        
        # Guardar por tipo de golpe
        if stroke_type == "forehand":
            stats.forehand_depths.append(depth)
        elif stroke_type == "backhand":
            stats.backhand_depths.append(depth)
        elif stroke_type == "serve":
            stats.serve_depths.append(depth)
        
        # Actualizar profundidad promedio
        all_depths = (stats.forehand_depths + 
                     stats.backhand_depths + 
                     stats.serve_depths)
        
        if all_depths:
            stats.avg_depth = np.mean(all_depths)
            
            # Calcular consistencia
            if len(all_depths) > 1:
                std_dev = np.std(all_depths)
                if stats.avg_depth > 0:
                    coef_variation = std_dev / stats.avg_depth
                    stats.consistency = max(0, 1 - coef_variation)
        
        # Actualizar porcentajes
        stats.update_percentages()
    
    def _update_heatmap(self, player_id: int, position: Tuple[float, float]):
        """Actualiza el mapa de calor del jugador"""
        if player_id not in self.heatmap_data:
            self.heatmap_data[player_id] = np.zeros(self.heatmap_resolution)
        
        x, y = position
        
        # Convertir a índices del grid
        h_bins, w_bins = self.heatmap_resolution
        
        x_idx = int((x / config.singles_width) * w_bins)
        y_idx = int((y / config.court_length) * h_bins)
        
        # Clamp a límites
        x_idx = max(0, min(w_bins - 1, x_idx))
        y_idx = max(0, min(h_bins - 1, y_idx))
        
        # Incrementar contador
        self.heatmap_data[player_id][y_idx, x_idx] += 1
    
    def get_player_depth_stats(self, player_id: int) -> DepthStats:
        """Obtiene estadísticas de profundidad de un jugador"""
        return self.player_stats[player_id]
    
    def get_comparison_stats(self) -> Dict:
        """
        Obtiene comparación de estadísticas entre jugadores.
        """
        if len(self.player_stats) < 2:
            return {}
        
        p0_stats = self.player_stats[0]
        p1_stats = self.player_stats[1]
        
        return {
            "player_0": {
                "avg_depth": p0_stats.avg_depth,
                "percent_deep": p0_stats.percent_deep,
                "consistency": p0_stats.consistency,
                "total_shots": p0_stats.total_shots
            },
            "player_1": {
                "avg_depth": p1_stats.avg_depth,
                "percent_deep": p1_stats.percent_deep,
                "consistency": p1_stats.consistency,
                "total_shots": p1_stats.total_shots
            },
            "depth_difference": abs(p0_stats.avg_depth - p1_stats.avg_depth),
            "more_consistent": 0 if p0_stats.consistency > p1_stats.consistency else 1,
            "deeper_player": 0 if p0_stats.avg_depth > p1_stats.avg_depth else 1
        }
    
    def get_zone_effectiveness(self, player_id: int) -> Dict[str, Dict]:
        """
        Analiza efectividad por zona (winners vs errores).
        """
        zone_stats = defaultdict(lambda: {
            "total": 0,
            "winners": 0,
            "errors": 0,
            "effectiveness": 0.0
        })
        
        for impact in self.all_impacts:
            if impact.player_id == player_id:
                zone = impact.zone_name
                zone_stats[zone]["total"] += 1
                
                if impact.resulted_in_winner:
                    zone_stats[zone]["winners"] += 1
                elif impact.resulted_in_error:
                    zone_stats[zone]["errors"] += 1
        
        # Calcular efectividad
        for zone, stats in zone_stats.items():
            if stats["total"] > 0:
                stats["effectiveness"] = (
                    (stats["winners"] - stats["errors"]) / stats["total"]
                )
        
        return dict(zone_stats)
    
    def get_pattern_analysis(self, player_id: int) -> Dict:
        """
        Analiza patrones de juego basados en profundidad.
        """
        impacts = [i for i in self.all_impacts if i.player_id == player_id]
        
        if len(impacts) < 10:
            return {"insufficient_data": True}
        
        # Analizar secuencias
        sequences = []
        current_seq = []
        
        for i in range(len(impacts)):
            if i == 0 or impacts[i].frame_idx - impacts[i-1].frame_idx < 150:
                current_seq.append(impacts[i].zone_name)
            else:
                if len(current_seq) > 2:
                    sequences.append(current_seq)
                current_seq = [impacts[i].zone_name]
        
        if len(current_seq) > 2:
            sequences.append(current_seq)
        
        # Encontrar patrones comunes
        pattern_counts = defaultdict(int)
        
        for seq in sequences:
            for i in range(len(seq) - 2):
                pattern = tuple(seq[i:i+3])
                pattern_counts[pattern] += 1
        
        # Patrones más comunes
        common_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_sequences": len(sequences),
            "avg_sequence_length": np.mean([len(s) for s in sequences]),
            "common_patterns": [
                {
                    "pattern": list(pattern),
                    "count": count,
                    "frequency": count / len(sequences) if sequences else 0
                }
                for pattern, count in common_patterns
            ],
            "tendency": self._determine_tendency(impacts)
        }
    
    def _determine_tendency(self, impacts: List[ZoneImpact]) -> str:
        """Determina la tendencia general de juego"""
        if not impacts:
            return "unknown"
        
        zone_counts = defaultdict(int)
        for impact in impacts:
            zone_counts[impact.zone_name] += 1
        
        total = sum(zone_counts.values())
        
        deep_ratio = zone_counts.get("profunda", 0) / total if total > 0 else 0
        short_ratio = zone_counts.get("corta", 0) / total if total > 0 else 0
        
        if deep_ratio > 0.5:
            return "baseline_player"  # Jugador de fondo
        elif short_ratio > 0.3:
            return "net_player"  # Jugador de red
        else:
            return "all_court"  # Todo terreno
    
    def export_depth_report(self) -> Dict:
        """
        Exporta reporte completo de métricas de profundidad.
        """
        report = {
            "summary": {
                "total_impacts": len(self.all_impacts),
                "players_analyzed": len(self.player_stats)
            },
            "player_stats": {},
            "zone_distribution": {},
            "effectiveness_by_zone": {},
            "patterns": {}
        }
        
        # Estadísticas por jugador
        for player_id, stats in self.player_stats.items():
            report["player_stats"][f"player_{player_id}"] = {
                "total_shots": stats.total_shots,
                "avg_depth": round(stats.avg_depth, 2),
                "consistency": round(stats.consistency, 2),
                "distribution": {
                    "short": round(stats.percent_short, 1),
                    "medium": round(stats.percent_medium, 1),
                    "deep": round(stats.percent_deep, 1)
                },
                "by_stroke": {
                    "forehand_avg": round(np.mean(stats.forehand_depths), 2) 
                                   if stats.forehand_depths else 0,
                    "backhand_avg": round(np.mean(stats.backhand_depths), 2)
                                   if stats.backhand_depths else 0,
                    "serve_avg": round(np.mean(stats.serve_depths), 2)
                                if stats.serve_depths else 0
                }
            }
            
            # Efectividad por zona
            report["effectiveness_by_zone"][f"player_{player_id}"] = (
                self.get_zone_effectiveness(player_id)
            )
            
            # Análisis de patrones
            report["patterns"][f"player_{player_id}"] = (
                self.get_pattern_analysis(player_id)
            )
        
        # Distribución general de zonas
        zone_totals = defaultdict(int)
        for impact in self.all_impacts:
            zone_totals[impact.zone_name] += 1
        
        total_impacts = len(self.all_impacts)
        report["zone_distribution"] = {
            zone: {
                "count": count,
                "percentage": round((count / total_impacts * 100), 1)
                             if total_impacts > 0 else 0
            }
            for zone, count in zone_totals.items()
        }
        
        return report