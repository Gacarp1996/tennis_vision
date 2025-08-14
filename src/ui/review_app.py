"""
Aplicación Streamlit para revisar y corregir resultados del análisis.
Permite ajustar manualmente clasificaciones dudosas.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import cv2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class TennisReviewApp:
    """
    UI de Streamlit para revisar y editar resultados del análisis.
    """
    
    def __init__(self):
        st.set_page_config(
            page_title="Tennis Analysis Review",
            page_icon="🎾",
            layout="wide"
        )
        
        # Estado de la sesión
        if 'current_point' not in st.session_state:
            st.session_state.current_point = 0
        if 'edits_made' not in st.session_state:
            st.session_state.edits_made = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
    
    def load_data(self, results_dir: Path):
        """Carga los datos del análisis"""
        try:
            # Buscar archivos más recientes
            events_files = list(results_dir.glob("events_*.csv"))
            points_files = list(results_dir.glob("points_*.csv"))
            players_files = list(results_dir.glob("players_stats_*.csv"))
            summary_files = list(results_dir.glob("match_summary_*.json"))
            
            if not events_files:
                st.error("No se encontraron archivos de resultados")
                return False
            
            # Usar el más reciente
            latest_events = max(events_files, key=lambda p: p.stat().st_mtime)
            latest_points = max(points_files, key=lambda p: p.stat().st_mtime)
            latest_players = max(players_files, key=lambda p: p.stat().st_mtime)
            
            # Cargar DataFrames
            st.session_state.events_df = pd.read_csv(latest_events)
            st.session_state.points_df = pd.read_csv(latest_points)
            st.session_state.players_df = pd.read_csv(latest_players)
            
            # Cargar resumen JSON si existe
            if summary_files:
                latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
                with open(latest_summary, 'r') as f:
                    st.session_state.summary_data = json.load(f)
            
            st.session_state.data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return False
    
    def save_edits(self):
        """Guarda las ediciones realizadas"""
        if not st.session_state.edits_made:
            st.info("No hay cambios para guardar")
            return
        
        # Aplicar ediciones a los DataFrames
        for edit_key, edit_value in st.session_state.edits_made.items():
            edit_type, idx = edit_key.split('_', 1)
            idx = int(idx)
            
            if edit_type == 'event':
                for col, val in edit_value.items():
                    st.session_state.events_df.loc[idx, col] = val
            elif edit_type == 'point':
                for col, val in edit_value.items():
                    st.session_state.points_df.loc[idx, col] = val
        
        # Guardar archivos actualizados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        events_file = config.outputs_dir / f"events_corrected_{timestamp}.csv"
        points_file = config.outputs_dir / f"points_corrected_{timestamp}.csv"
        
        st.session_state.events_df.to_csv(events_file, index=False)
        st.session_state.points_df.to_csv(points_file, index=False)
        
        st.success(f"✅ Cambios guardados en:\n- {events_file.name}\n- {points_file.name}")
        st.session_state.edits_made = {}
    
    def show_summary_page(self):
        """Muestra página de resumen del partido"""
        st.title("📊 Resumen del Análisis")
        
        if not st.session_state.data_loaded:
            st.warning("Por favor, carga los datos primero")
            return
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        total_points = len(st.session_state.points_df)
        total_strokes = st.session_state.events_df[
            st.session_state.events_df['event_type'] == 'stroke'
        ].shape[0]
        
        with col1:
            st.metric("Total Puntos", total_points)
        
        with col2:
            st.metric("Total Golpes", total_strokes)
        
        with col3:
            avg_rally = st.session_state.points_df['total_strokes'].mean()
            st.metric("Rally Promedio", f"{avg_rally:.1f}")
        
        with col4:
            longest_rally = st.session_state.points_df['total_strokes'].max()
            st.metric("Rally Más Largo", longest_rally)
        
        st.divider()
        
        # Comparación de jugadores
        st.subheader("Comparación de Jugadores")
        
        players_data = st.session_state.players_df.to_dict('records')
        if len(players_data) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Jugador 0")
                p0 = players_data[0]
                st.metric("Golpes Totales", p0.get('total_strokes', 0))
                st.metric("Winners", p0.get('total_winners', 0))
                st.metric("Errores No Forzados", p0.get('unforced_errors', 0))
                st.metric("Profundidad Promedio", f"{p0.get('avg_depth_meters', 0):.2f}m")
            
            with col2:
                st.markdown("### Jugador 1")
                p1 = players_data[1]
                st.metric("Golpes Totales", p1.get('total_strokes', 0))
                st.metric("Winners", p1.get('total_winners', 0))
                st.metric("Errores No Forzados", p1.get('unforced_errors', 0))
                st.metric("Profundidad Promedio", f"{p1.get('avg_depth_meters', 0):.2f}m")
        
        # Gráficos
        st.divider()
        st.subheader("Visualizaciones")
        
        tab1, tab2, tab3 = st.tabs(["Distribución de Golpes", "Evolución del Partido", "Zonas de Profundidad"])
        
        with tab1:
            # Gráfico de tipos de golpes
            stroke_types = st.session_state.events_df[
                st.session_state.events_df['stroke_type'].notna()
            ]['stroke_type'].value_counts()
            
            fig = px.bar(
                x=stroke_types.index,
                y=stroke_types.values,
                labels={'x': 'Tipo de Golpe', 'y': 'Cantidad'},
                title="Distribución de Tipos de Golpes"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Rally length por punto
            fig = px.line(
                st.session_state.points_df,
                x='point_id',
                y='total_strokes',
                title="Duración de Rallies por Punto",
                labels={'total_strokes': 'Número de Golpes', 'point_id': 'Punto'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Distribución de profundidad
            depth_zones = st.session_state.events_df[
                st.session_state.events_df['depth_zone'].notna()
            ]['depth_zone'].value_counts()
            
            fig = px.pie(
                values=depth_zones.values,
                names=depth_zones.index,
                title="Distribución de Zonas de Profundidad"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_points_review_page(self):
        """Página para revisar y editar puntos"""
        st.title("🎾 Revisión de Puntos")
        
        if not st.session_state.data_loaded:
            st.warning("Por favor, carga los datos primero")
            return
        
        # Selector de punto
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Punto Anterior"):
                if st.session_state.current_point > 0:
                    st.session_state.current_point -= 1
        
        with col2:
            point_id = st.selectbox(
                "Seleccionar Punto",
                range(len(st.session_state.points_df)),
                index=st.session_state.current_point,
                format_func=lambda x: f"Punto {x}"
            )
            st.session_state.current_point = point_id
        
        with col3:
            if st.button("Punto Siguiente ➡️"):
                if st.session_state.current_point < len(st.session_state.points_df) - 1:
                    st.session_state.current_point += 1
        
        st.divider()
        
        # Datos del punto actual
        point_data = st.session_state.points_df.iloc[st.session_state.current_point]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Información del Punto")
            
            # Campos editables
            server_id = st.selectbox(
                "Servidor",
                [0, 1],
                index=int(point_data['server_id']),
                key=f"server_{point_id}"
            )
            
            winner_id = st.selectbox(
                "Ganador",
                [0, 1, -1],
                index=int(point_data['winner_id']) if not pd.isna(point_data['winner_id']) else 2,
                format_func=lambda x: f"Jugador {x}" if x >= 0 else "Sin ganador",
                key=f"winner_{point_id}"
            )
            
            end_reason = st.selectbox(
                "Razón del fin",
                ['winner', 'unforced_error', 'forced_error', 'ace', 'double_fault', 'unknown'],
                index=['winner', 'unforced_error', 'forced_error', 'ace', 'double_fault', 'unknown'].index(
                    point_data.get('end_reason', 'unknown')
                ),
                key=f"end_reason_{point_id}"
            )
            
            # Botón para guardar cambios del punto
            if st.button("💾 Guardar Cambios del Punto", key=f"save_point_{point_id}"):
                edits = {
                    'server_id': server_id,
                    'winner_id': winner_id if winner_id >= 0 else np.nan,
                    'end_reason': end_reason
                }
                st.session_state.edits_made[f'point_{point_id}'] = edits
                st.success("Cambios registrados (no guardados aún)")
        
        with col2:
            st.subheader("Estadísticas del Punto")
            st.metric("Duración", f"{point_data['duration_seconds']:.1f}s")
            st.metric("Total de Golpes", int(point_data['total_strokes']))
            st.metric("Golpes J0", int(point_data['strokes_player_0']))
            st.metric("Golpes J1", int(point_data['strokes_player_1']))
            
            if not pd.isna(point_data.get('serve_speed')):
                st.metric("Velocidad de Saque", f"{point_data['serve_speed']:.1f} m/s")
        
        st.divider()
        
        # Eventos del punto
        st.subheader("Golpes del Punto")
        
        point_events = st.session_state.events_df[
            st.session_state.events_df['point_id'] == point_id
        ]
        
        if not point_events.empty:
            # Tabla editable de eventos
            edited_events = st.data_editor(
                point_events[['stroke_number', 'player_id', 'stroke_type', 
                             'ball_speed', 'depth_zone', 'is_error', 'is_winner']],
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "stroke_number": st.column_config.NumberColumn("# Golpe", disabled=True),
                    "player_id": st.column_config.SelectboxColumn(
                        "Jugador",
                        options=[0, 1]
                    ),
                    "stroke_type": st.column_config.SelectboxColumn(
                        "Tipo",
                        options=['forehand', 'backhand', 'serve', 'volley', 'unknown']
                    ),
                    "ball_speed": st.column_config.NumberColumn("Velocidad (m/s)", format="%.1f"),
                    "depth_zone": st.column_config.SelectboxColumn(
                        "Zona",
                        options=['corta', 'media', 'profunda']
                    ),
                    "is_error": st.column_config.CheckboxColumn("Error"),
                    "is_winner": st.column_config.CheckboxColumn("Winner")
                }
            )
            
            # Guardar cambios de eventos
            if st.button("💾 Guardar Cambios de Eventos", key=f"save_events_{point_id}"):
                for idx, row in edited_events.iterrows():
                    original_idx = point_events.iloc[idx].name
                    changes = {}
                    
                    for col in edited_events.columns:
                        if row[col] != point_events.iloc[idx][col]:
                            changes[col] = row[col]
                    
                    if changes:
                        st.session_state.edits_made[f'event_{original_idx}'] = changes
                
                st.success("Cambios en eventos registrados")
    
    def show_errors_review_page(self):
        """Página para revisar y corregir clasificación de errores"""
        st.title("❌ Revisión de Errores")
        
        if not st.session_state.data_loaded:
            st.warning("Por favor, carga los datos primero")
            return
        
        # Filtrar eventos con errores
        errors_df = st.session_state.events_df[
            st.session_state.events_df['is_error'] == True
        ].copy()
        
        if errors_df.empty:
            st.info("No se encontraron errores para revisar")
            return
        
        st.subheader(f"Total de errores: {len(errors_df)}")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            player_filter = st.selectbox(
                "Filtrar por jugador",
                ["Todos", "Jugador 0", "Jugador 1"],
                key="error_player_filter"
            )
        
        with col2:
            error_type_filter = st.selectbox(
                "Filtrar por tipo",
                ["Todos", "net", "long", "wide", "unknown"],
                key="error_type_filter"
            )
        
        with col3:
            forced_filter = st.selectbox(
                "Forzado/No forzado",
                ["Todos", "Forzado", "No forzado"],
                key="error_forced_filter"
            )
        
        # Aplicar filtros
        if player_filter != "Todos":
            player_id = 0 if player_filter == "Jugador 0" else 1
            errors_df = errors_df[errors_df['player_id'] == player_id]
        
        if error_type_filter != "Todos":
            errors_df = errors_df[errors_df['error_type'] == error_type_filter]
        
        if forced_filter != "Todos":
            is_forced = forced_filter == "Forzado"
            errors_df = errors_df[errors_df.get('is_forced_error', False) == is_forced]
        
        st.divider()
        
        # Tabla de errores editable
        if not errors_df.empty:
            st.subheader("Errores para revisar")
            
            edited_errors = st.data_editor(
                errors_df[['frame_idx', 'point_id', 'player_id', 'stroke_type',
                          'error_type', 'is_forced_error']],
                use_container_width=True,
                column_config={
                    "frame_idx": st.column_config.NumberColumn("Frame", disabled=True),
                    "point_id": st.column_config.NumberColumn("Punto", disabled=True),
                    "player_id": st.column_config.SelectboxColumn(
                        "Jugador",
                        options=[0, 1]
                    ),
                    "stroke_type": st.column_config.SelectboxColumn(
                        "Golpe",
                        options=['forehand', 'backhand', 'serve', 'volley']
                    ),
                    "error_type": st.column_config.SelectboxColumn(
                        "Tipo Error",
                        options=['net', 'long', 'wide', 'double_fault', 'unknown']
                    ),
                    "is_forced_error": st.column_config.CheckboxColumn("Forzado")
                }
            )
            
            if st.button("💾 Guardar Cambios de Errores"):
                for idx, row in edited_errors.iterrows():
                    original_idx = errors_df.iloc[idx].name
                    changes = {}
                    
                    for col in ['player_id', 'stroke_type', 'error_type', 'is_forced_error']:
                        if col in row and row[col] != errors_df.iloc[idx].get(col):
                            changes[col] = row[col]
                    
                    if changes:
                        st.session_state.edits_made[f'event_{original_idx}'] = changes
                
                st.success("Cambios en errores registrados")
    
    def show_export_page(self):
        """Página para exportar datos corregidos"""
        st.title("💾 Exportar Resultados")
        
        if not st.session_state.data_loaded:
            st.warning("Por favor, carga los datos primero")
            return
        
        st.subheader("Cambios Pendientes")
        
        if st.session_state.edits_made:
            st.warning(f"Hay {len(st.session_state.edits_made)} cambios sin guardar")
            
            if st.button("💾 Guardar Todos los Cambios", type="primary"):
                self.save_edits()
        else:
            st.success("No hay cambios pendientes")
        
        st.divider()
        
        st.subheader("Exportar Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Exportar a Excel"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file = config.outputs_dir / f"tennis_analysis_{timestamp}.xlsx"
                
                with pd.ExcelWriter(excel_file) as writer:
                    st.session_state.events_df.to_excel(
                        writer, sheet_name='Eventos', index=False
                    )
                    st.session_state.points_df.to_excel(
                        writer, sheet_name='Puntos', index=False
                    )
                    st.session_state.players_df.to_excel(
                        writer, sheet_name='Jugadores', index=False
                    )
                
                st.success(f"✅ Exportado a: {excel_file.name}")
        
        with col2:
            if st.button("📄 Generar Reporte PDF"):
                st.info("Función de reporte PDF próximamente...")
        
        st.divider()
        
        st.subheader("Estadísticas de Edición")
        
        # Mostrar resumen de cambios realizados
        if hasattr(st.session_state, 'edit_history'):
            st.metric("Total de Ediciones", len(st.session_state.get('edit_history', [])))
    
    def run(self):
        """Ejecuta la aplicación Streamlit"""
        st.title("🎾 Tennis Analysis Review Tool")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuración")
            
            # Selector de directorio de resultados
            results_dir = st.text_input(
                "Directorio de Resultados",
                value=str(config.outputs_dir),
                key="results_dir"
            )
            
            if st.button("📂 Cargar Datos"):
                if self.load_data(Path(results_dir)):
                    st.success("✅ Datos cargados correctamente")
                else:
                    st.error("❌ Error al cargar datos")
            
            st.divider()
            
            # Navegación
            page = st.selectbox(
                "Página",
                ["📊 Resumen", "🎾 Revisar Puntos", "❌ Revisar Errores", "💾 Exportar"],
                key="page_selector"
            )
        
        # Contenido principal según página seleccionada
        if page == "📊 Resumen":
            self.show_summary_page()
        elif page == "🎾 Revisar Puntos":
            self.show_points_review_page()
        elif page == "❌ Revisar Errores":
            self.show_errors_review_page()
        elif page == "💾 Exportar":
            self.show_export_page()


def main():
    """Función principal para la UI"""
    app = TennisReviewApp()
    app.run()


if __name__ == "__main__":
    main()