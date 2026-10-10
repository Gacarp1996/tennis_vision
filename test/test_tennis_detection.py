"""
Script de prueba rápida para verificar detección de pelota de tenis.
Úsalo con cualquier video ATP que hayas descargado.
"""
import cv2
import sys
from pathlib import Path
import numpy as np
import time

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics no está instalado")
    print("Ejecutá: pip install ultralytics")
    sys.exit(1)

def test_tennis_detection(video_path, max_frames=300):
    """
    Prueba detección en los primeros segundos del video
    
    Args:
        video_path: Ruta al video ATP descargado
        max_frames: Frames a procesar (300 = 10 segundos @ 30fps)
    """
    print("=" * 60)
    print("TEST DE DETECCIÓN DE PELOTA DE TENIS")
    print("=" * 60)
    
    # Verificar video
    if not Path(video_path).exists():
        print(f"❌ No se encuentra el video: {video_path}")
        return
    
    # Buscar modelo de tenis
    model_paths = [
        "models/tennis_ball_v8n.pt",
        "models/yolov8n_tennis.pt",
        "models/best.pt",
        "models/yolov8n.pt"  # Fallback al genérico
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            print(f"✅ Usando modelo: {path}")
            break
    
    if not model_path:
        print("❌ No se encuentra ningún modelo. Ejecutá primero:")
        print("   python download_tennis_models.py")
        return
    
    # Cargar modelo
    print("\nCargando modelo...")
    model = YOLO(model_path)
    
    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video info:")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duración: {total_frames/fps:.1f} segundos")
    print(f"\n🔍 Procesando primeros {max_frames} frames...")
    
    # Estadísticas
    detections_count = 0
    frames_with_ball = 0
    processing_times = []
    ball_positions = []
    
    # Procesar frames
    for frame_idx in range(min(max_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Timer
        start_time = time.time()
        
        # Detectar
        results = model(frame, conf=0.25, verbose=False)
        
        # Tiempo de procesamiento
        process_time = time.time() - start_time
        processing_times.append(process_time)
        
        # Analizar detecciones
        ball_found = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # Obtener confianza
                    conf = float(box.conf[0])
                    
                    # Si es pelota de tenis (depende del modelo)
                    # Para modelo genérico, clase 32 es "sports ball"
                    # Para modelo específico, clase 0 es "tennis ball"
                    if conf > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        ball_positions.append((frame_idx, cx, cy))
                        ball_found = True
                        detections_count += 1
                        
                        # Dibujar en frame (opcional)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if ball_found:
            frames_with_ball += 1
        
        # Mostrar progreso
        if frame_idx % 30 == 0:
            avg_time = np.mean(processing_times) if processing_times else 0
            estimated_fps = 1/avg_time if avg_time > 0 else 0
            print(f"   Frame {frame_idx}/{max_frames} | "
                  f"Detecciones: {detections_count} | "
                  f"FPS procesamiento: {estimated_fps:.1f}")
        
        # Mostrar frame (opcional - comentar si no querés ventana)
        if frame_idx < 150:  # Solo mostrar primeros 5 segundos
            cv2.imshow('Detección Tennis', cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Resultados
    print("\n" + "=" * 60)
    print("RESULTADOS DEL TEST")
    print("=" * 60)
    
    detection_rate = (frames_with_ball / max_frames) * 100 if max_frames > 0 else 0
    avg_process_time = np.mean(processing_times) if processing_times else 0
    
    print(f"✅ Frames procesados: {len(processing_times)}")
    print(f"📊 Detecciones totales: {detections_count}")
    print(f"🎾 Frames con pelota: {frames_with_ball}/{max_frames} ({detection_rate:.1f}%)")
    print(f"⚡ Tiempo promedio/frame: {avg_process_time*1000:.1f}ms")
    print(f"📈 FPS de procesamiento: {1/avg_process_time:.1f}")
    
    # Análisis de calidad
    print("\n🎯 ANÁLISIS DE CALIDAD:")
    
    if detection_rate > 60:
        print("   ✅ EXCELENTE - Detección confiable")
    elif detection_rate > 30:
        print("   ⚠️ REGULAR - Necesita ajustes de threshold")
    else:
        print("   ❌ POBRE - Modelo no adecuado o video difícil")
    
    # Estimar tiempo para partido completo
    match_duration_hours = 2
    total_frames_match = match_duration_hours * 3600 * fps
    estimated_time_minutes = (total_frames_match * avg_process_time) / 60
    
    print(f"\n⏱️ ESTIMACIÓN PARTIDO COMPLETO (2 horas):")
    print(f"   Tiempo de procesamiento: {estimated_time_minutes:.1f} minutos")
    
    # Guardar posiciones para análisis
    if ball_positions:
        output_file = Path("test_ball_positions.txt")
        with open(output_file, 'w') as f:
            f.write("frame,x,y\n")
            for frame, x, y in ball_positions:
                f.write(f"{frame},{x:.1f},{y:.1f}\n")
        print(f"\n💾 Posiciones guardadas en: {output_file}")
    
    return detection_rate > 30  # True si funciona aceptablemente

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_tennis_detection.py <video.mp4>")
        print("\nEjemplo:")
        print("  python test_tennis_detection.py data/videos/atp_match.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Test con configuración por defecto
    success = test_tennis_detection(video_path, max_frames=300)
    
    if not success:
        print("\n💡 SUGERENCIAS:")
        print("1. Probá con otro modelo:")
        print("   - Ejecutá: python download_tennis_models.py")
        print("2. Ajustá el threshold de confianza")
        print("3. Probá con TrackNet (más preciso pero más lento)")

if __name__ == "__main__":
    main()