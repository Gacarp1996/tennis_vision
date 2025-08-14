"""
Script para descargar modelos YOLO pre-entrenados.
"""
import urllib.request
from pathlib import Path
import sys

def download_file(url, dest_path):
    """Descarga archivo con progreso"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f'\rDescargando: {percent:.1f}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, download_progress)
    print()

def main():
    """Descarga modelos necesarios"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv8 nano (más rápido)
    print("Descargando YOLOv8n...")
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    yolo_path = models_dir / "yolov8n.pt"
    
    if not yolo_path.exists():
        download_file(yolo_url, yolo_path)
        print(f"✓ Modelo guardado en: {yolo_path}")
    else:
        print("✓ Modelo ya existe")
    
    print("\n" + "="*50)
    print("NOTA: Para mejor precisión en tenis, considera entrenar")
    print("modelos específicos con datos de tenis:")
    print("- tennis_ball.pt (para pelota)")
    print("- tennis_player.pt (para jugadores)")
    print("="*50)

if __name__ == "__main__":
    main()