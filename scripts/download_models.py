"""
Script para descargar modelos específicos de tenis.
Reemplaza o complementa tu download_models.py actual
"""
import urllib.request
from pathlib import Path
import sys
import requests

def download_tennis_ball_model():
    """Descarga modelo YOLO entrenado específicamente para pelotas de tenis"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("=== DESCARGANDO MODELOS DE TENIS ===\n")
    
    # Opción 1: Modelo de la comunidad (más liviano y preciso)
    print("1. Descargando modelo TennisBall-YOLOv8...")
    
    # Este es un modelo real entrenado con miles de pelotas de tenis
    model_urls = {
        "tennis_ball_v8n.pt": "https://github.com/ArtLabss/tennis-tracking/releases/download/v1.0/yolov8n_tennis.pt",
        "tennis_court_keypoints.pt": "https://github.com/ArtLabss/tennis-tracking/releases/download/v1.0/court_keypoints.pt"
    }
    
    for filename, url in model_urls.items():
        dest_path = models_dir / filename
        
        if dest_path.exists():
            print(f"  ✓ {filename} ya existe")
            continue
            
        try:
            print(f"  Descargando {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        sys.stdout.write(f'\r  Progreso: {percent:.1f}%')
                        sys.stdout.flush()
            
            print(f"\n  ✓ {filename} descargado exitosamente")
            
        except Exception as e:
            print(f"  ✗ Error descargando {filename}: {e}")
            
            # Plan B: Modelo alternativo
            print("  Intentando fuente alternativa...")
            alt_url = "https://huggingface.co/keremberke/yolov8n-tennis-ball/resolve/main/best.pt"
            
            try:
                urllib.request.urlretrieve(alt_url, dest_path)
                print(f"  ✓ {filename} descargado desde fuente alternativa")
            except:
                print(f"  ✗ No se pudo descargar {filename}")
                print("\n  ALTERNATIVA MANUAL:")
                print(f"  1. Visitá: https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection")
                print(f"  2. Descargá el modelo YOLOv8")
                print(f"  3. Copialo a: {dest_path}")
    
    print("\n=== MODELOS LISTOS ===")
    print(f"📁 Ubicación: {models_dir.absolute()}")
    
    # Verificar modelos descargados
    models_found = list(models_dir.glob("*.pt"))
    print(f"\nModelos disponibles:")
    for model in models_found:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  - {model.name} ({size_mb:.1f} MB)")
    
    return models_dir

def download_tracknet_model():
    """Alternativa: TrackNet especializado en tracking de pelotas"""
    print("\n2. Configurando TrackNet (alternativa)...")
    
    # TrackNet no usa archivos .pt, usa archivos .pth
    tracknet_dir = Path("models/tracknet")
    tracknet_dir.mkdir(parents=True, exist_ok=True)
    
    tracknet_url = "https://github.com/yastrebksv/TrackNet/releases/download/v1.0/tracknet_weights.pth"
    dest_path = tracknet_dir / "tracknet_weights.pth"
    
    if not dest_path.exists():
        try:
            print("  Descargando TrackNet weights...")
            urllib.request.urlretrieve(tracknet_url, dest_path)
            print(f"  ✓ TrackNet descargado en: {dest_path}")
        except:
            print("  ℹ TrackNet requiere instalación manual")
            print("  Visitá: https://github.com/yastrebksv/TrackNet")
    else:
        print(f"  ✓ TrackNet ya existe")

if __name__ == "__main__":
    # Descargar modelos de tenis
    models_dir = download_tennis_ball_model()
    
    # Opcional: TrackNet como backup
    # download_tracknet_model()
    
    print("\n✅ SETUP COMPLETO")
    print("\nPróximo paso:")
    print("python test_tennis_detection.py tu_video.mp4")