"""
Extractor de 100 frames para entrenar modelo
"""
import cv2
from pathlib import Path

# Tu video
VIDEO = r"C:\Users\gaby_\OneDrive\Desktop\Analisis partidos\Prueba tennis_vision.mp4"

# Carpeta de salida
OUTPUT_DIR = Path("frames_video1")
OUTPUT_DIR.mkdir(exist_ok=True)

# Abrir video
cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duracion_seg = total_frames / fps

print(f"📹 Video: {Path(VIDEO).name}")
print(f"   FPS: {fps}")
print(f"   Duración: {duracion_seg/60:.1f} minutos")
print(f"\n🎯 Extrayendo 100 frames...")

# Configuración
INICIO_SEG = 15  # Empezar en segundo 15
FRAMES_OBJETIVO = 100
INTERVALO_SEG = 3  # Un frame cada 3 segundos

frames_extraidos = 0

for i in range(FRAMES_OBJETIVO):
    # Calcular segundo actual
    segundo = INICIO_SEG + (i * INTERVALO_SEG)
    
    # Si nos pasamos del video, parar
    if segundo > duracion_seg:
        print(f"   ⚠️ Llegamos al final del video")
        break
    
    # Ir a ese segundo
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segundo * fps))
    ret, frame = cap.read()
    
    if ret:
        # Guardar frame
        filename = OUTPUT_DIR / f"frame_{segundo:05d}.jpg"
        cv2.imwrite(str(filename), frame)
        frames_extraidos += 1
        
        # Progreso
        if frames_extraidos % 10 == 0:
            print(f"   ✓ {frames_extraidos}/100 frames...")

cap.release()

print(f"\n✅ Completado!")
print(f"   Frames extraídos: {frames_extraidos}")
print(f"   Guardados en: {OUTPUT_DIR.absolute()}")
print(f"\n💡 Próximos pasos:")
print(f"   1. Subir carpeta '{OUTPUT_DIR}' a Roboflow")
print(f"   2. Etiquetar pelotas")
print(f"   3. Cuando tengas más videos, repetir proceso")