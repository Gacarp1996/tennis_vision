import os
from pathlib import Path

# Probar diferentes rutas
paths = [
    r"C:\Users\gaby_\OneDrive\Desktop\Analisis partidos",
    r"C:\Users\gaby_\Desktop\Analisis partidos",
    r"C:\Users\gaby_\OneDrive\Desktop\Analisis_partidos",
]

for p in paths:
    folder = Path(p)
    print(f"\nProbando: {p}")
    print(f"  ¿Existe? {folder.exists()}")
    
    if folder.exists():
        files = list(folder.iterdir())
        print(f"  Archivos: {len(files)}")
        
        # Buscar videos con cualquier extensión
        videos = []
        for ext in ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']:
            videos.extend(folder.glob(ext))
        
        print(f"  Videos encontrados: {len(videos)}")
        if videos:
            for v in videos[:3]:
                size_mb = v.stat().st_size / (1024*1024)
                print(f"    - {v.name} ({size_mb:.1f} MB)")