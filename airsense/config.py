"""Configuracion central de AirSense."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"

MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

# Resolucion de trabajo (el frame de la camara se reescala a esto)
FRAME_W, FRAME_H = 1280, 720

# Paleta de colores (BGR) para dibujar
PALETTE = [
    ("Blanco", (255, 255, 255)),
    ("Rojo", (60, 60, 255)),
    ("Naranja", (0, 150, 255)),
    ("Amarillo", (0, 230, 255)),
    ("Verde", (80, 220, 80)),
    ("Cian", (255, 220, 0)),
    ("Azul", (255, 120, 40)),
    ("Magenta", (230, 60, 230)),
]

BRUSH_MIN, BRUSH_MAX, BRUSH_DEFAULT = 2, 40, 8

# Tiempos (segundos) que hay que mantener un gesto para disparar su accion
HOLD_CLEAR = 1.2      # palma abierta -> limpiar lienzo
HOLD_SAVE = 1.0       # pulgar arriba -> guardar imagen
HOLD_UNDO = 0.8       # "llamame" (pulgar + menique) -> deshacer
HOLD_ERASER = 0.8     # "rock" (indice + menique) -> alternar borrador
DWELL_BUTTON = 0.7    # tiempo apuntando a un boton de la barra para activarlo

GESTURE_STABLE_FRAMES = 3   # inferencias seguidas con el mismo gesto para aceptarlo
DRAW_STOP_FRAMES = 6        # inferencias con otro gesto "dudoso" antes de cortar un trazo
# Recorte de la "cola" del trazo: al cerrar la mano, sacarla de la camara o ir a la barra
# superior, el dedo sigue detectado un instante y deja una raya. Al terminar el trazo se
# descartan los puntos de ese final:
STROKE_TAIL_TRIM = 0.15     # s antes de que el gesto empezara a dejar de ser "indice"
HAND_LOST_TRIM = 0.30       # s antes de perder la mano
TOOLBAR_TRIM_MAX = 0.8      # s maximos del tramo final subiendo hacia la barra superior
# Filtro One Euro del puntero: min_cutoff bajo = mas suave en reposo; beta alto = menos retraso al moverse
POINTER_MIN_CUTOFF = 1.2
POINTER_BETA = 0.03

FACE_EVERY_N_FRAMES = 2     # el analisis facial se ejecuta 1 de cada N inferencias (ahorra CPU)
ENHANCE_IMAGE = True        # realce ligero de nitidez y contraste del video
HUD_TEXT_REFRESH = 6        # el texto de los paneles se vuelve a renderizar 1 de cada N frames
EMOTION_SMOOTHING = 0.15    # EMA de las puntuaciones de emocion

# Easter eggs: dibujar ciertas figuras lanza una lluvia de emojis
EASTER_EGGS = True
EASTER_EGG_WINDOW = 25.0    # s: solo se tienen en cuenta los trazos recientes
RAIN_DURATION = 3.0         # s que dura la lluvia

# Colores de la interfaz (BGR)
UI_BG = (22, 20, 18)
UI_BORDER = (80, 78, 74)
UI_ACCENT = (0, 200, 255)
UI_TEXT = (240, 240, 240)
UI_MUTED = (160, 160, 160)

EMOTION_COLORS = {
    "Feliz": (60, 220, 120),
    "Sorprendido": (0, 170, 255),
    "Triste": (230, 140, 60),
    "Enojado": (60, 60, 240),
    "Neutral": (170, 170, 170),
}
