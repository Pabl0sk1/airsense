"""Seguimiento de manos y reconocimiento de gestos con MediaPipe Hand Landmarker."""
import math
from collections import deque

import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

# Indices de landmarks (21 puntos por mano)
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

GESTURE_LABELS = {
    "point": "Dibujar",
    "peace": "Cursor",
    "open": "Palma abierta",
    "fist": "Puno",
    "thumbs_up": "Pulgar arriba",
    "rock": "Rock",
    "call": "Llamame",
    "pinch": "Pinza",
    "three": "Tres dedos",
    "unknown": "---",
}


def _read_bytes(path):
    """Carga el modelo en memoria (evita problemas con rutas con acentos en MediaPipe)."""
    with open(path, "rb") as f:
        return f.read()


def _dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


class HandInfo:
    """Resultado de una mano: puntos en pixeles, dedos extendidos y gesto."""

    def __init__(self, pts, handedness, score):
        self.pts = pts                      # np.ndarray (21, 2) en pixeles
        self.handedness = handedness        # "Izquierda" / "Derecha"
        self.score = score
        self.size = max(_dist(pts[WRIST], pts[9]), 1.0)   # muneca -> base del corazon
        self.fingers = self._finger_states()               # [pulgar, indice, corazon, anular, menique]
        self.pinch_dist = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / self.size
        self.gesture = self._classify()

    # -- dedos ---------------------------------------------------------------
    def _finger_states(self):
        w = self.pts[WRIST]
        states = []
        # Pulgar: extendido si la punta esta lejos de la base del menique y del indice
        thumb_ext = (_dist(self.pts[THUMB_TIP], self.pts[17]) > _dist(self.pts[3], self.pts[17]) * 1.05
                     and _dist(self.pts[THUMB_TIP], self.pts[5]) > self.size * 0.55)
        states.append(bool(thumb_ext))
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            states.append(bool(_dist(self.pts[tip], w) > _dist(self.pts[pip], w) * 1.05))
        return states

    # -- gesto ---------------------------------------------------------------
    def _classify(self):
        thumb, index, middle, ring, pinky = self.fingers
        others_up = middle and ring and pinky
        if self.pinch_dist < 0.35 and others_up:
            return "pinch"
        if index and not middle and not ring and not pinky:
            return "point"
        if index and middle and not ring and not pinky:
            return "peace"
        if index and middle and ring and not pinky:
            return "three"
        if index and middle and ring and pinky:
            return "open"
        if index and not middle and not ring and pinky:
            return "rock"
        if thumb and pinky and not index and not middle and not ring:
            return "call"
        if thumb and not index and not middle and not ring and not pinky:
            # pulgar arriba: la punta del pulgar esta claramente por encima de la muneca
            if self.pts[THUMB_TIP][1] < self.pts[WRIST][1] - self.size * 0.4:
                return "thumbs_up"
            return "fist"
        if not index and not middle and not ring and not pinky:
            return "fist"
        return "unknown"

    @property
    def index_tip(self):
        return (int(self.pts[INDEX_TIP][0]), int(self.pts[INDEX_TIP][1]))

    @property
    def gesture_label(self):
        return GESTURE_LABELS.get(self.gesture, self.gesture)

    def bbox(self):
        x0, y0 = self.pts.min(axis=0)
        x1, y1 = self.pts.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


class HandTracker:
    """Envuelve el Hand Landmarker de MediaPipe en modo VIDEO."""

    def __init__(self, model_path, num_hands=2, mirrored=True):
        # Con la imagen en espejo (webcam volteada) la etiqueta de MediaPipe sale invertida
        self.mirrored = mirrored
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_buffer=_read_bytes(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, mp_image, ts_ms, w, h):
        ts_ms = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts_ms
        res = self._lm.detect_for_video(mp_image, ts_ms)
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            cat = hd[0]
            is_right = cat.category_name == "Right"
            if self.mirrored:
                is_right = not is_right
            label = "Derecha" if is_right else "Izquierda"
            hands.append(HandInfo(pts, label, float(cat.score)))
        # Orden estable: por posicion horizontal (izquierda -> derecha en pantalla)
        hands.sort(key=lambda hnd: hnd.pts[WRIST][0])
        return hands

    def close(self):
        self._lm.close()


class GestureStabilizer:
    """Acepta un gesto solo cuando se repite N frames seguidos (anti-parpadeo)."""

    def __init__(self, frames=3):
        self.frames = frames
        self._hist = deque(maxlen=frames)
        self.current = "unknown"

    def update(self, gesture):
        self._hist.append(gesture)
        if len(self._hist) == self.frames and len(set(self._hist)) == 1:
            self.current = gesture
        return self.current

    def reset(self):
        self._hist.clear()
        self.current = "unknown"


class OneEuroFilter2D:
    """Filtro One Euro (Casiez 2012): suave en reposo, poco retraso en movimientos rapidos."""

    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = (0.0, 0.0)
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, p, t):
        x = (float(p[0]), float(p[1]))
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t
            return (int(round(x[0])), int(round(x[1])))
        dt = max(1e-3, t - self._t)
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        dx = tuple((x[i] - self._x[i]) / dt for i in range(2))
        self._dx = tuple(a_d * dx[i] + (1 - a_d) * self._dx[i] for i in range(2))
        speed = math.hypot(*self._dx)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = tuple(a * x[i] + (1 - a) * self._x[i] for i in range(2))
        return (int(round(self._x[0])), int(round(self._x[1])))

