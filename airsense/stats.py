"""Estadisticas de sesion: rendimiento, manos, dibujo y emociones."""
import json
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path


class SessionStats:
    def __init__(self):
        self.t_start = time.time()
        self.frames = 0
        self.fps = 0.0
        self.infer_fps = 0.0
        self._t_last_infer = None
        self._t_last = time.perf_counter()
        self._dt = 0.0

        # Manos
        self.frames_with_hand = 0
        self.hand_distance_px = 0.0        # recorrido de la punta del indice
        self.speed_px_s = 0.0
        self.max_speed_px_s = 0.0
        self.gesture_counts = Counter()    # veces que aparece cada gesto (estable)
        self.gesture_time = Counter()      # segundos por gesto
        self.handedness_time = Counter()
        self._last_tip = None
        self._last_tip_t = None
        self._speed_hist = deque(maxlen=8)

        # Dibujo
        self.draw_time = 0.0
        self.actions = Counter()           # limpiar, guardar, deshacer, borrador, color...

        # Rostro
        self.frames_with_face = 0
        self.emotion_time = Counter()
        self.emotion_timeline = deque(maxlen=600)   # (t, emocion) a ~10 muestras/s
        self._last_timeline_t = -1.0
        self.blinks = 0
        self.smiles = 0
        self._smiling = False

    # -- por frame -------------------------------------------------------
    def tick(self):
        now = time.perf_counter()
        self._dt = now - self._t_last
        self._t_last = now
        self.frames += 1
        if self._dt > 0:
            inst = 1.0 / self._dt
            self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1
        return self._dt

    def tick_inference(self):
        now = time.perf_counter()
        if self._t_last_infer is not None:
            dt = now - self._t_last_infer
            if dt > 0:
                inst = 1.0 / dt
                self.infer_fps = inst if self.infer_fps == 0 else self.infer_fps * 0.9 + inst * 0.1
        self._t_last_infer = now

    def update_hands(self, hands, primary_tip, stable_gesture, drawing, new_result=True):
        if hands:
            self.frames_with_hand += 1
            for hnd in hands:
                self.handedness_time[hnd.handedness] += self._dt
        if stable_gesture and stable_gesture != "unknown":
            self.gesture_time[stable_gesture] += self._dt
        if drawing:
            self.draw_time += self._dt

        if not new_result:
            return
        now = time.perf_counter()
        if primary_tip is not None and self._last_tip is not None:
            d = math.hypot(primary_tip[0] - self._last_tip[0], primary_tip[1] - self._last_tip[1])
            dt = max(now - self._last_tip_t, 1e-3)
            if d < 400:  # ignorar saltos (cambio de mano, deteccion perdida)
                self.hand_distance_px += d
                self._speed_hist.append(d / dt)
                self.speed_px_s = sum(self._speed_hist) / len(self._speed_hist)
                self.max_speed_px_s = max(self.max_speed_px_s, self.speed_px_s)
        if primary_tip is not None:
            self._last_tip, self._last_tip_t = primary_tip, now
        else:
            self._last_tip = None
            self._speed_hist.clear()
            self.speed_px_s = 0.0

    def count_gesture(self, gesture):
        self.gesture_counts[gesture] += 1

    def count_action(self, name):
        self.actions[name] += 1

    def update_face(self, face, new_result=True):
        """face se reutiliza en varios frames de video; los eventos (parpadeo, sonrisa)
        solo se cuentan cuando llega un resultado facial nuevo."""
        if face is None:
            return
        self.frames_with_face += 1
        self.emotion_time[face.emotion] += self._dt
        t = self.elapsed
        if t - self._last_timeline_t >= 0.1:
            self.emotion_timeline.append((t, face.emotion))
            self._last_timeline_t = t
        if not new_result:
            return
        if face.blink:
            self.blinks += 1
        smiling = face.smile > 0.55
        if smiling and not self._smiling:
            self.smiles += 1
        self._smiling = smiling

    # -- derivadas -------------------------------------------------------
    @property
    def elapsed(self):
        return time.time() - self.t_start

    @property
    def blinks_per_min(self):
        m = self.elapsed / 60.0
        return self.blinks / m if m > 0.05 else 0.0

    @property
    def dominant_emotion(self):
        if not self.emotion_time:
            return None
        return self.emotion_time.most_common(1)[0][0]

    def emotion_percent(self):
        total = sum(self.emotion_time.values())
        if total <= 0:
            return {}
        return {e: v / total * 100.0 for e, v in self.emotion_time.items()}

    # -- exportar --------------------------------------------------------
    def to_dict(self, canvas=None):
        d = {
            "inicio": datetime.fromtimestamp(self.t_start).isoformat(timespec="seconds"),
            "duracion_s": round(self.elapsed, 1),
            "frames": self.frames,
            "fps_medio": round(self.frames / max(self.elapsed, 1e-6), 1),
            "manos": {
                "frames_con_mano": self.frames_with_hand,
                "recorrido_indice_px": round(self.hand_distance_px),
                "velocidad_max_px_s": round(self.max_speed_px_s),
                "gestos_detectados": dict(self.gesture_counts),
                "segundos_por_gesto": {k: round(v, 1) for k, v in self.gesture_time.items()},
                "segundos_por_mano": {k: round(v, 1) for k, v in self.handedness_time.items()},
            },
            "dibujo": {
                "tiempo_dibujando_s": round(self.draw_time, 1),
                "acciones": dict(self.actions),
            },
            "rostro": {
                "frames_con_rostro": self.frames_with_face,
                "emocion_dominante": self.dominant_emotion,
                "porcentaje_emociones": {k: round(v, 1) for k, v in self.emotion_percent().items()},
                "parpadeos": self.blinks,
                "parpadeos_por_min": round(self.blinks_per_min, 1),
                "sonrisas": self.smiles,
            },
        }
        if canvas is not None:
            d["dibujo"].update({
                "trazos": canvas.n_strokes,
                "puntos": canvas.total_points,
                "cobertura_lienzo_pct": round(canvas.coverage(), 2),
            })
        return d

    def save(self, out_dir, canvas=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(canvas), f, ensure_ascii=False, indent=2)
        return p
