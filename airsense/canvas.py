"""Lienzo de dibujo con trazos suavizados (Catmull-Rom), borrador, deshacer y guardado."""
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class Stroke:
    __slots__ = ("color", "thickness", "points", "eraser", "drawn")

    def __init__(self, color, thickness, eraser=False):
        self.color = color
        self.thickness = int(thickness)
        self.eraser = eraser
        self.points = []
        self.drawn = 0          # numero de segmentos ya rasterizados


class Canvas:
    MIN_STEP = 2.0              # px: puntos mas cercanos al anterior se ignoran (temblor)

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.strokes = []
        self.active = {}          # id de mano -> Stroke en curso
        self.total_points = 0

    # -- trazos --------------------------------------------------------------
    def begin(self, hand_id, color, thickness, eraser=False):
        s = Stroke(color, thickness, eraser)
        self.active[hand_id] = s
        self.strokes.append(s)
        return s

    def add_point(self, hand_id, pt):
        s = self.active.get(hand_id)
        if s is None:
            return
        pt = (int(pt[0]), int(pt[1]))
        if s.points and math.hypot(pt[0] - s.points[-1][0], pt[1] - s.points[-1][1]) < self.MIN_STEP:
            return
        s.points.append(pt)
        self.total_points += 1
        self._advance(s, final=False)

    def end(self, hand_id):
        s = self.active.pop(hand_id, None)
        if s is None:
            return
        if len(s.points) == 0:
            if s in self.strokes:
                self.strokes.remove(s)
            return
        self._advance(s, final=True)

    def is_drawing(self, hand_id):
        return hand_id in self.active

    # -- rasterizado con curvas ---------------------------------------------
    def _advance(self, s, final):
        """Dibuja los segmentos pendientes del trazo.

        El segmento i (points[i] -> points[i+1]) se traza como curva Catmull-Rom usando
        points[i-1] y points[i+2] como control. Mientras se dibuja, el ultimo segmento se
        pospone hasta conocer el punto siguiente (asi la curva no cambia a posteriori);
        al terminar (final=True) se traza todo lo que quede.
        """
        p = s.points
        n = len(p)
        if n == 1:
            if s.drawn == 0 and final:
                self._draw_poly(s, [p[0], p[0]])
                s.drawn = 1
            return
        last_ready = n - 1 if final else n - 2     # ultimo indice de segmento trazable
        i = s.drawn
        while i < last_ready:
            p0 = p[i - 1] if i - 1 >= 0 else p[i]
            p1, p2 = p[i], p[i + 1]
            p3 = p[i + 2] if i + 2 < n else p[i + 1]
            self._draw_poly(s, self._catmull_rom(p0, p1, p2, p3))
            i += 1
        s.drawn = i

    @staticmethod
    def _catmull_rom(p0, p1, p2, p3):
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(max(2, min(24, d / 3.0)))
        pts = []
        for k in range(steps + 1):
            t = k / steps
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            pts.append((int(round(x)), int(round(y))))
        return pts

    def _draw_poly(self, s, pts):
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if s.eraser:
            t = s.thickness * 2
            cv2.polylines(self.layer, [arr], False, (0, 0, 0), t, cv2.LINE_AA)
            cv2.polylines(self.mask, [arr], False, 0, t, cv2.LINE_AA)
        else:
            cv2.polylines(self.layer, [arr], False, s.color, s.thickness, cv2.LINE_AA)
            cv2.polylines(self.mask, [arr], False, 255, s.thickness, cv2.LINE_AA)

    # -- edicion -------------------------------------------------------------
    def undo(self):
        """Quita el ultimo trazo terminado (no uno que se este dibujando)."""
        active = set(id(s) for s in self.active.values())
        for i in range(len(self.strokes) - 1, -1, -1):
            if id(self.strokes[i]) not in active:
                del self.strokes[i]
                self._rebuild()
                return True
        return False

    def clear(self):
        self.strokes = list(self.active.values())
        for s in self.strokes:
            s.points = []
            s.drawn = 0
        self._rebuild()

    def _rebuild(self):
        self.layer[:] = 0
        self.mask[:] = 0
        active = set(id(s) for s in self.active.values())
        for s in self.strokes:
            s.drawn = 0
            if s.points:
                self._advance(s, final=id(s) not in active)

    # -- salida --------------------------------------------------------------
    def render(self, frame):
        """Superpone el lienzo sobre el frame (in place)."""
        cv2.copyTo(self.layer, self.mask, frame)
        return frame

    def coverage(self):
        return float(np.count_nonzero(self.mask)) / float(self.mask.size) * 100.0

    def save(self, out_dir, with_frame=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = []
        p1 = out_dir / f"dibujo_{stamp}.png"
        cv2.imwrite(str(p1), self.layer)
        paths.append(p1)
        rgba = cv2.cvtColor(self.layer, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = self.mask
        p2 = out_dir / f"dibujo_{stamp}_alpha.png"
        cv2.imwrite(str(p2), rgba)
        paths.append(p2)
        if with_frame is not None:
            p3 = out_dir / f"captura_{stamp}.png"
            cv2.imwrite(str(p3), with_frame)
            paths.append(p3)
        return paths

    @property
    def n_strokes(self):
        return sum(1 for s in self.strokes if s.points)
