"""Reconocimiento de dibujos con el algoritmo $P (Vatavu, Anthony y Wobbrock, 2012).

$P compara nubes de puntos, asi que no importa el orden ni el numero de trazos con que
se dibuje la figura. Se usa para el easter egg de la taza de cafe: el dibujo se compara con
plantillas de tazas y con plantillas "trampa" (U, rectangulo, circulo...) y solo cuenta como
taza si la plantilla mas parecida es una taza y la distancia es suficientemente baja.
"""
import math

import numpy as np

N_POINTS = 48


# ---------------------------------------------------------------------------
# $P: normalizacion y distancia entre nubes
# ---------------------------------------------------------------------------
def _resample(strokes, n=N_POINTS):
    """Reparte n puntos equidistantes a lo largo de todos los trazos (sin unir trazos)."""
    strokes = [np.asarray(s, np.float64) for s in strokes if len(s) > 0]
    lengths = [np.hypot(*np.diff(s, axis=0).T).sum() if len(s) > 1 else 0.0 for s in strokes]
    total = sum(lengths)
    if total <= 0:
        return np.repeat(strokes[0][:1], n, axis=0)
    step = total / (n - 1)
    out = [strokes[0][0]]
    acc = 0.0
    for s in strokes:
        prev = s[0]
        for p in s[1:]:
            d = math.hypot(p[0] - prev[0], p[1] - prev[1])
            while d > 0 and acc + d >= step:
                q = prev + (step - acc) / d * (p - prev)
                out.append(q)
                d -= step - acc
                prev, acc = q, 0.0
            acc += d
            prev = p
    while len(out) < n:
        out.append(strokes[-1][-1])
    return np.array(out[:n])


def normalize(strokes, n=N_POINTS):
    """Remuestrea, escala de forma uniforme (mantiene la proporcion) y centra en el origen."""
    pts = _resample(strokes, n)
    size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6)
    pts = (pts - pts.min(axis=0)) / size
    return pts - pts.mean(axis=0)


def _cloud_distance(d, start):
    n = d.shape[0]
    used = np.zeros(n, bool)
    total = 0.0
    for k in range(n):
        i = (start + k) % n
        row = np.where(used, np.inf, d[i])
        j = int(np.argmin(row))
        used[j] = True
        total += (1.0 - k / n) * row[j]
    return total


def greedy_match(a, b):
    """Distancia $P entre dos nubes normalizadas del mismo tamano (menor = mas parecidas)."""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    n = len(a)
    step = max(1, int(n ** 0.5))
    best = min(min(_cloud_distance(d, i), _cloud_distance(d.T, i)) for i in range(0, n, step))
    return best / n


# ---------------------------------------------------------------------------
# Plantillas
# ---------------------------------------------------------------------------
def _line(p0, p1, k=12):
    t = np.linspace(0, 1, k)[:, None]
    return (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)


def _arc(cx, cy, rx, ry, a0, a1, k=24):
    a = np.radians(np.linspace(a0, a1, k))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def _cup_body(w, h, style):
    """Cuerpo de la taza (y hacia abajo): U recta, U redondeada, trapecio o rectangulo cerrado."""
    if style == "u":
        return [np.vstack([_line((0, 0), (0, h)), _line((0, h), (w, h)), _line((w, h), (w, 0))])]
    if style == "round":
        r = w * 0.3
        return [np.vstack([_line((0, 0), (0, h - r)), _arc(r, h - r, r, r, 180, 90),
                           _line((r, h), (w - r, h)), _arc(w - r, h - r, r, r, 90, 0),
                           _line((w, h - r), (w, 0))])]
    if style == "trap":
        return [np.vstack([_line((0, 0), (w * 0.15, h)), _line((w * 0.15, h), (w * 0.85, h)),
                           _line((w * 0.85, h), (w, 0))])]
    # rectangulo cerrado
    return [np.vstack([_line((0, 0), (0, h)), _line((0, h), (w, h)), _line((w, h), (w, 0)),
                       _line((w, 0), (0, 0))])]


def _cup_handle(w, h, side, size):
    """Asa en forma de D pegada al lateral (side=1 derecha, -1 izquierda)."""
    x = w if side > 0 else 0
    r = h * size
    cy = h * 0.42
    if side > 0:
        return _arc(x, cy, r, r * 1.1, -90, 90)
    return _arc(x, cy, r, r * 1.1, 270, 90)


def cup_templates():
    out = []
    for style in ("u", "round", "trap", "rect"):
        for aspect in (0.8, 1.1, 1.4):         # ancho / alto del cuerpo
            for side in (1, -1):
                for hs in (0.2, 0.3):
                    h = 1.0
                    w = aspect * h
                    strokes = _cup_body(w, h, style) + [_cup_handle(w, h, side, hs)]
                    if style == "u" and aspect == 1.1:
                        # variante con borde superior en elipse (taza vista desde arriba)
                        strokes.append(_arc(w / 2, 0, w / 2, h * 0.1, 0, 360, 30))
                    out.append(strokes)
                    if style in ("u", "round") and hs == 0.3:
                        # variante con platito debajo
                        out.append(strokes + [_line((-0.25 * w, h * 1.08), (1.25 * w, h * 1.08), 16)])
    return out


def decoy_templates():
    """Figuras parecidas pero que NO son tazas: el dibujo debe parecerse mas a una taza."""
    out = []
    for aspect in (0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.6, 2.0):
        w = aspect
        for style in ("u", "round", "trap", "rect"):
            out.append(_cup_body(w, 1.0, style))
        out.append([_arc(w / 2, 0.5, w / 2, 0.5, 0, 360, 40)])                    # ovalo
        out.append(_cup_body(w, 1.0, "u") + [_line((0, 0), (w, 0))])              # caja con tapa
    out += [
        [_line((0, 0), (1, 0), 30)], [_line((0, 0), (0, 1), 30)], [_line((0, 0), (1, 1), 30)],
        [np.vstack([_line((0, 1), (0.5, 0)), _line((0.5, 0), (1, 1)), _line((1, 1), (0, 1))])],   # triangulo
        [np.array([[i / 8, (i % 2) * 0.6] for i in range(9)], float)],                          # zigzag
        [_arc(0, 0.5, 0.5, 0.5, -90, 90)],                                                      # D sola
        [_line((0, 0), (0, 1)), _arc(0, 0.25, 0.5, 0.25, -90, 90)],                             # P
        [_line((0, 0), (0, 1)), _arc(0, 0.5, 0.6, 0.5, -90, 90)],                               # D
        [np.array([[0.5 + 0.08 * t * math.cos(t), 0.5 + 0.08 * t * math.sin(t)]
                   for t in np.linspace(0, 6 * math.pi, 60)])],                                  # espiral
        [_arc(0.5, 0.5, 0.5, 0.5, 0, 360, 40), _line((0.5, 0), (0.5, 1))],                      # circulo partido
        [_line((0.5, 0), (0.5, 1)), _line((0, 0.5), (1, 0.5))],                                 # cruz
        [np.vstack([_line((0, 0), (0.5, 1)), _line((0.5, 1), (1, 0))])],                       # V
        _cup_body(1.0, 1.0, "rect") + [np.vstack([_line((0, 0), (0.5, -0.5)), _line((0.5, -0.5), (1, 0))])],  # casa
        [np.stack([np.linspace(0, 1, 40), 0.3 * np.sin(np.linspace(0, 4 * math.pi, 40))], 1)],  # ola
    ]
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def _chamfer(a, b):
    """Distancia de Chamfer simetrica (vectorizada): prefiltro barato antes de $P."""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return d.min(axis=1).mean() + d.min(axis=0).mean()


class Recognizer:
    """Clasifica un dibujo por la plantilla mas cercana.

    templates: lista de (etiqueta, trazos); etiqueta None = figura trampa (no dispara nada).
    """

    def __init__(self, templates, threshold, margin, shortlist=12):
        self.templates = [(label, normalize(t)) for label, t in templates]
        self.threshold = threshold
        self.margin = margin
        self.shortlist = shortlist

    def scores(self, strokes):
        """Distancia minima por etiqueta (None = trampas) entre las plantillas preseleccionadas."""
        cloud = normalize(strokes)
        cand = sorted(self.templates, key=lambda lt: _chamfer(cloud, lt[1]))[:self.shortlist]
        best = {}
        for label, tpl in cand:
            d = greedy_match(cloud, tpl)
            if d < best.get(label, math.inf):
                best[label] = d
        return best

    def classify(self, strokes):
        """Etiqueta reconocida o None.

        Ademas de estar por debajo del umbral, la figura debe ser claramente mas parecida a su
        clase que a cualquier trampa (d < margin * d_trampa): asi una U sin asa no es una taza.
        """
        best = self.scores(strokes)
        decoy = best.pop(None, math.inf)
        if not best:
            return None
        label = min(best, key=best.get)
        d = best[label]
        return label if d < self.threshold and d < self.margin * decoy else None


def heart_templates():
    out = []
    t = np.linspace(0, 2 * math.pi, 60)
    for aspect in (0.85, 1.0, 1.2):
        for lobe in (0.0, 0.5):
            x = 16 * np.sin(t) ** 3
            y = -(13 * np.cos(t) - (5 + lobe * 3) * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            pts = np.stack([x * aspect, y], 1) / 32
            out.append([pts])
            # dibujado en dos mitades (izquierda y derecha desde la muesca de arriba)
            out.append([pts[:31], pts[30:]])
    return out


def easter_egg_recognizer(threshold=0.045, margin=0.9):
    tpl = [("cup", t) for t in cup_templates()] + [("heart", t) for t in heart_templates()] +           [(None, t) for t in decoy_templates()]
    return Recognizer(tpl, threshold, margin)


def find_shape(recognizer, strokes, min_size=80, max_group=4):
    """Prueba los ultimos 1..max_group trazos (el mas reciente siempre incluido).

    strokes: lista de arrays (m, 2) en pixeles, del mas antiguo al mas reciente.
    Devuelve (etiqueta, k) con el numero de trazos que forman la figura, o (None, 0).
    """
    for k in range(1, min(max_group, len(strokes)) + 1):
        group = strokes[-k:]
        pts = np.vstack(group)
        if max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) < min_size:
            continue
        label = recognizer.classify(group)
        if label is not None:
            return label, k
    return None, 0
