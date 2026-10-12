"""Dibujo de la interfaz (HUD): texto con fuente real, paneles redondeados, manos, rostro y graficos."""
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np

from . import config as C
from .hands import CONNECTIONS, FINGER_TIPS, THUMB_TIP, WRIST, GESTURE_LABELS
from .face import EMOTIONS

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # noqa: BLE001
    _PIL_OK = False

FONT_CV = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Texto con fuente TrueType (Pillow) y cache de lineas renderizadas
# ---------------------------------------------------------------------------
class TextRenderer:
    """Renderiza lineas de texto con una fuente TTF y las cachea como mascaras alfa.

    La API imita a cv2.putText: `org` es la esquina inferior izquierda (linea base) y
    `scale` se traduce a un tamano en pixeles. Si no hay Pillow o fuente, usa Hershey.
    """

    CANDIDATES_REGULAR = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Roboto-Regular.ttf"]
    CANDIDATES_BOLD = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Roboto-Bold.ttf"]

    def __init__(self, max_cache=1500):
        self._fonts = {}
        self._cache = OrderedDict()
        self._max = max_cache
        self.regular = self._find(self.CANDIDATES_REGULAR)
        self.bold = self._find(self.CANDIDATES_BOLD) or self.regular
        self.ok = _PIL_OK and self.regular is not None

    @staticmethod
    def _find(names):
        dirs = []
        if sys.platform == "win32":
            dirs.append(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))
        dirs += ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/Library/Fonts",
                 "/System/Library/Fonts/Supplemental"]
        for d in dirs:
            for n in names:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    @staticmethod
    def px(scale):
        return max(9, int(round(scale * 30)))

    def _font(self, size, bold):
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            f = ImageFont.truetype(self.bold if bold else self.regular, size)
            self._fonts[key] = f
        return f

    def _mask(self, text, size, bold):
        key = (text, size, bold)
        ent = self._cache.get(key)
        if ent is not None:
            self._cache.move_to_end(key)
            return ent
        font = self._font(size, bold)
        ascent, descent = font.getmetrics()
        w = int(font.getlength(text)) + 3
        h = ascent + descent + 2
        im = Image.new("L", (max(w, 1), max(h, 1)), 0)
        ImageDraw.Draw(im).text((1, 0), text, font=font, fill=255)
        ent = (np.asarray(im), ascent)
        self._cache[key] = ent
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return ent

    def width(self, text, scale=0.5, thick=1):
        if not self.ok:
            return cv2.getTextSize(text, FONT_CV, scale, thick)[0][0]
        return int(self._font(self.px(scale), thick >= 2).getlength(text))

    def draw(self, img, text, org, scale=0.5, color=C.UI_TEXT, thick=1, shadow=False):
        if not text:
            return
        if not self.ok:
            x, y = int(org[0]), int(org[1])
            if shadow:
                cv2.putText(img, text, (x + 1, y + 1), FONT_CV, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), FONT_CV, scale, color, thick, cv2.LINE_AA)
            return
        mask, ascent = self._mask(text, self.px(scale), thick >= 2)
        x0, y0 = int(org[0]), int(org[1]) - ascent
        if shadow:
            self._blend(img, mask, x0 + 1, y0 + 1, (0, 0, 0), 0.8)
        self._blend(img, mask, x0, y0, color, 1.0)

    @staticmethod
    def _blend(img, mask, x0, y0, color, opacity):
        H, W = img.shape[:2]
        h, w = mask.shape
        x1, y1 = min(W, x0 + w), min(H, y0 + h)
        mx0, my0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            return
        m = mask[my0:my0 + (y1 - y0), mx0:mx0 + (x1 - x0)]
        roi = img[y0:y1, x0:x1]
        a = (m.astype(np.float32) * (opacity / 255.0))[..., None]
        col = np.array(color, dtype=np.float32)
        roi[:] = (roi.astype(np.float32) * (1.0 - a) + col * a).astype(np.uint8)


TEXT = TextRenderer()


def put(img, text, org, scale=0.5, color=C.UI_TEXT, thick=1, shadow=None):
    if shadow is None:
        shadow = scale >= 0.55 or thick >= 2
    TEXT.draw(img, text, org, scale, color, thick, shadow)


def text_width(text, scale=0.5, thick=1):
    return TEXT.width(text, scale, thick)


# ---------------------------------------------------------------------------
# Primitivas graficas
# ---------------------------------------------------------------------------
_corner_masks = {}


def _corner_mask(r):
    """Mascara (r x r) con 255 FUERA del cuarto de circulo (esquina superior izquierda)."""
    m = _corner_masks.get(r)
    if m is None:
        yy, xx = np.mgrid[0:r, 0:r]
        outside = ((xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2) > r * r
        m = (outside.astype(np.uint8) * 255)
        _corner_masks[r] = m
    return m


def panel(img, x, y, w, h, alpha=0.7, color=C.UI_BG, border=None, radius=14):
    """Panel semitransparente con esquinas redondeadas (rapido: una pasada + 4 esquinas)."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    rw, rh = x1 - x0, y1 - y0
    r = int(min(radius, rw // 2, rh // 2))
    saved = []
    if r > 1:
        cm = _corner_mask(r)
        for (cx, cy, fx, fy) in [(0, 0, False, False), (rw - r, 0, True, False),
                                 (0, rh - r, False, True), (rw - r, rh - r, True, True)]:
            m = cm
            if fx:
                m = cv2.flip(m, 1)
            if fy:
                m = cv2.flip(m, 0)
            saved.append((cx, cy, roi[cy:cy + r, cx:cx + r].copy(), m))
    cv2.convertScaleAbs(roi, dst=roi, alpha=1.0 - alpha, beta=alpha * (sum(color) / 3.0))
    for cx, cy, orig, m in saved:
        cv2.copyTo(orig, m, roi[cy:cy + r, cx:cx + r])
    if border is not None:
        rounded_rect(img, x0, y0, rw - 1, rh - 1, r, border, 1)


def rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    x, y, w, h = int(x), int(y), int(w), int(h)
    r = int(max(0, min(r, w // 2, h // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        t = thickness
        cv2.line(img, (x + r, y), (x + w - r, y), color, t, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, t, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, t, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color, bg=(58, 58, 64)):
    frac = max(0.0, min(1.0, float(frac)))
    r = h // 2
    rounded_rect(img, x, y, w, h, r, bg)
    fw = int(w * frac)
    if fw >= h:
        rounded_rect(img, x, y, fw, h, r, color)
    elif fw > 0:
        cv2.circle(img, (int(x + r), int(y + r)), r, color, -1, cv2.LINE_AA)


def progress_ring(img, center, radius, frac, color, thick=4):
    frac = max(0.0, min(1.0, float(frac)))
    cv2.circle(img, center, radius, (40, 40, 40), thick + 2, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(img, center, (radius, radius), -90, 0, int(360 * frac), color, thick, cv2.LINE_AA)


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Cache de paneles (el texto se vuelve a renderizar solo cada N frames)
# ---------------------------------------------------------------------------
class _PatchCache:
    def __init__(self):
        self._store = {}

    def get(self, name, w, h, key, render):
        ent = self._store.get(name)
        if ent is None or ent[0] != key or ent[1].shape[0] != h or ent[1].shape[1] != w:
            patch_img = np.zeros((h, w, 3), np.uint8)
            extra = render(patch_img) or {}
            mask = cv2.threshold(cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
            ent = (key, patch_img, mask, extra)
            self._store[name] = ent
        return ent[1], ent[2], ent[3]

    @staticmethod
    def blit(img, x, y, patch_img, mask):
        H, W = img.shape[:2]
        h, w = patch_img.shape[:2]
        x, y = int(x), int(y)
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return
        cv2.copyTo(patch_img, mask, img[y:y + h, x:x + w])


_cache = _PatchCache()


# ---------------------------------------------------------------------------
# Barra de herramientas
# ---------------------------------------------------------------------------
class Toolbar:
    HEIGHT = 60

    def __init__(self, frame_w):
        self.items = []
        x, y, s, gap = 16, 8, 40, 6
        for i, (name, col) in enumerate(C.PALETTE):
            self.items.append({"kind": "color", "rect": (x, y, s, s), "label": name, "value": i})
            x += s + gap
        x += 10
        for key, label, w in [("pen", "Lápiz", 80), ("eraser", "Borrador", 92), ("undo", "Deshacer", 92),
                              ("clear", "Limpiar", 84), ("save", "Guardar", 84),
                              ("minus", "-", 40), ("plus", "+", 40)]:
            self.items.append({"kind": key, "rect": (x, y, w, s), "label": label, "value": None})
            x += w + gap
        self.width = x
        self.zone = (0, 0, frame_w, self.HEIGHT + 16)
        self._cache = None

    def in_zone(self, pt):
        x, y, w, h = self.zone
        return x <= pt[0] < x + w and y <= pt[1] < y + h

    def hit(self, pt):
        for it in self.items:
            x, y, w, h = it["rect"]
            if x <= pt[0] < x + w and y <= pt[1] < y + h:
                return it
        return None

    def draw(self, img, hover=None, progress=0.0, color_idx=0, tool="cursor", brush=8):
        """tool: 'cursor' (no dibuja), 'pen' (lapiz) o 'eraser' (borrador)."""
        panel(img, 6, 2, self.width + 96, self.HEIGHT + 8, alpha=0.6, radius=12)
        key = (id(hover) if hover is not None else 0, int(progress * 25), color_idx, tool, brush)
        pw, ph = self.width + 110, self.HEIGHT + 12
        if self._cache is None or self._cache[0] != key:
            patch_img = np.zeros((ph, pw, 3), np.uint8)
            self._render(patch_img, hover, progress, color_idx, tool, brush)
            mask = cv2.threshold(cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
            self._cache = (key, patch_img, mask)
        _cache.blit(img, 0, 0, self._cache[1], self._cache[2])

    def _render(self, img, hover, progress, color_idx, tool, brush):
        eraser = tool == "eraser"
        for it in self.items:
            x, y, w, h = it["rect"]
            is_hover = hover is it
            if it["kind"] == "color":
                col = C.PALETTE[it["value"]][1]
                rounded_rect(img, x, y, w, h, 8, col)
                if it["value"] == color_idx and not eraser:
                    rounded_rect(img, x - 3, y - 3, w + 6, h + 6, 10, (255, 255, 255), 2)
            else:
                active = it["kind"] == tool
                bg = (78, 78, 86) if not active else C.UI_ACCENT
                rounded_rect(img, x, y, w, h, 8, bg)
                label = it["label"]
                sc = 0.46 if len(label) > 2 else 0.75
                tw = text_width(label, sc, 1)
                fg = (255, 255, 255) if not active else (20, 20, 20)
                put(img, label, (x + (w - tw) // 2, y + h // 2 + (6 if sc < 0.6 else 9)), sc, fg, 1, shadow=False)
            if is_hover:
                rounded_rect(img, x - 2, y - 2, w + 4, h + 4, 9, C.UI_ACCENT, 2)
                if progress > 0:
                    bar(img, x, y + h + 3, w, 4, progress, C.UI_ACCENT)
        # Vista previa del pincel
        px = self.width + 28
        cy = 8 + self.HEIGHT // 2
        col = (200, 200, 200) if eraser else C.PALETTE[color_idx][1]
        cv2.circle(img, (px, cy), max(2, int(brush / 2)), col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, cy), max(2, int(brush / 2)) + 1, (255, 255, 255), 1, cv2.LINE_AA)
        put(img, f"{brush} px", (px + 26, cy + 6), 0.45, C.UI_MUTED, shadow=False)


# ---------------------------------------------------------------------------
# Manos y rostro
# ---------------------------------------------------------------------------
def draw_hand(img, hand, gesture, pointer, hold_progress=0.0, hold_color=C.UI_ACCENT, drawing=False):
    pts = hand.pts.astype(int)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (30, 30, 30), 5, cv2.LINE_AA)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (235, 235, 235), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 6 if i in FINGER_TIPS or i == THUMB_TIP else 4
        up = False
        if i == THUMB_TIP:
            up = hand.fingers[0]
        elif i in FINGER_TIPS:
            up = hand.fingers[1 + FINGER_TIPS.index(i)]
        col = (90, 235, 130) if up else (70, 100, 255)
        cv2.circle(img, tuple(p), r + 1, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), r, col, -1, cv2.LINE_AA)

    if pointer is not None:
        if drawing:
            cv2.circle(img, pointer, 11, (30, 30, 30), 3, cv2.LINE_AA)
            cv2.circle(img, pointer, 11, (255, 255, 255), 2, cv2.LINE_AA)
        elif gesture in ("peace", "point"):
            cv2.circle(img, pointer, 13, (30, 30, 30), 3, cv2.LINE_AA)
            cv2.circle(img, pointer, 13, C.UI_ACCENT, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 3, C.UI_ACCENT, -1, cv2.LINE_AA)

    wx, wy = pts[WRIST]
    label = f"{hand.handedness} · {GESTURE_LABELS.get(gesture, gesture)}"
    tw = text_width(label, 0.48, 1)
    panel(img, wx - tw // 2 - 8, wy + 12, tw + 16, 26, alpha=0.65, radius=8)
    put(img, label, (wx - tw // 2, wy + 30), 0.48, C.UI_TEXT, shadow=False)

    if hold_progress > 0:
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        progress_ring(img, (cx, cy), 30, hold_progress, hold_color, 5)


def draw_face(img, face):
    if face is None or face.bbox is None:
        return
    x0, y0, x1, y1 = face.bbox
    col = C.EMOTION_COLORS.get(face.emotion, C.UI_ACCENT)
    L = max(12, (x1 - x0) // 6)
    for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), col, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), col, 2, cv2.LINE_AA)
    label = face.emotion
    tw = text_width(label, 0.55, 1)
    panel(img, x0, y0 - 30, tw + 16, 26, alpha=0.65, radius=8)
    put(img, label, (x0 + 8, y0 - 11), 0.55, col, shadow=False)


# ---------------------------------------------------------------------------
# Paneles de informacion
# ---------------------------------------------------------------------------
def draw_mode_banner(img, mode_text, color, y=Toolbar.HEIGHT + 24):
    tw = text_width(mode_text, 0.6, 2)
    panel(img, 16, y - 22, tw + 24, 32, alpha=0.65, radius=10)
    put(img, mode_text, (28, y + 2), 0.6, color, 2, shadow=False)


def draw_face_card(img, face, stats, x, y, w=330, h=150):
    panel(img, x, y, w, h, alpha=0.7, border=C.UI_BORDER)
    key = (stats.frames // C.HUD_TEXT_REFRESH, face is not None, None if face is None else face.emotion)
    bars_y = 80

    def render(p):
        put(p, "ROSTRO", (14, 24), 0.48, C.UI_ACCENT, 2, shadow=False)
        if face is None:
            put(p, "Sin rostro detectado", (14, 58), 0.55, C.UI_MUTED, shadow=False)
            return
        col = C.EMOTION_COLORS.get(face.emotion, C.UI_TEXT)
        put(p, face.emotion, (14, 60), 0.95, col, 2, shadow=False)
        for i, e in enumerate(EMOTIONS):
            put(p, e, (14, bars_y + 12 * i + 9), 0.4, C.UI_MUTED, shadow=False)

    patch_img, mask, _ = _cache.get("face_card", w, h, key, render)
    _cache.blit(img, x, y, patch_img, mask)
    if face is not None:
        for i, e in enumerate(EMOTIONS):
            v = face.emotion_scores.get(e, 0.0)
            bar(img, x + 104, y + bars_y + 12 * i, w - 134, 8, v, C.EMOTION_COLORS[e])


def draw_stats_panel(img, stats, canvas, hands, mode_text, brush, color_idx, eraser, x, y, w=330, h=500):
    panel(img, x, y, w, h, alpha=0.7, border=C.UI_BORDER)
    key = stats.frames // C.HUD_TEXT_REFRESH

    def render(p):
        ly = 48
        lh = 19
        pos = {}

        def row(label, value, color=C.UI_TEXT):
            nonlocal ly
            put(p, label, (14, ly), 0.44, C.UI_MUTED, shadow=False)
            put(p, str(value), (162, ly), 0.44, color, shadow=False)
            ly += lh

        def header(t):
            nonlocal ly
            ly += 6
            cv2.line(p, (14, ly - 13), (w - 14, ly - 13), C.UI_BORDER, 1)
            put(p, t, (14, ly), 0.42, C.UI_ACCENT, 2, shadow=False)
            ly += lh

        put(p, "ESTADÍSTICAS", (14, 24), 0.48, C.UI_ACCENT, 2, shadow=False)
        row("FPS vídeo / IA", f"{stats.fps:.0f} / {stats.infer_fps:.0f}")
        row("Sesión", fmt_time(stats.elapsed))
        row("Manos", f"{len(hands)}  " + ", ".join(hd.handedness for hd in hands))
        row("Modo", mode_text.split(":")[0][:22])
        row("Pincel", ("Borrador " if eraser else C.PALETTE[color_idx][0] + " ") + f"{brush} px")
        header("DIBUJO")
        row("Trazos", canvas.n_strokes)
        row("Puntos", canvas.total_points)
        row("Cobertura", f"{canvas.coverage():.2f} %")
        row("Tiempo dibujando", fmt_time(stats.draw_time))
        header("MOVIMIENTO")
        row("Recorrido índice", f"{stats.hand_distance_px / 1000:.2f} kpx")
        row("Velocidad", f"{stats.speed_px_s:.0f} px/s")
        row("Vel. máxima", f"{stats.max_speed_px_s:.0f} px/s")
        pos["bar_y"] = ly - 6
        ly += 10
        header("GESTOS (veces)")
        top = stats.gesture_counts.most_common(3)
        if not top:
            row("-", "aún no hay gestos")
        for g, n in top:
            row(GESTURE_LABELS.get(g, g), n)
        header("ROSTRO")
        row("Parpadeos", f"{stats.blinks}  ({stats.blinks_per_min:.0f}/min)")
        row("Sonrisas", stats.smiles)
        dom = stats.dominant_emotion
        row("Emoción dom.", dom or "-", C.EMOTION_COLORS.get(dom, C.UI_TEXT))
        return pos

    patch_img, mask, pos = _cache.get("stats", w, h, key, render)
    _cache.blit(img, x, y, patch_img, mask)
    bar(img, x + 14, y + pos.get("bar_y", 318), w - 28, 6, stats.speed_px_s / 2500.0, C.UI_ACCENT)


def draw_emotion_timeline(img, stats, x, y, w=330, h=120, window_s=60.0):
    panel(img, x, y, w, h, alpha=0.7, border=C.UI_BORDER)
    gx, gy, gw, gh = x + 14, y + 34, w - 28, 32
    strip = np.full((1, gw, 3), 52, np.uint8)
    now = stats.elapsed
    t0 = now - window_s
    for t, e in stats.emotion_timeline:
        if t < t0:
            continue
        px = min(gw - 1, max(0, int((t - t0) / window_s * gw)))
        strip[0, px] = C.EMOTION_COLORS.get(e, (120, 120, 120))
    H, W = img.shape[:2]
    if gx + gw <= W and gy + gh <= H:
        img[gy:gy + gh, gx:gx + gw] = cv2.resize(strip, (gw, gh), interpolation=cv2.INTER_NEAREST)

    key = stats.frames // C.HUD_TEXT_REFRESH

    def render(p):
        put(p, f"ÁNIMO · últimos {int(window_s)} s", (14, 24), 0.48, C.UI_ACCENT, 2, shadow=False)
        pct = stats.emotion_percent()
        for i, e in enumerate(EMOTIONS):
            v = pct.get(e, 0.0)
            lx = 14 + (i % 3) * 104
            ly = 86 + (i // 3) * 16
            rounded_rect(p, lx, ly - 9, 10, 10, 3, C.EMOTION_COLORS[e])
            put(p, f"{e} {v:.0f}%", (lx + 15, ly), 0.38, C.UI_TEXT, shadow=False)
        put(p, "% de la sesión", (14 + 2 * 104, 102), 0.36, C.UI_MUTED, shadow=False)

    patch_img, mask, _ = _cache.get("timeline", w, h, key, render)
    _cache.blit(img, x, y, patch_img, mask)


_HELP_LINES = [
    "LÁPIZ:  activa el botón Lápiz (arriba) o pulsa 'p' para que el índice dibuje; si no, solo es cursor",
    "GESTOS:  índice = dibujar / cursor  |  V (índice + corazón) = cursor  |  apunta 0,7 s a un botón",
    "pinza (pulgar + índice) = grosor  |  palma 1,2 s = limpiar  |  pulgar arriba 1 s = guardar",
    "rock 0,8 s = borrador  |  llámame 0,8 s = deshacer",
    "TECLAS:  p lápiz   c limpiar   s guardar   u deshacer   e borrador   +/- grosor   f pantalla   h ocultar   a ayuda   q salir",
]


def draw_help(img, x, y):
    w, h = 700, 18 * len(_HELP_LINES) + 12

    def render(p):
        for i, ln in enumerate(_HELP_LINES):
            col = C.UI_ACCENT if i == 0 else (C.UI_TEXT if i < len(_HELP_LINES) - 1 else C.UI_MUTED)
            put(p, ln, (10, 17 + 18 * i), 0.4, col, shadow=False)

    patch_img, mask, _ = _cache.get("help", w, h, 0, render)
    panel(img, x, y, w, h, alpha=0.6, radius=10)
    _cache.blit(img, x, y, patch_img, mask)


class Toast:
    def __init__(self):
        self.msg = None
        self.until = 0.0
        self.color = C.UI_ACCENT

    def show(self, msg, seconds=2.5, color=None):
        self.msg = msg
        self.until = time.time() + seconds
        self.color = color or C.UI_ACCENT

    def draw(self, img):
        if not self.msg or time.time() > self.until:
            return
        h, w = img.shape[:2]
        tw = text_width(self.msg, 0.7, 2)
        x = (w - tw) // 2
        panel(img, x - 20, h // 2 - 28, tw + 40, 48, alpha=0.8, border=self.color, radius=12)
        put(img, self.msg, (x, h // 2 + 8), 0.7, self.color, 2, shadow=False)
