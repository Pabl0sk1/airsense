"""Efectos visuales de los easter eggs: lluvia de emojis (tazas de cafe, corazones...)."""
import math
import os
import random
import sys
import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # noqa: BLE001
    _PIL_OK = False

EMOJI_FONTS = ["seguiemj.ttf", "NotoColorEmoji.ttf", "Apple Color Emoji.ttc"]


def _emoji_font_path():
    dirs = []
    if sys.platform == "win32":
        dirs.append(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"))
    dirs += ["/usr/share/fonts/truetype/noto", "/usr/share/fonts/noto", "/System/Library/Fonts"]
    for d in dirs:
        for n in EMOJI_FONTS:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


def _emoji_sprite(char, size, font_path):
    """Renderiza un emoji a color como BGRA (o None si la fuente no lo soporta)."""
    if not _PIL_OK or font_path is None:
        return None
    try:
        font = ImageFont.truetype(font_path, 109 if "Noto" in font_path else size)
        im = Image.new("RGBA", (int(font.size * 1.4),) * 2, (0, 0, 0, 0))
        ImageDraw.Draw(im).text((font.size * 0.1, font.size * 0.1), char, font=font, embedded_color=True)
        box = im.getbbox()
        if box is None:
            return None
        im = im.crop(box)
        scale = size / max(im.size)
        im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))), Image.LANCZOS)
        a = np.asarray(im)
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)
    except Exception:  # noqa: BLE001
        return None


def _drawn_sprite(kind, size):
    """Sprite dibujado con OpenCV para cuando no hay fuente de emojis."""
    s = size
    img = np.zeros((s, s, 4), np.uint8)
    if kind == "heart":
        t = np.linspace(0, 2 * math.pi, 80)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.stack([s / 2 + x * s / 36, s / 2 + y * s / 36 - s * 0.03], 1).astype(np.int32)
        cv2.fillPoly(img, [pts], (70, 60, 235, 255), cv2.LINE_AA)
    else:
        bw, bh = int(s * 0.55), int(s * 0.5)
        x0, y0 = int(s * 0.12), int(s * 0.35)
        cv2.ellipse(img, (x0 + bw, y0 + bh // 2), (int(s * 0.14), int(s * 0.15)), 0, -90, 90,
                    (245, 245, 245, 255), max(2, s // 14), cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (245, 245, 245, 255), -1)
        cv2.rectangle(img, (x0 + 3, y0 + 3), (x0 + bw - 3, y0 + s // 10), (40, 75, 120, 255), -1)
        for k in range(3):
            sx = x0 + bw // 4 + k * bw // 4
            cv2.line(img, (sx, y0 - 4), (sx + 3, int(s * 0.1)), (210, 210, 210, 200), 2, cv2.LINE_AA)
    return img


# emojis por efecto y forma de respaldo si no hay fuente
RAIN_KINDS = {
    "cup": (["\u2615"], "cup"),
    "heart": (["\u2764\ufe0f", "\U0001F496", "\U0001F495", "\U0001F497", "\U0001F498"], "heart"),
}


class EmojiRain:
    """Lluvia de emojis sobre toda la imagen durante `duration` segundos."""

    SIZES = (44, 60, 80)

    def __init__(self, width, height, duration=3.0, count=46):
        self.w, self.h = width, height
        self.duration = duration
        self.count = count
        self._font = _emoji_font_path()
        self._sprites = {}
        self._parts = []
        self._t0 = None
        self.kind = None

    def sprites(self, kind):
        if kind not in self._sprites:
            chars, fallback = RAIN_KINDS[kind]
            out = []
            for size in self.SIZES:
                for ch in chars:
                    sp = _emoji_sprite(ch, size, self._font)
                    out.append(sp if sp is not None else _drawn_sprite(fallback, size))
            self._sprites[kind] = out
        return self._sprites[kind]

    @property
    def active(self):
        return self._t0 is not None and time.perf_counter() - self._t0 < self.duration

    def start(self, kind):
        sprites = self.sprites(kind)
        self.kind = kind
        self._t0 = time.perf_counter()
        self._parts = []
        for _ in range(self.count):
            sp = random.choice(sprites)
            self._parts.append({
                "sp": sp,
                "x": random.uniform(-20, self.w - 20),
                "y0": random.uniform(-self.h * 1.3, -sp.shape[0]),     # escalonados: caen en oleadas
                "vy": random.uniform(380, 720) * (0.7 + sp.shape[0] / 160),
                "amp": random.uniform(8, 30),
                "freq": random.uniform(1.0, 2.5),
                "ph": random.uniform(0, 2 * math.pi),
            })

    def draw(self, img):
        if not self.active:
            return
        t = time.perf_counter() - self._t0
        fade = min(1.0, (self.duration - t) / 0.6)          # se desvanece en el ultimo 0,6 s
        for p in self._parts:
            y = p["y0"] + p["vy"] * t
            x = p["x"] + p["amp"] * math.sin(p["freq"] * 2 * math.pi * t + p["ph"])
            _blit_bgra(img, p["sp"], int(x), int(y), fade)


def _blit_bgra(img, sp, x, y, opacity=1.0):
    H, W = img.shape[:2]
    h, w = sp.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + s[:, :, :3] * a).astype(np.uint8)
