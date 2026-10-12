"""Recorte de la cola de los trazos, reconocimiento de figuras y lluvia de emojis."""
import math
import time

import numpy as np
import pytest

from airsense import shapes as S
from airsense.canvas import Canvas
from airsense.effects import EmojiRain


# ---------------------------------------------------------------------------
# Recorte de la cola del trazo
# ---------------------------------------------------------------------------
def test_end_with_cutoff_removes_tail_from_canvas():
    c = Canvas(300, 100)
    c.begin(0, (0, 0, 255), 6)
    for k, x in enumerate(range(10, 290, 20)):
        c.add_point(0, (x, 50), t=float(k))
    assert c.mask[50, 250] == 255                 # la cola se ve mientras se dibuja
    s = c.end(0, cutoff=6.0)                      # se queda con los puntos de t <= 6 (x <= 130)
    assert s is not None and len(s.points) == 7
    assert c.total_points == 7
    assert c.mask[50, 100] == 255
    assert c.mask[50, 200] == 0 and c.mask[50, 270] == 0


def test_cutoff_keeps_previous_strokes():
    c = Canvas(300, 100)
    c.begin(0, (255, 255, 255), 6)
    c.add_point(0, (10, 20), t=0.0)
    c.add_point(0, (290, 20), t=1.0)
    c.end(0)
    c.begin(0, (0, 0, 255), 6)
    c.add_point(0, (10, 80), t=2.0)
    c.add_point(0, (290, 80), t=3.0)
    c.end(0, cutoff=1.0)                          # todo el segundo trazo era cola
    assert c.n_strokes == 1
    assert c.mask[20, 150] == 255 and c.mask[80, 150] == 0


def test_cutoff_with_two_active_hands_rebuilds():
    c = Canvas(300, 100)
    c.begin(0, (0, 0, 255), 6)
    c.begin(1, (0, 255, 0), 6)
    for k, x in enumerate(range(10, 290, 40)):
        c.add_point(0, (x, 20), t=float(k))
        c.add_point(1, (x, 80), t=float(k))
    c.end(0, cutoff=2.0)
    assert c.mask[20, 200] == 0
    assert c.is_drawing(1) and c.mask[80, 200] == 255


def test_tail_run_start_finds_final_move_up():
    c = Canvas(400, 400)
    c.begin(0, (255, 255, 255), 4)
    # dibujo horizontal y luego sube recto hacia la barra superior
    pts = [(x, 300) for x in range(50, 250, 25)] + [(250, y) for y in range(280, 60, -40)]
    t0 = 100.0
    for k, p in enumerate(pts):
        c.add_point(0, p, t=t0 + k * 0.05)
    t_run = c.tail_run_start(0, max_s=0.8)
    s = c.active[0]
    kept = [p for p, t in zip(s.points, s.times) if t <= t_run]
    assert kept[-1][1] == 300                     # la subida se descarta, el dibujo se queda


# ---------------------------------------------------------------------------
# Reconocimiento de figuras
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rec():
    return S.easter_egg_recognizer()


def _px(strokes, size=250, ox=300, oy=200):
    return [np.asarray(s) * size + [ox, oy] for s in strokes]


def _jitter(strokes, seed=0, amp=3.0):
    rng = np.random.default_rng(seed)
    return [s + rng.normal(0, amp, s.shape) for s in strokes]


def test_recognizes_cup_in_several_strokes(rec):
    cup = S._cup_body(1.0, 1.0, "u") + [S._arc(1.0, 0.45, 0.25, 0.3, -90, 90)]
    label, k = S.find_shape(rec, _jitter(_px(cup)))
    assert label == "cup" and k == 2


def test_recognizes_heart(rec):
    t = np.linspace(0, 2 * math.pi, 50)
    heart = np.stack([16 * np.sin(t) ** 3,
                      -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))], 1) / 32
    label, _ = S.find_shape(rec, _jitter(_px([heart])))
    assert label == "heart"


@pytest.mark.parametrize("strokes", [
    S._cup_body(1.0, 1.0, "u"),                                  # taza sin asa
    S._cup_body(1.0, 1.0, "rect"),                               # rectangulo
    [S._arc(0.5, 0.5, 0.5, 0.5, 0, 360, 40)],                    # circulo
    [S._line((0, 0), (1, 0), 30)],                               # linea
])
def test_does_not_trigger_on_other_shapes(rec, strokes):
    assert S.find_shape(rec, _jitter(_px(strokes)))[0] is None


def test_tiny_drawings_are_ignored(rec):
    cup = S._cup_body(1.0, 1.0, "u") + [S._arc(1.0, 0.45, 0.25, 0.3, -90, 90)]
    assert S.find_shape(rec, _px(cup, size=30)) == (None, 0)


def test_cup_with_saucer_line_still_counts(rec):
    cup = _px(S._cup_body(1.0, 1.0, "u") + [S._arc(1.0, 0.45, 0.25, 0.3, -90, 90)])
    saucer = _px([S._line((-0.2, 1.08), (1.2, 1.08), 20)])
    assert S.find_shape(rec, _jitter(cup + saucer))[0] == "cup"


# ---------------------------------------------------------------------------
# Lluvia
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["cup", "heart"])
def test_rain_draws_and_ends(kind):
    rain = EmojiRain(320, 240, duration=0.2, count=10)
    frame = np.zeros((240, 320, 3), np.uint8)
    rain.start(kind)
    assert rain.active
    rain._t0 -= 0.1                               # avanzar el tiempo para que esten en pantalla
    for p in rain._parts:
        p["y0"] = 50
    rain.draw(frame)
    assert frame.any()
    time.sleep(0.25)
    assert not rain.active
