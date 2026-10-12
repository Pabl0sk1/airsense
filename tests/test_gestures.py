"""Clasificacion de gestos con manos sinteticas (sin camara ni modelos)."""
import numpy as np
import pytest

from airsense.hands import GestureStabilizer, HandInfo, OneEuroFilter2D

WRIST = (200, 400)
MCP = {1: (170, 300), 2: (200, 295), 3: (230, 300), 4: (255, 310)}   # indice..menique


def make_hand(thumb, index, middle, ring, pinky, thumb_tip=None):
    """Mano derecha vertical en coordenadas de imagen (y hacia abajo)."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = WRIST
    pts[1], pts[2], pts[3] = (160, 380), (130, 350), (110, 320)
    pts[4] = (90, 290) if thumb else (180, 320)
    for f, up in enumerate([index, middle, ring, pinky], start=1):
        mx, my = MCP[f]
        base = 1 + 4 * f
        pts[base] = (mx, my)
        if up:
            pts[base + 1], pts[base + 2], pts[base + 3] = (mx, my - 40), (mx, my - 70), (mx, my - 100)
        else:
            pts[base + 1], pts[base + 2], pts[base + 3] = (mx, my - 30), (mx, my - 10), (mx, my + 10)
    if thumb_tip is not None:
        pts[4] = thumb_tip
    return HandInfo(pts, "Derecha", 1.0)


@pytest.mark.parametrize("fingers, expected", [
    ((False, True, False, False, False), "point"),
    ((False, True, True, False, False), "peace"),
    ((False, True, True, True, False), "three"),
    ((True, True, True, True, True), "open"),
    ((False, True, False, False, True), "rock"),
    ((True, False, False, False, True), "call"),
    ((True, False, False, False, False), "thumbs_up"),
    ((False, False, False, False, False), "fist"),
])
def test_classify(fingers, expected):
    assert make_hand(*fingers).gesture == expected


def test_finger_states():
    assert make_hand(True, False, True, False, True).fingers == [True, False, True, False, True]


def test_pinch():
    # pulgar tocando la punta del indice con el resto de dedos extendidos
    hand = make_hand(False, True, True, True, True, thumb_tip=(175, 205))
    assert hand.gesture == "pinch"
    assert hand.pinch_dist < 0.35


def test_index_tip_is_int_tuple():
    tip = make_hand(False, True, False, False, False).index_tip
    assert tip == (170, 200) and all(isinstance(v, int) for v in tip)


def test_stabilizer_needs_consecutive_frames():
    st = GestureStabilizer(3)
    assert st.update("point") == "unknown"
    assert st.update("point") == "unknown"
    assert st.update("point") == "point"
    # un gesto suelto no cambia el estado
    assert st.update("open") == "point"
    assert st.update("point") == "point"
    st.reset()
    assert st.current == "unknown"


def test_one_euro_smooths_and_converges():
    f = OneEuroFilter2D(min_cutoff=1.0, beta=0.0)
    assert f.update((100, 100), 0.0) == (100, 100)
    # un salto en reposo se suaviza (no llega entero en un paso)
    x, _ = f.update((200, 100), 1 / 30)
    assert 100 < x < 200
    # con muestras repetidas converge al valor
    t = 1 / 30
    for _ in range(200):
        t += 1 / 30
        p = f.update((200, 100), t)
    assert p == (200, 100)
