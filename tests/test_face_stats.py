"""Emociones, parpadeos y estadisticas sin cargar el modelo facial."""
import json

from airsense.canvas import Canvas
from airsense.face import EMOTIONS, FaceAnalyzer, FaceResult
from airsense.stats import SessionStats


def analyzer():
    a = FaceAnalyzer.__new__(FaceAnalyzer)       # sin modelo: solo la logica de blendshapes
    a.alpha = 1.0                                # sin suavizado para que el test sea directo
    a._scores = {e: (1.0 if e == "Neutral" else 0.0) for e in EMOTIONS}
    a._eyes_closed = False
    return a


def face(a, **blend):
    out = FaceResult()
    out.blend = blend
    a._update_emotion(out)
    return out


def test_neutral_by_default():
    assert face(analyzer()).emotion == "Neutral"


def test_smile_is_happy():
    out = face(analyzer(), mouthSmileLeft=0.8, mouthSmileRight=0.8)
    assert out.emotion == "Feliz"
    assert out.smile == 0.8


def test_surprise():
    out = face(analyzer(), jawOpen=0.7, eyeWideLeft=0.6, eyeWideRight=0.6,
               browOuterUpLeft=0.5, browOuterUpRight=0.5)
    assert out.emotion == "Sorprendido"


def test_blink_hysteresis_counts_once():
    a = analyzer()
    seq = [0.1, 0.5, 0.35, 0.5, 0.2, 0.6, 0.1]   # 0.35 no reabre el ojo: dos parpadeos, no tres
    blinks = [face(a, eyeBlinkLeft=v, eyeBlinkRight=v).blink for v in seq]
    assert blinks == [False, True, False, False, False, True, False]


def test_stats_count_blink_only_on_new_face_results():
    st = SessionStats()
    out = FaceResult()
    out.blink = True
    st.tick()
    st.update_face(out, new_result=True)
    for _ in range(3):              # el mismo resultado se reutiliza en varios frames de video
        st.tick()
        st.update_face(out, new_result=False)
    assert st.blinks == 1


def test_stats_smiles_are_edges():
    st = SessionStats()
    out = FaceResult()
    for smile in [0.1, 0.7, 0.8, 0.2, 0.9]:
        out.smile = smile
        st.tick()
        st.update_face(out)
    assert st.smiles == 2


def test_stats_save_json(tmp_path):
    st = SessionStats()
    st.tick()
    st.count_gesture("point")
    st.count_action("save")
    p = st.save(tmp_path, Canvas(32, 32))
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["manos"]["gestos_detectados"] == {"point": 1}
    assert d["dibujo"]["acciones"] == {"save": 1}
    assert d["dibujo"]["trazos"] == 0
