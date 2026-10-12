"""Analisis facial: emocion, parpadeos y sonrisas a partir de los blendshapes de MediaPipe."""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from .hands import _read_bytes

EMOTIONS = ["Feliz", "Sorprendido", "Triste", "Enojado", "Neutral"]

# Umbrales de parpadeo (blendshape eyeBlink medio de ambos ojos) con histeresis:
# se considera cerrado al superar BLINK_CLOSE y abierto de nuevo al bajar de BLINK_OPEN.
BLINK_CLOSE = 0.42
BLINK_OPEN = 0.28


class FaceResult:
    def __init__(self):
        self.bbox = None            # (x0, y0, x1, y1)
        self.pts = None             # np.ndarray (478, 2)
        self.blend = {}             # nombre -> valor 0..1
        self.emotion_scores = {}    # emocion -> 0..1 (suavizado)
        self.emotion = "Neutral"
        self.blink = False
        self.mouth_open = 0.0
        self.smile = 0.0


class FaceAnalyzer:
    def __init__(self, face_model, emotion_alpha=0.15):
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_buffer=_read_bytes(face_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

        self.alpha = emotion_alpha
        self._scores = {e: (1.0 if e == "Neutral" else 0.0) for e in EMOTIONS}
        self._eyes_closed = False

    # ------------------------------------------------------------------
    def process(self, rgb, ts_ms, frame_bgr):
        """rgb: frame completo en RGB (numpy). Devuelve FaceResult o None.

        El detector se ejecuta sobre un recorte cuadrado central del frame: asi el
        rostro ocupa mas area relativa y se detecta a mas distancia de la camara.
        """
        ts_ms = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        ox, oy = (w - side) // 2, (h - side) // 2
        crop = np.ascontiguousarray(rgb[oy:oy + side, ox:ox + side])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        res = self._lm.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            return None

        out = FaceResult()
        lms = res.face_landmarks[0]
        out.pts = np.array([[lm.x * side + ox, lm.y * side + oy] for lm in lms], dtype=np.float32)
        x0, y0 = out.pts.min(axis=0)
        x1, y1 = out.pts.max(axis=0)
        out.bbox = (int(x0), int(y0), int(x1), int(y1))

        if res.face_blendshapes:
            out.blend = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        self._update_emotion(out)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _b(blend, *names):
        vals = [blend.get(n, 0.0) for n in names]
        return sum(vals) / max(len(vals), 1)

    def _update_emotion(self, out):
        b = out.blend
        smile = self._b(b, "mouthSmileLeft", "mouthSmileRight")
        cheek = self._b(b, "cheekSquintLeft", "cheekSquintRight")
        frown = self._b(b, "mouthFrownLeft", "mouthFrownRight")
        brow_down = self._b(b, "browDownLeft", "browDownRight")
        brow_inner_up = b.get("browInnerUp", 0.0)
        brow_outer_up = self._b(b, "browOuterUpLeft", "browOuterUpRight")
        jaw = b.get("jawOpen", 0.0)
        eye_wide = self._b(b, "eyeWideLeft", "eyeWideRight")
        sneer = self._b(b, "noseSneerLeft", "noseSneerRight")
        press = self._b(b, "mouthPressLeft", "mouthPressRight")
        lower_down = self._b(b, "mouthLowerDownLeft", "mouthLowerDownRight")
        shrug = b.get("mouthShrugLower", 0.0)

        raw = {
            "Feliz": min(1.0, smile * 1.4 + cheek * 0.3),
            "Sorprendido": min(1.0, jaw * 0.9 + eye_wide * 0.8 + brow_outer_up * 0.5 + brow_inner_up * 0.3),
            "Triste": min(1.0, frown * 1.2 + brow_inner_up * 0.5 + shrug * 0.4 + lower_down * 0.2),
            "Enojado": min(1.0, max(0.0, brow_down - 0.35) * 1.6 + sneer * 0.6 + press * 0.4),
        }
        # Una sonrisa clara anula tristeza/enojo aparentes (ojos entrecerrados al reir)
        if raw["Feliz"] > 0.5:
            raw["Enojado"] *= 0.4
            raw["Triste"] *= 0.4
        raw["Neutral"] = max(0.0, 0.55 - max(raw.values()))

        a = self.alpha
        for e in EMOTIONS:
            self._scores[e] = self._scores[e] * (1 - a) + raw[e] * a
        out.emotion_scores = dict(self._scores)
        out.emotion = max(self._scores, key=self._scores.get)

        out.smile = smile
        out.mouth_open = jaw
        blink_l = b.get("eyeBlinkLeft", 0.0)
        blink_r = b.get("eyeBlinkRight", 0.0)
        blink = (blink_l + blink_r) / 2
        closed = blink > (BLINK_OPEN if self._eyes_closed else BLINK_CLOSE)
        out.blink = closed and not self._eyes_closed   # flanco: solo el frame en que se cierran
        self._eyes_closed = closed

    def close(self):
        self._lm.close()
