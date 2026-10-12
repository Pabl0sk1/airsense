"""AirSense - dibuja en el aire con la mano, controla con gestos y analiza tu rostro.

Uso:
    python app.py                 # abre la camara automaticamente
    python app.py --camera 1      # elige otra camara
    python app.py --source v.mp4  # usa un video en lugar de la camara
    python app.py --no-face       # desactiva el analisis facial
    python app.py --fullscreen    # arranca en pantalla completa (tecla 'f' alterna)

El indice solo dibuja cuando el Lapiz esta activo (boton "Lapiz" de la barra superior o
tecla 'p'); si no, la punta del indice actua como cursor para pulsar los botones.

El video se muestra a la velocidad de la camara (30 fps); la inferencia (manos, rostro)
corre en un hilo aparte y se aplica el ultimo resultado disponible.
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault("GLOG_minloglevel", "2")   # silencia avisos internos de MediaPipe
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from airsense import __version__  # noqa: E402
from airsense import config as C  # noqa: E402
from airsense import ui  # noqa: E402
from airsense.camera import open_source  # noqa: E402
from airsense.canvas import Canvas  # noqa: E402
from airsense.face import FaceAnalyzer  # noqa: E402
from airsense.hands import GestureStabilizer, HandTracker, OneEuroFilter2D  # noqa: E402
from airsense.stats import SessionStats  # noqa: E402

WINDOW = "AirSense"

HOLD_ACTIONS = {
    "open": ("clear", C.HOLD_CLEAR, (60, 60, 240)),
    "thumbs_up": ("save", C.HOLD_SAVE, (60, 220, 120)),
    "call": ("undo", C.HOLD_UNDO, (0, 170, 255)),
    "rock": ("eraser", C.HOLD_ERASER, (230, 60, 230)),
}

# Gestos que cortan un trazo inmediatamente (los demas necesitan DRAW_STOP_FRAMES seguidos)
DRAW_STOP_NOW = {"peace", "open", "fist", "pinch"}

MODE_TEXT = {
    "point": ("DIBUJANDO", (80, 230, 120)),
    "point_cursor": ("CURSOR: activa el Lápiz para dibujar", C.UI_ACCENT),
    "peace": ("CURSOR / MENÚ", C.UI_ACCENT),
    "pinch": ("AJUSTANDO GROSOR", (230, 60, 230)),
    "open": ("PALMA: mantener para limpiar", (60, 60, 240)),
    "thumbs_up": ("PULGAR: mantener para guardar", (60, 220, 120)),
    "call": ("LLÁMAME: mantener para deshacer", (0, 170, 255)),
    "rock": ("ROCK: mantener para borrador", (230, 60, 230)),
    "fist": ("PAUSA (puño)", C.UI_MUTED),
    "three": ("TRES DEDOS", C.UI_MUTED),
    "unknown": ("SIN GESTO", C.UI_MUTED),
}


def check_models(need_face=True):
    missing = [p for p in [C.HAND_MODEL] if not p.exists()]
    if need_face:
        missing += [p for p in [C.FACE_MODEL] if not p.exists()]
    if missing:
        print("Faltan modelos:")
        for p in missing:
            print("  -", p)
        print("Ejecuta:  python download_models.py")
        sys.exit(1)


class InferenceResult:
    __slots__ = ("hands", "face", "ran_face", "ts_ms", "t_done")

    def __init__(self, hands, face, ran_face, ts_ms):
        self.hands = hands
        self.face = face
        self.ran_face = ran_face
        self.ts_ms = ts_ms
        self.t_done = time.perf_counter()


class InferenceWorker(threading.Thread):
    """Ejecuta manos y rostro en segundo plano sobre el frame mas reciente enviado."""

    def __init__(self, tracker, face_analyzer, face_every):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.face = face_analyzer
        self.face_every = max(1, int(face_every))
        self._cv = threading.Condition()
        self._pending = None
        self._result = None
        self._stop = False
        self.busy = False
        self._count = 0
        self.error = None

    def submit(self, frame_bgr, ts_ms):
        with self._cv:
            self._pending = (frame_bgr, ts_ms)
            self._cv.notify()

    def poll(self):
        with self._cv:
            r, self._result = self._result, None
        return r

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, ts_ms = self._pending
                self._pending = None
                self.busy = True
            try:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hands = self.tracker.process(mp_img, ts_ms, w, h)
                face, ran_face = None, False
                if self.face is not None and self._count % self.face_every == 0:
                    face = self.face.process(rgb, ts_ms, frame)
                    ran_face = True
                self._count += 1
                with self._cv:
                    self._result = InferenceResult(hands, face, ran_face, ts_ms)
            except Exception as exc:  # noqa: BLE001
                self.error = exc
                with self._cv:
                    self._result = InferenceResult([], None, False, ts_ms)
            finally:
                self.busy = False


class HandState:
    """Estado por mano (indice 0/1): estabilizador de gesto, filtro del puntero, trazo."""

    def __init__(self):
        self.stab = GestureStabilizer(C.GESTURE_STABLE_FRAMES)
        self.filt = OneEuroFilter2D(C.POINTER_MIN_CUTOFF, C.POINTER_BETA)
        self.gesture = "unknown"
        self.last_gesture = "unknown"
        self.pointer = None
        self.non_point_frames = 0

    def reset(self):
        self.stab.reset()
        self.filt.reset()
        self.gesture = "unknown"
        self.last_gesture = "unknown"
        self.pointer = None
        self.non_point_frames = 0


class App:
    def __init__(self, args):
        self.args = args
        src = args.source if args.source else (str(args.camera) if args.camera is not None else None)
        self.cap = open_source(src, C.FRAME_W, C.FRAME_H)
        self.tracker = HandTracker(C.HAND_MODEL, num_hands=args.hands, mirrored=not args.no_mirror)
        self.face = None
        if not args.no_face:
            self.face = FaceAnalyzer(C.FACE_MODEL, emotion_alpha=C.EMOTION_SMOOTHING)
        self.worker = InferenceWorker(self.tracker, self.face, C.FACE_EVERY_N_FRAMES)
        self.canvas = Canvas(C.FRAME_W, C.FRAME_H)
        self.stats = SessionStats()
        self.toolbar = ui.Toolbar(C.FRAME_W)
        self.toast = ui.Toast()

        self.color_idx = 1
        self.brush = C.BRUSH_DEFAULT
        # Herramienta activa: "cursor" (el indice no dibuja), "pen" (lapiz) o "eraser" (borrador).
        # Arranca en cursor para que un dedo suelto no pinte por accidente.
        self.tool = "cursor"
        self.show_hud = True
        self.show_help = True
        self.mirror = not args.no_mirror
        self.fullscreen = False
        self.enhance = C.ENHANCE_IMAGE and not args.no_enhance

        self.hand_states = [HandState(), HandState()]
        self.hands = []
        self.hold_gesture = None
        self.hold_start = 0.0
        self.hold_fired = False
        self.hover_item = None
        self.hover_start = 0.0
        self.hover_fired = False

        self.t0 = time.perf_counter()
        self.prof = {}
        self.last_face = None
        self.face_missing = 0
        self.last_frame = None

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------
    @property
    def eraser(self):
        return self.tool == "eraser"

    @property
    def can_draw(self):
        return self.tool != "cursor"

    def set_tool(self, tool):
        if tool == self.tool:
            return
        self.tool = tool
        for i in range(2):
            self.canvas.end(i)
        if tool == "pen":
            self.toast.show("Lápiz activado: el índice dibuja", color=(80, 230, 120))
        elif tool == "eraser":
            self.toast.show("Borrador activado", color=(230, 60, 230))
        else:
            self.toast.show("Lápiz desactivado: el índice es cursor", color=C.UI_ACCENT)

    def do_action(self, name, frame_clean=None):
        if name == "clear":
            self.canvas.clear()
            self.toast.show("Lienzo limpiado", color=(60, 60, 240))
        elif name == "save":
            paths = self.canvas.save(C.OUTPUT_DIR, with_frame=frame_clean)
            self.toast.show(f"Guardado: {paths[-1].name}", color=(60, 220, 120))
            print("[guardado]", ", ".join(str(p) for p in paths), flush=True)
        elif name == "undo":
            ok = self.canvas.undo()
            self.toast.show("Trazo deshecho" if ok else "Nada que deshacer", color=(0, 170, 255))
        elif name == "pen":
            self.set_tool("cursor" if self.tool == "pen" else "pen")
        elif name == "eraser":
            self.set_tool("pen" if self.tool == "eraser" else "eraser")
        elif name == "plus":
            self.brush = min(C.BRUSH_MAX, self.brush + 2)
        elif name == "minus":
            self.brush = max(C.BRUSH_MIN, self.brush - 2)
        elif name.startswith("color:"):
            self.color_idx = int(name.split(":")[1])
            if self.tool == "eraser":
                self.tool = "pen"
            self.toast.show(f"Color: {C.PALETTE[self.color_idx][0]}", color=C.PALETTE[self.color_idx][1])
        self.stats.count_action(name.split(":")[0])

    def handle_key(self, key, frame_clean):
        if key in (27, ord("q")):
            return False
        if key == ord("c"):
            self.do_action("clear")
        elif key == ord("s"):
            self.do_action("save", frame_clean)
        elif key == ord("u"):
            self.do_action("undo")
        elif key == ord("p"):
            self.do_action("pen")
        elif key == ord("e"):
            self.do_action("eraser")
        elif key in (ord("+"), ord("=")):
            self.do_action("plus")
        elif key in (ord("-"), ord("_")):
            self.do_action("minus")
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key in (ord("?"), ord("a")):
            self.show_help = not self.show_help
        elif key == ord("m"):
            self.mirror = not self.mirror
            self.tracker.mirrored = self.mirror
        elif key == ord("n"):
            self.enhance = not self.enhance
        elif key == ord("f") and not self.args.headless:
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif ord("1") <= key <= ord("8"):
            self.do_action(f"color:{key - ord('1')}")
        return True

    # ------------------------------------------------------------------
    # Logica de manos (gestos, trazos, acciones)
    # ------------------------------------------------------------------
    def update_hand_states(self, hands, t_now):
        """Se llama solo cuando llega un resultado nuevo de inferencia."""
        for i, hand in enumerate(hands[:2]):
            st = self.hand_states[i]
            raw = hand.gesture
            g = st.stab.update(raw)
            st.pointer = st.filt.update(hand.index_tip, t_now)
            # Histeresis del trazo: mientras se dibuja, gestos dudosos no cortan enseguida
            if self.canvas.is_drawing(i):
                if raw == "point" or g == "point":
                    st.non_point_frames = 0
                    g = "point"
                elif g in DRAW_STOP_NOW:
                    st.non_point_frames = C.DRAW_STOP_FRAMES
                else:
                    st.non_point_frames += 1
                    if st.non_point_frames < C.DRAW_STOP_FRAMES:
                        g = "point"
            else:
                st.non_point_frames = 0
            st.gesture = g
            if g != st.last_gesture:
                if g != "unknown":
                    self.stats.count_gesture(g)
                st.last_gesture = g
        for i in range(len(hands), 2):
            self.canvas.end(i)
            self.hand_states[i].reset()

    def process_hands(self, hands, frame_clean, new_result):
        now = time.perf_counter()
        if new_result:
            self.update_hand_states(hands, now)

        drawing_any = False
        primary_tip = None
        hover_candidate = None
        pinch_preview = None

        for i, hand in enumerate(hands[:2]):
            st = self.hand_states[i]
            g, tip = st.gesture, st.pointer
            if tip is None:
                continue
            if primary_tip is None:
                primary_tip = tip
            in_toolbar = self.toolbar.in_zone(tip)
            if g == "point" and not in_toolbar and self.can_draw:
                if not self.canvas.is_drawing(i):
                    self.canvas.begin(i, C.PALETTE[self.color_idx][1], self.brush, self.eraser)
                if new_result:
                    self.canvas.add_point(i, tip)
                drawing_any = True
            else:
                self.canvas.end(i)
                if g in ("point", "peace") and in_toolbar:
                    hover_candidate = self.toolbar.hit(tip)
                if g == "pinch":
                    f = (hand.pinch_dist - 0.05) / 0.30
                    f = max(0.0, min(1.0, f))
                    self.brush = int(round(C.BRUSH_MIN + f * (C.BRUSH_MAX - C.BRUSH_MIN)))
                    mid = ((hand.pts[4][0] + hand.pts[8][0]) / 2, (hand.pts[4][1] + hand.pts[8][1]) / 2)
                    pinch_preview = (int(mid[0]), int(mid[1]))

        gestures = [self.hand_states[i].gesture if i < len(hands) else "unknown" for i in range(2)]

        # Acciones por gesto mantenido (cualquier mano)
        hold_progress = [0.0, 0.0]
        hold_color = C.UI_ACCENT
        held = None
        for i, g in enumerate(gestures[:len(hands)]):
            if g in HOLD_ACTIONS:
                held = (i, g)
                break
        if held is None:
            self.hold_gesture, self.hold_fired = None, False
        else:
            i, g = held
            if g != self.hold_gesture:
                self.hold_gesture, self.hold_start, self.hold_fired = g, now, False
            action, duration, hold_color = HOLD_ACTIONS[g]
            frac = (now - self.hold_start) / duration
            hold_progress[i] = min(1.0, frac)
            if frac >= 1.0 and not self.hold_fired:
                self.do_action(action, frame_clean)
                self.hold_fired = True

        # Barra de herramientas por permanencia del puntero
        hover_progress = 0.0
        if hover_candidate is None:
            self.hover_item, self.hover_fired = None, False
        else:
            if hover_candidate is not self.hover_item:
                self.hover_item, self.hover_start, self.hover_fired = hover_candidate, now, False
            frac = (now - self.hover_start) / C.DWELL_BUTTON
            hover_progress = min(1.0, frac)
            if frac >= 1.0 and not self.hover_fired:
                it = self.hover_item
                name = f"color:{it['value']}" if it["kind"] == "color" else it["kind"]
                self.do_action(name, frame_clean)
                if it["kind"] in ("plus", "minus"):
                    self.hover_start = now
                else:
                    self.hover_fired = True

        mode_key = "nohand" if not hands else "unknown"
        for g in gestures[:len(hands)]:
            if g != "unknown":
                mode_key = g
                break
        if mode_key == "point" and not self.can_draw:
            mode_key = "point_cursor"

        return {
            "pointers": [self.hand_states[0].pointer, self.hand_states[1].pointer],
            "gestures": gestures, "drawing": drawing_any, "primary_tip": primary_tip,
            "hold_progress": hold_progress, "hold_color": hold_color,
            "hover_progress": hover_progress, "pinch_preview": pinch_preview, "mode_key": mode_key,
        }

    # ------------------------------------------------------------------
    # Imagen
    # ------------------------------------------------------------------
    @staticmethod
    def fit_frame(frame):
        """Escala manteniendo la proporcion y rellena con bandas negras hasta FRAME_W x FRAME_H."""
        h, w = frame.shape[:2]
        s = min(C.FRAME_W / w, C.FRAME_H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if nw == C.FRAME_W and nh == C.FRAME_H:
            return resized
        out = np.zeros((C.FRAME_H, C.FRAME_W, 3), dtype=np.uint8)
        ox, oy = (C.FRAME_W - nw) // 2, (C.FRAME_H - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = resized
        return out

    @staticmethod
    def enhance_frame(frame):
        """Realce ligero: mascara de enfoque + un poco de contraste."""
        blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
        cv2.addWeighted(frame, 1.45, blur, -0.45, 0, dst=frame)
        cv2.convertScaleAbs(frame, dst=frame, alpha=1.06, beta=-6)
        return frame

    def _prof(self, name, t_prev):
        now = time.perf_counter()
        self.prof[name] = self.prof.get(name, 0.0) + (now - t_prev)
        return now

    # ------------------------------------------------------------------
    def run(self):
        if not self.args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, C.FRAME_W, C.FRAME_H)
            if self.args.fullscreen:
                self.fullscreen = True
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.worker.start()
        print("[airsense] listo. Pulsa 'q' o ESC para salir.", flush=True)
        is_file = bool(self.args.source)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    if is_file:
                        print("[video] fin del video")
                        break
                    print("[camara] frame perdido, reintentando...")
                    time.sleep(0.05)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                if frame.shape[1] != C.FRAME_W or frame.shape[0] != C.FRAME_H:
                    frame = self.fit_frame(frame)

                self.stats.tick()
                pt = time.perf_counter()
                ts_ms = int((pt - self.t0) * 1000)

                # Inferencia en segundo plano: se envia el frame si el hilo esta libre.
                # Con un archivo de video se procesa cada frame (espera al hilo).
                if is_file:
                    self.worker.submit(frame.copy(), ts_ms)
                    res = None
                    while res is None:
                        res = self.worker.poll()
                        if res is None:
                            time.sleep(0.001)
                else:
                    if not self.worker.busy:
                        self.worker.submit(frame.copy(), ts_ms)
                    res = self.worker.poll()
                pt = self._prof("inferencia(espera)", pt)

                new_result = res is not None
                new_face = False
                if new_result:
                    self.stats.tick_inference()
                    self.hands = res.hands
                    if res.ran_face:
                        if res.face is not None:
                            self.last_face, self.face_missing = res.face, 0
                            new_face = True
                        else:
                            self.face_missing += 1
                if self.worker.error is not None:
                    raise self.worker.error
                face = self.last_face if (self.face is not None and self.face_missing < 4) else None

                if self.enhance:
                    self.enhance_frame(frame)
                self.canvas.render(frame)
                frame_clean = frame.copy()
                pt = self._prof("imagen", pt)

                info = self.process_hands(self.hands, frame_clean, new_result)
                self.stats.update_hands(self.hands, info["primary_tip"],
                                        info["gestures"][0] if self.hands else None,
                                        info["drawing"], new_result)
                self.stats.update_face(face, new_face)
                pt = self._prof("logica", pt)

                self.draw_hud(frame, self.hands, face, info)
                self.toast.draw(frame)
                pt = self._prof("hud", pt)

                if self.args.snapshot:
                    self.last_frame = frame
                if self.args.headless:
                    continue
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key, frame_clean):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.shutdown()

    def draw_hud(self, frame, hands, face, info):
        if not self.show_hud:
            for i, hand in enumerate(hands[:2]):
                p = info["pointers"][i]
                if p is not None and info["gestures"][i] == "point":
                    cv2.circle(frame, p, 8, (255, 255, 255), 2, cv2.LINE_AA)
            return
        for i, hand in enumerate(hands[:2]):
            ui.draw_hand(frame, hand, info["gestures"][i], info["pointers"][i],
                         hold_progress=info["hold_progress"][i], hold_color=info["hold_color"],
                         drawing=self.canvas.is_drawing(i))
        if info["pinch_preview"] is not None:
            col = (200, 200, 200) if self.eraser else C.PALETTE[self.color_idx][1]
            cv2.circle(frame, info["pinch_preview"], max(2, self.brush // 2), col, -1, cv2.LINE_AA)
            cv2.circle(frame, info["pinch_preview"], max(2, self.brush // 2) + 2, (255, 255, 255), 1, cv2.LINE_AA)
        if self.face is not None:
            ui.draw_face(frame, face)

        self.toolbar.draw(frame, hover=self.hover_item, progress=info["hover_progress"],
                          color_idx=self.color_idx, tool=self.tool, brush=self.brush)

        if info["mode_key"] == "nohand":
            text, col = "Muestra una mano a la cámara", C.UI_MUTED
        else:
            text, col = MODE_TEXT.get(info["mode_key"], ("", C.UI_MUTED))
        ui.draw_mode_banner(frame, text, col)

        px = C.FRAME_W - 330 - 10
        sh = 500
        ui.draw_stats_panel(frame, self.stats, self.canvas, hands, text, self.brush,
                            self.color_idx, self.eraser, px, 80, 330, sh)
        ui.draw_emotion_timeline(frame, self.stats, px, 80 + sh + 8, 330, C.FRAME_H - (80 + sh + 8) - 10)
        if self.face is not None:
            ui.draw_face_card(frame, face, self.stats, 16, C.FRAME_H - 160, 330, 150)
        if self.show_help:
            ui.draw_help(frame, 16, 98)

    def shutdown(self):
        self.worker.stop()
        if self.args.snapshot and self.last_frame is not None:
            cv2.imwrite(self.args.snapshot, self.last_frame)
            print(f"[snapshot] {self.args.snapshot}")
        self.canvas.end(0)
        self.canvas.end(1)
        path = self.stats.save(C.OUTPUT_DIR, self.canvas)
        d = self.stats.to_dict(self.canvas)
        print("\n===== RESUMEN DE LA SESION =====")
        print(f"Duracion: {d['duracion_s']} s   Frames: {d['frames']}   FPS medio: {d['fps_medio']}   "
              f"FPS inferencia: {self.stats.infer_fps:.1f}")
        print(f"Trazos: {d['dibujo'].get('trazos', 0)}   Puntos: {d['dibujo'].get('puntos', 0)}   "
              f"Tiempo dibujando: {d['dibujo']['tiempo_dibujando_s']} s")
        print(f"Recorrido del indice: {d['manos']['recorrido_indice_px']} px   "
              f"Vel. max: {d['manos']['velocidad_max_px_s']} px/s")
        print(f"Gestos: {d['manos']['gestos_detectados']}")
        print(f"Emocion dominante: {d['rostro']['emocion_dominante']}   "
              f"Distribucion: {d['rostro']['porcentaje_emociones']}")
        print(f"Parpadeos: {d['rostro']['parpadeos']} ({d['rostro']['parpadeos_por_min']}/min)   "
              f"Sonrisas: {d['rostro']['sonrisas']}")
        print(f"Estadisticas guardadas en: {path}")
        if self.args.profile and self.stats.frames:
            n = self.stats.frames
            print("Tiempo medio por frame de video (ms): "
                  + "  ".join(f"{k}={v / n * 1000:.1f}" for k, v in self.prof.items()))
        self.worker.join(timeout=2.0)
        self.cap.release()
        self.tracker.close()
        if self.face is not None:
            self.face.close()
        cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="AirSense: dibujo en el aire + gestos + emocion")
    ap.add_argument("--camera", type=int, default=None, help="indice de la camara (por defecto: auto)")
    ap.add_argument("--source", type=str, default=None, help="ruta a un video en lugar de la camara")
    ap.add_argument("--no-face", action="store_true", help="desactiva el analisis facial")
    ap.add_argument("--no-mirror", action="store_true", help="no voltear la imagen horizontalmente")
    ap.add_argument("--no-enhance", action="store_true", help="sin realce de nitidez del video")
    ap.add_argument("--hands", type=int, default=1, choices=[1, 2], help="numero maximo de manos (2 = ambas manos, mas lento)")
    ap.add_argument("--fullscreen", action="store_true", help="arrancar en pantalla completa (tecla 'f')")
    ap.add_argument("--version", action="version", version=f"AirSense {__version__}")
    ap.add_argument("--profile", action="store_true", help="imprime el tiempo medio por etapa al salir")
    ap.add_argument("--headless", action="store_true", help="(pruebas) no abrir ventana")
    ap.add_argument("--snapshot", type=str, default=None, help="(pruebas) guardar el ultimo frame con HUD en esta ruta")
    return ap.parse_args()


def main():
    args = parse_args()
    check_models(need_face=not args.no_face)
    try:
        app = App(args)
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
