"""Apertura robusta de la camara con deteccion automatica de indice y backend."""
import sys
import threading
import time

import cv2


class LatestFrameGrabber:
    """Lee la camara en un hilo aparte y entrega siempre el frame mas reciente.

    Sin esto, si el procesado va mas lento que la camara, OpenCV acumula frames
    en su buffer y el dibujo va con retraso respecto a la mano.
    """

    def __init__(self, cap):
        self.cap = cap
        self._cv = threading.Condition()
        self._frame = None
        self._seq = 0              # numero de frames recibidos de la camara
        self._read_seq = 0         # ultimo frame entregado a read()
        self._ok = True
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                fails = 0
                with self._cv:
                    self._frame = frame
                    self._seq += 1
                    self._cv.notify_all()
            else:
                fails += 1
                if fails > 60:
                    with self._cv:
                        self._ok = False
                        self._cv.notify_all()
                time.sleep(0.01)

    def read(self, timeout=1.0):
        """Espera un frame nuevo (no repetido) y lo devuelve.

        Asi el bucle principal va al ritmo real de la camara en lugar de procesar
        varias veces el mismo frame (FPS inflados y CPU desperdiciada).
        """
        with self._cv:
            self._cv.wait_for(lambda: self._seq != self._read_seq or not self._ok, timeout)
            if self._frame is None or self._seq == self._read_seq:
                return False, None
            self._read_seq = self._seq
            return True, self._frame

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


def open_camera(index=None, width=1280, height=720):
    """Devuelve un cv2.VideoCapture abierto o lanza RuntimeError."""
    backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
    if sys.platform != "win32":
        backends = [("ANY", cv2.CAP_ANY)]
    indices = [index] if index is not None else list(range(0, 4))

    for idx in indices:
        for name, be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[camara] abierta: indice {idx}, backend {name}, "
                      f"{frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()

    raise RuntimeError(
        "No se encontro ninguna camara. Comprueba que este conectada, que ninguna otra "
        "aplicacion la este usando y que Windows permita el acceso a la camara "
        "(Configuracion > Privacidad > Camara). Tambien puedes indicar el indice con --camera N "
        "o usar un video con --source ruta.mp4"
    )


def open_source(source, width=1280, height=720):
    """Abre un archivo de video (ruta) o la camara (None / entero)."""
    if source is None:
        return LatestFrameGrabber(open_camera(None, width, height))
    if isinstance(source, str) and source.isdigit():
        return LatestFrameGrabber(open_camera(int(source), width, height))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {source}")
    print(f"[video] abierto: {source}")
    return cap
