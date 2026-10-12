import threading
import time

import numpy as np

from airsense.camera import LatestFrameGrabber


class FakeCap:
    """Camara falsa a ~100 fps que numera cada frame en el pixel (0, 0)."""

    def __init__(self, fail=False):
        self.n = 0
        self.fail = fail
        self.released = threading.Event()

    def read(self):
        time.sleep(0.01)
        if self.fail:
            return False, None
        self.n += 1
        f = np.zeros((4, 4, 3), np.uint8)
        f[0, 0, 0] = self.n % 256
        return True, f

    def release(self):
        self.released.set()


def test_read_never_returns_the_same_frame_twice():
    g = LatestFrameGrabber(FakeCap())
    try:
        ids = []
        for _ in range(5):
            ok, f = g.read()
            assert ok
            ids.append(int(f[0, 0, 0]))
        assert len(set(ids)) == 5
        assert ids == sorted(ids)
    finally:
        g.release()


def test_read_times_out_when_camera_fails():
    cap = FakeCap(fail=True)
    g = LatestFrameGrabber(cap)
    ok, f = g.read(timeout=0.2)
    assert not ok and f is None
    g.release()
    assert cap.released.is_set()
