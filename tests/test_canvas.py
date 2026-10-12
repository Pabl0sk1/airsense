import cv2

from airsense.canvas import Canvas


def draw_line(c, hand, pts, color=(0, 0, 255), thickness=6, eraser=False):
    c.begin(hand, color, thickness, eraser)
    for p in pts:
        c.add_point(hand, p)
    c.end(hand)


def test_stroke_paints_layer_and_mask():
    c = Canvas(200, 100)
    draw_line(c, 0, [(10, 50), (60, 50), (120, 50)])
    assert c.n_strokes == 1
    assert c.total_points == 3
    assert c.mask[50, 60] == 255
    assert tuple(c.layer[50, 60]) == (0, 0, 255)
    assert c.coverage() > 0


def test_jitter_points_are_ignored():
    c = Canvas(200, 100)
    draw_line(c, 0, [(10, 10), (10, 11), (11, 10), (40, 10)])
    assert c.total_points == 2


def test_single_point_stroke_draws_a_dot():
    c = Canvas(100, 100)
    draw_line(c, 0, [(50, 50)])
    assert c.mask[50, 50] == 255


def test_empty_stroke_is_discarded():
    c = Canvas(100, 100)
    c.begin(0, (255, 255, 255), 4)
    c.end(0)
    assert c.strokes == []


def test_eraser_removes_paint():
    c = Canvas(200, 100)
    draw_line(c, 0, [(10, 50), (190, 50)], thickness=10)
    draw_line(c, 0, [(100, 20), (100, 80)], thickness=10, eraser=True)
    assert c.mask[50, 100] == 0
    assert c.mask[50, 30] == 255


def test_undo_and_clear():
    c = Canvas(200, 100)
    draw_line(c, 0, [(10, 20), (190, 20)])
    draw_line(c, 0, [(10, 80), (190, 80)])
    assert c.undo()
    assert c.n_strokes == 1
    assert c.mask[80, 100] == 0 and c.mask[20, 100] == 255
    c.clear()
    assert c.n_strokes == 0 and c.coverage() == 0
    assert not c.undo()


def test_undo_skips_stroke_in_progress():
    c = Canvas(200, 100)
    draw_line(c, 0, [(10, 20), (190, 20)])
    c.begin(1, (255, 0, 0), 4)
    c.add_point(1, (10, 80))
    c.add_point(1, (100, 80))
    assert c.undo()                 # borra el trazo terminado, no el activo
    assert c.is_drawing(1)
    assert c.mask[20, 100] == 0


def test_two_hands_draw_independently():
    c = Canvas(200, 100)
    c.begin(0, (0, 0, 255), 4)
    c.begin(1, (0, 255, 0), 4)
    for x in range(10, 190, 20):
        c.add_point(0, (x, 20))
        c.add_point(1, (x, 80))
    c.end(0)
    c.end(1)
    assert c.n_strokes == 2


def test_render_overlays_only_painted_pixels():
    c = Canvas(100, 100)
    draw_line(c, 0, [(10, 50), (90, 50)], color=(0, 255, 0))
    frame = c.layer.copy()
    frame[:] = 100
    c.render(frame)
    assert tuple(frame[50, 50]) == (0, 255, 0)
    assert tuple(frame[5, 5]) == (100, 100, 100)


def test_save_writes_png_alpha_and_capture(tmp_path):
    c = Canvas(64, 48)
    draw_line(c, 0, [(5, 5), (60, 40)])
    paths = c.save(tmp_path, with_frame=c.layer.copy())
    assert len(paths) == 3 and all(p.exists() for p in paths)
    rgba = cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (48, 64, 4)
    assert (rgba[:, :, 3] == c.mask).all()
