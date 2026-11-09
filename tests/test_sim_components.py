"""Simulator components against fake PyBullet backends: geometry, vision, arm, suction, force sensing."""
import numpy as np

from fakes import FakeJoints, ScriptedSensors
from l2r.config import SimParams
from l2r.sim.arm import Arm
from l2r.sim.arm_control import joint_limits, rate_limited_command
from l2r.sim.constants import COLORS, LABELS
from l2r.sim.force_sensing import ForceMonitor
from l2r.sim.suction import SuctionCup
from l2r.sim.vision import recolor_by_class, relabel_colormap
from l2r.sim.workspace import Workspace


def test_workspace_pixel_geometry():
    ws = Workspace()
    assert ws.heightmap_shape == (120, 100)  # rows = y (0.6 m), cols = x (0.5 m) at 5 mm
    x, y = ws.px_to_xy(10, 20)
    assert np.isclose(x, -0.25 + 0.05) and np.isclose(y, -0.3 + 0.1)
    assert ws.y_to_row(y) == 20
    assert Workspace.from_params(SimParams(cabinet_depth=0.4, cabinet_height=0.27)).heightmap_shape == (120, 80)


def test_force_monitor_removes_offset_rejects_spikes_and_debounces():
    base = [40, 0, 0, 0, 0, 0]  # static offset of 40 N (weight of the tool): must not trigger once tared
    spike = [40, 0, 400, 0, 0, 0]
    sequence = [base] * 4 + [spike] + [base] * 6 + [spike] * 4
    monitor = ForceMonitor(ScriptedSensors(sequence), 0, [0], window=3, debounce=2)
    monitor.tare(samples=3)
    assert np.allclose(monitor.bias[0], base)
    assert not monitor.update(100.0)  # offset removed
    assert not monitor.update(100.0)  # a single-sample spike (400 N) is rejected by the median filter
    for _ in range(6):
        assert not monitor.update(100.0)
    hits = [monitor.update(100.0) for _ in range(4)]  # sustained contact is reported
    assert hits[-1] and not hits[0]


def test_recolouring_is_robust_against_ids_that_look_like_labels():
    # object ids 3 and 5 are also class labels (green = 3, yellow = 5): in-place relabelling could confuse them
    names = {3: 'yellow CrackerBox', 5: 'blue ChipsCan', 9: 'green MustardBottle'}
    segm = np.zeros((6, 8), np.int64)
    segm[0:2, 0:2], segm[2:4, 2:4], segm[4:6, 4:6] = 3, 5, 9
    segm[0:2, 6:8] = 1  # robot / cabinet: not an object
    color, objects = recolor_by_class(np.zeros((6, 8, 3), np.uint8), segm, names)
    assert (color[0, 6] == 255).all() and (objects[0, 6] == 0).all()  # background white, not an object
    for uid, label in ((3, 'yellow'), (5, 'blue'), (9, 'green')):
        region = segm == uid
        assert (objects[region] == uid).all() and (color[region] == color[region][0]).all()
        assert (color[region][0] == np.multiply(COLORS[label][:3], 255)).all()


def test_relabel_colormap_recolours_the_unpadded_window():
    colormap = np.zeros((12, 10, 3), np.uint8)  # padded by 4 px (rows both sides, columns left)
    id_map = np.zeros((4, 6), np.float32)  # un-padded
    id_map[1:3, 2:4] = 7
    out = relabel_colormap(colormap, id_map, [7], [LABELS['green']])
    assert (colormap == 0).all()  # the stored original is not modified
    assert (out[4 + 1:4 + 3, 4 + 2:4 + 4] == np.multiply((89 / 255, 169 / 255, 79 / 255), 255)).all()
    assert out.sum() == out[5:7, 6:8].sum()


def make_arm(params=None, backend=None):
    arm = Arm(backend or FakeJoints(), params or SimParams(), Workspace(), 1 / 480)
    arm.body, arm.ee_link, arm.joints = 0, 9, list(range(7))
    arm.limits = joint_limits(arm.p, 0, arm.joints)
    return arm


def test_arm_waypoints_legacy_spacing_and_adaptive_coarsening():
    a, b = np.array([0.35, 0.0, 0.45]), np.array([-0.1, 0.0, 0.45])
    legacy = make_arm().waypoints(a, b)
    assert len(legacy) == int(np.floor(np.linalg.norm(b - a) / 0.002))  # same count as the original loop
    assert np.allclose(legacy[1] - legacy[0], [-0.002, 0, 0], atol=1e-9)
    arm = make_arm(SimParams(adaptive_step=True))
    monitored, retreat = arm.waypoints(tuple(a), tuple(b), monitored=True), arm.waypoints(tuple(a), tuple(b), monitored=False)
    xs = np.array([point[0] for point in monitored])
    gaps = np.abs(np.diff(xs))
    assert gaps[xs[:-1] > 0.27].max() > 0.0099 and gaps[xs[:-1] < 0.25].max() < 0.0021  # fine inside the cabinet
    assert len(retreat) <= len(legacy) // 5 + 1 and len(monitored) < len(legacy)


def test_ik_arguments_use_urdf_limits_and_limit_the_base_slide():
    arm = make_arm()
    arm.joints = [0, 1, 2]
    kwargs = arm._ik_kwargs()
    assert kwargs['lowerLimits'][0] == -0.5 and kwargs['upperLimits'][0] == 0.5
    assert kwargs['jointRanges'][1] == 12.56 and len(kwargs['restPoses']) == 3


def test_rate_limited_controller_never_jumps_and_does_not_overshoot():
    command = rate_limited_command(np.zeros(3), np.array([1.0, -1.0, 0.001]), np.array([2.0, 2.0, 2.0]), 0.01)
    assert np.allclose(command, [0.02, -0.02, 0.001])  # at most v_max * dt, never past the goal
    backend = FakeJoints()
    arm = make_arm(SimParams(controller='pd'), backend)
    limits = arm.limits
    assert limits['upper'][0] == 0.5 and limits['v_max'][0] == 1.0 and limits['effort'][0] == 1000.0
    goal = np.array([0.1, 0.5, -0.4, 0.3, 0.0, 0.2, np.pi])
    assert arm.move_joints(goal)
    assert np.all(np.abs(backend.q[:6] - goal[:6]) < 1e-3)
    steps = np.array(backend.history)
    assert np.all(np.diff(steps, axis=0)[:, 1] <= limits['v_max'][1] / 480 + 1e-9)  # velocity limit respected
    assert np.all(steps[:, 1] <= goal[1] + 1e-6)  # no overshoot
    assert np.max(np.abs(np.diff(steps, axis=0))) < abs(goal[1]) / 20  # the legacy command is the whole error at once


def test_suction_hold_verification():
    backend = FakeJoints()
    arm = make_arm(backend=backend)
    cup = SuctionCup(backend, SimParams(), arm, scene=None)
    cup.activated, cup.constraint, cup.grip_body, cup.grip_distance = True, 1, 7, 0.05  # grasped 5 cm from the cup
    backend.object_position = (0.5, 0.0, 0.0)  # tool frame is at z = 0.05
    assert cup.holds()
    backend.object_position = (0.5, 0.0, -0.1)  # the object sagged: grasp lost
    assert not cup.holds()


def test_force_monitor_reports_the_filtered_force():
    monitor = ForceMonitor(ScriptedSensors([[0, 0, 0, 0, 0, 0]] * 5 + [[0, 0, 200, 0, 0, 0]] * 6), 0, [0], window=3, debounce=2)
    monitor.tare(samples=3)
    monitor.update(500.0)
    assert monitor.last_force == 0.0
    for _ in range(4):
        monitor.update(500.0)
    assert monitor.last_force == 200.0  # median of the last three samples


def _random_scene(seed, rows=70, cols=90, objects=5):
    """Random point cloud + object ids with the structure of a camera image (blobs of ids, random heights)."""
    rng = np.random.default_rng(seed)
    ws = Workspace()
    b = ws.bounds
    points = np.stack([rng.uniform(b[0, 0] - 0.02, b[0, 1] + 0.02, (rows, cols)),
                       rng.uniform(b[1, 0] - 0.02, b[1, 1] + 0.02, (rows, cols)),
                       rng.uniform(b[2, 0], b[2, 1], (rows, cols))], axis=-1).astype(np.float32)
    segm = np.zeros((rows, cols), np.int64)
    for uid in rng.choice([3, 5, 7, 9, 11, 13], objects, replace=False):  # blobs; 1 = robot (not an object)
        r, c = rng.integers(0, rows - 20), rng.integers(0, cols - 25)
        segm[r:r + rng.integers(5, 20), c:c + rng.integers(5, 25)] = uid
    segm[rng.random((rows, cols)) < 0.05] = 1
    names = {3: 'blue A', 5: 'yellow B', 7: 'green C', 9: 'blue D', 11: 'yellow E', 13: 'green F'}
    return ws, points, segm, names


def test_single_pass_object_maps_equal_the_per_object_reference():
    import reference_vision as ref
    from l2r.sim.vision import object_maps, recolor_by_class
    for seed in range(12):
        ws, points, segm, names = _random_scene(seed)
        rows = ([8, 40, 75], [30, 62, 100], None)
        color_new, objects = recolor_by_class(np.zeros(segm.shape + (3,), np.uint8), segm.copy(), names)
        color_ref, masks, labels, colors, ids = ref.class_masks(np.zeros(segm.shape + (3,), np.uint8), segm.copy(), names)
        assert (color_new == color_ref).all()
        expected = ref.object_maps(points, masks, labels, colors, ids, ws, rows)
        for got, want in zip(object_maps(points, objects, names, ws, rows)[:4], expected):
            assert np.array_equal(got, want), seed
