"""Pictures of the tool designs of `flange_study.py`: the UR5e in the cabinet with each tool at the same target, at two headings.

    python scripts/render_tool_designs.py --out runs/tool_designs.png

Red arrow: the axis of joint 6 (wrist 3). Cabinet walls are drawn transparent; the boxes are example objects (not simulated)."""
import argparse
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import flange_study as F  # noqa: E402
from l2r.config import SimParams  # noqa: E402
from l2r.sim.workspace import Workspace  # noqa: E402

PADDLE = 0.035  # m: half width of the blade drawn at the tip


def vertical_rotation(heading):
    """Tool pointing down, its blade normal ('x' axis) horizontal at `heading` (0: towards the back, -x)."""
    z = np.array([0, 0, -1.0])
    x = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


DESIGNS = [  # (title, design, orientation of the tip for a heading, target depth override)
    ('A  current: straight 0.445 m\njoint 6 = roll about the tool axis', (0.0, 0.0, 0.445), F.target_rotation, None),
    ('B  L-bracket 0.10 + 0.60 m\njoint 6 vertical, tip 0.60 m off its axis', (0.10, 90.0, 0.60), F.target_rotation, None),
    ('C  tool pointing down 0.25 m (front target: the back is out of reach)\njoint 6 vertical through the tip',
     (0.0, 0.0, 0.25), vertical_rotation, 0.10),
]


def decorate(model):
    """Transparent cabinet, hidden object envelope, brighter light."""
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)
    model.vis.headlight.specular[:] = (0.1, 0.1, 0.1)
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name == 'scene_envelope':
            model.geom_rgba[g] = (0, 0, 0, 0)
        elif name in ('scene_back', 'scene_left', 'scene_right'):
            model.geom_rgba[g] = (0.75, 0.7, 0.65, 0.18)
        elif name == 'scene_ceiling':
            model.geom_rgba[g] = (0.75, 0.7, 0.65, 0.10)
        elif name == 'scene_floor':
            model.geom_rgba[g] = (0.8, 0.77, 0.72, 1.0)


def add_scene_extras(scene, model, data, objects):
    """Example objects, the joint-6 axis arrow and the blade at the tip, as decor geoms of the rendered scene."""
    def add(kind, size, pos, mat, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(),
                            np.asarray(rgba, np.float32))
        scene.ngeom += 1
    for pos, half, rgba in objects:
        add(mujoco.mjtGeom.mjGEOM_BOX, half, pos, np.eye(3), rgba)
    wrist = model.body('wrist_3_link').id
    joint = model.joint('wrist_3_joint').id
    axis = data.xmat[wrist].reshape(3, 3) @ model.jnt_axis[joint]
    start = data.xpos[wrist] + 0.02 * axis
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.array([0.9, 0.1, 0.1, 1], np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, start, start + 0.4 * axis)
    scene.ngeom += 1
    tip = model.site('tip').id
    R = data.site_xmat[tip].reshape(3, 3)
    add(mujoco.mjtGeom.mjGEOM_BOX, [0.004, PADDLE, 0.02], data.site_xpos[tip] - 0.02 * R[:, 2], R, [0.2, 0.2, 0.25, 1])


def pose(solver, target, rotation, rng, q_hint=None):
    """A collision-free IK solution of the tip pose (warm start from `q_hint` first, then several random starts)."""
    starts = ([q_hint] if q_hint is not None else []) + [F.HOME.copy()] + [
        rng.uniform(np.maximum(solver.low, -np.pi), np.minimum(solver.high, np.pi)) for _ in range(40)]
    best = None
    for q0 in starts:
        q0 = np.array(q0, float)
        q, ok = solver.ik(q0, target, rotation)
        if not ok:
            continue
        cabinet, envelope, _, margin = solver.check(q)
        score = (not cabinet, not envelope, margin)
        if best is None or score > best[0]:
            best = (score, q)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='runs/tool_designs.png')
    parser.add_argument('--x', type=float, default=-0.05, help='target depth (m, cabinet frame; back wall at -0.25)')
    parser.add_argument('--y', type=float, default=0.05)
    parser.add_argument('--headings', type=float, nargs='+', default=[0.0, 30.0])
    args = parser.parse_args()
    ws = Workspace.from_params(SimParams())
    floor = ws.bounds[2][0]
    target = np.array([args.x, args.y, floor + 0.05])
    objects = [((args.x - 0.08, args.y, floor + 0.06), (0.03, 0.03, 0.06), (0.2, 0.4, 0.9, 1)),
               ((0.05, -0.18, floor + 0.08), (0.03, 0.05, 0.08), (0.2, 0.75, 0.3, 1)),
               ((-0.15, 0.2, floor + 0.085), (0.035, 0.035, 0.085), (0.95, 0.8, 0.15, 1)),
               ((0.1, 0.18, floor + 0.05), (0.03, 0.03, 0.05), (0.2, 0.4, 0.9, 1))]
    rows, cols = len(args.headings), len(DESIGNS)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.7 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for c, (title, design, orientation, depth) in enumerate(DESIGNS):
        goal = target if depth is None else np.array([depth, target[1], target[2]])
        model = F.build(design, ws)
        decorate(model)
        solver = F.Solver(model)
        renderer = mujoco.Renderer(model, 600, 800)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.15, 0.0, floor + 0.12)
        camera.distance, camera.azimuth, camera.elevation = 1.85, 135.0, -28.0
        q_hint = None
        for r, heading in enumerate(args.headings):
            found = pose(solver, goal, orientation(np.radians(heading)), rng, q_hint)
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if found is None:
                ax.text(0.5, 0.5, 'no IK solution', ha='center', va='center', transform=ax.transAxes)
                continue
            (free, clear, _), q = found
            q_hint = q
            solver.d.qpos[:] = q
            mujoco.mj_forward(model, solver.d)
            renderer.update_scene(solver.d, camera=camera)
            add_scene_extras(renderer.scene, model, solver.d, objects)
            ax.imshow(renderer.render())
            touched = sorted({solver.scene.get(con.geom1) or solver.scene.get(con.geom2) for con in solver.d.contact[:solver.d.ncon]
                              if (solver.scene.get(con.geom1) or solver.scene.get(con.geom2)) not in (None, 'envelope') and con.dist <= 0})
            status = ('ok' if free else 'hits the cabinet ' + '/'.join(touched)) + ('' if clear else ', arm at object height')
            ax.set_title('%s\nheading %+.0f deg: %s' % (title, heading, status) if r == 0 else
                         'heading %+.0f deg: %s' % (heading, status), fontsize=10, loc='left')
        renderer.close()
    fig.tight_layout(h_pad=2.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
