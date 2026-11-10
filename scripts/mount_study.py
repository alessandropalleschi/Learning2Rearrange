"""Where to mount the arm for the L tool: base x (distance behind the cabinet front) and height (pedestal).

    python scripts/mount_study.py

For each base: the home posture (the staging pose in front of the cabinet, elbow up, best dexterity), then every approach line
of the flange study grid solved from the home with the controller's null-space pull (no random restarts, as the controller runs
it), collision checked against the cabinet. Reported: reachable share (of the lines through the opening), dexterity (smallest
singular value of the position Jacobian, low = near a singularity) and how far the joints stray from the home."""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.sim.mj.control import Kinematics  # noqa: E402
from l2r.sim.mj.robot import ToolGeometry  # noqa: E402
from l2r.sim.mj.world import world_spec  # noqa: E402
from l2r.sim.workspace import Workspace  # noqa: E402


def horizontal(heading):
    z = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    x = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


class Probe:
    def __init__(self, model, kin):
        self.m, self.d, self.k = model, mujoco.MjData(model), kin

    def free(self, q):
        self.d.qpos[self.k.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        return not any(c.dist < 0 for c in self.d.contact[:self.d.ncon])

    def dexterity(self, q):
        return float(np.linalg.svd(self.k.jacobian(q)[:3], compute_uv=False)[-1])

    def elbow_up(self, q):
        self.d.qpos[self.k.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        sh, el, wr = (self.d.xpos[self.m.body(n).id] for n in ('upper_arm_link', 'forearm_link', 'wrist_1_link'))
        t = np.clip(np.dot(el - sh, wr - sh) / max(np.dot(wr - sh, wr - sh), 1e-9), 0, 1)
        return float(el[2] - (sh + t * (wr - sh))[2])


def find_home(kin, probe, stage, rot, rng, starts=200):
    """Collision-free, elbow-up solution of the staging pose with the best dexterity (slide centred)."""
    best = None
    for _ in range(starts):
        q0 = rng.uniform(np.maximum(kin.low, -np.pi), np.minimum(kin.high, np.pi))
        q0[0] = 0.0
        kin.rest_gain = 0.0
        q, ok = kin.solve(q0, stage, rot)
        if not ok or not probe.free(q) or probe.elbow_up(q) < 0.05:
            continue
        q = np.concatenate([[q[0]], (q[1:] + np.pi) % (2 * np.pi) - np.pi])
        if abs(q[0]) > 0.05:
            continue  # the slide centred at home
        score = probe.dexterity(q)
        if best is None or score > best[0]:
            best = (score, q)
    kin.rest_gain = 0.05
    return None if best is None else best[1]


def evaluate(base, ws, rng, blade=0.75):
    model = world_spec(ws, tool=ToolGeometry(blade=blade), base=base).compile()
    kin = Kinematics(model)
    probe = Probe(model, kin)
    b, floor = ws.bounds, ws.bounds[2][0]
    front = b[0][1] + 0.1
    home = find_home(kin, probe, np.array([front, 0.0, floor + 0.05]), horizontal(0.0), rng)
    if home is None:
        return None
    kin.home = home
    xs = np.linspace(b[0][0] + 0.06, b[0][1] - 0.06, 5)
    ys = np.linspace(b[1][0] + 0.06, b[1][1] - 0.06, 5)
    total = reached = 0
    dexterity, stray = [], []
    for x, y, heading in itertools.product(xs, ys, np.radians([-60, -30, 0, 30, 60])):
        rot = horizontal(heading)
        direction = rot[:, 2]
        crossing = y - direction[1] * (b[0][1] - x) / -direction[0]
        if not (b[1][0] + 0.02 <= crossing <= b[1][1] - 0.02):
            continue  # the line does not pass the opening: no tool reaches it
        total += 1
        target = np.array([x, y, floor + 0.05])
        start = target - direction * ((front - x) / -direction[0])
        q, ok = kin.solve(home, start, rot)
        if not ok or not probe.free(q):
            continue
        fine = True
        for s in range(1, 6):
            q, ok = kin.solve(q, start + (target - start) * s / 5, rot)
            if not ok or not probe.free(q):
                fine = False
                break
            dexterity.append(probe.dexterity(q))
            stray.append(float(np.abs(q[1:] - home[1:]).max()))
        reached += fine
    return {'home': np.round(home, 2).tolist(), 'reachable': reached / total,
            'dexterity p10': float(np.percentile(dexterity, 10)) if dexterity else None,
            'dexterity median': float(np.median(dexterity)) if dexterity else None,
            'max joint stray from home (rad, median)': float(np.median(stray)) if stray else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--x', type=float, nargs='+', default=[0.6, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument('--blade', type=float, nargs='+', default=[0.75], help='blade length (m)')
    parser.add_argument('--y', type=float, nargs='+', default=[0.0], help='sideways offset of the base (slide centre)')
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 0.25, 0.45])
    args = parser.parse_args()
    ws = Workspace.from_params(SimParams())
    for blade, x, y, z in itertools.product(args.blade, args.x, args.y, args.z):
        out = evaluate((x, y, z), ws, np.random.default_rng(0), blade)
        if out is None:
            print('blade %.2f base x %.2f y %.2f z %.2f: no home posture' % (blade, x, y, z), flush=True)
            continue
        print('blade %.2f base x %.2f y %.2f z %.2f: reachable %.2f  dexterity p10 %s median %s  joint stray %s  home %s' % (
            blade, x, y, z, out['reachable'], *(None if out[k] is None else round(out[k], 3) for k in
                                      ('dexterity p10', 'dexterity median', 'max joint stray from home (rad, median)')),
            out['home']), flush=True)


if __name__ == '__main__':
    main()
