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
    elbow_high = False

    def __init__(self, model, kin):
        self.m, self.d, self.k = model, mujoco.MjData(model), kin

    def free(self, q):
        self.d.qpos[self.k.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        return not any(c.dist < 0 for c in self.d.contact[:self.d.ncon])

    def dexterity(self, q):
        return float(np.linalg.svd(self.k.jacobian(q)[:3], compute_uv=False)[-1])

    def facing(self, q):
        """The arm faces the cabinet: the elbow between the shoulder and the cabinet (smaller x) and above the shoulder."""
        self.d.qpos[self.k.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        shoulder, elbow = (self.d.xpos[self.m.body(n).id] for n in ('upper_arm_link', 'forearm_link'))
        if not (elbow[0] < shoulder[0] - 0.05 and elbow[2] > shoulder[2] + 0.1):
            return False
        if self.elbow_high:  # the classic pose: the elbow above the wrist, the forearm going down to it
            return bool(elbow[2] > self.d.xpos[self.m.body('wrist_1_link').id][2] + 0.05)
        return True

    def elbow_up(self, q):
        self.d.qpos[self.k.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        sh, el, wr = (self.d.xpos[self.m.body(n).id] for n in ('upper_arm_link', 'forearm_link', 'wrist_1_link'))
        t = np.clip(np.dot(el - sh, wr - sh) / max(np.dot(wr - sh, wr - sh), 1e-9), 0, 1)
        return float(el[2] - (sh + t * (wr - sh))[2])


def seed_grid(n_joints):
    """Deterministic seed postures (6 arm joints, slide prepended if present): shoulder pan around facing the cabinet and its
    mirror, several shoulder / elbow / wrist combinations, flange up or down."""
    grid = itertools.product([0.0, 0.3, 0.6, 0.9, 2.4, 2.8], [-2.6, -2.0, -1.4], [-2.4, -1.6, 1.6, 2.4], [-2.0, 0.0, 2.0],
                             [-1.5708, 1.5708], [0.0])
    arm = [np.array(g) for g in grid]
    return [np.concatenate([[0.0], a]) if n_joints == 7 else a for a in arm]


def candidate_homes(kin, probe, stage, rot, rng=None, facing=False):
    """Distinct collision-free solutions of the staging pose (slide centred), from the seed grid and the PyBullet home
    posture; with `facing`, only the arm facing the cabinet. Deterministic."""
    first = 1 if kin.names[0] == 'slide' else 0
    seeds = [(np.array([0, 1, -1, 0.5, 0.5, 0.5, 1]) * np.pi)[-len(kin.low):]] + seed_grid(len(kin.low))
    kin.rest_gain, found = 0.0, []
    for q0 in seeds:
        q, ok = kin.solve(np.array(q0, float), stage, rot)
        if not ok or not probe.free(q) or (first and abs(q[0]) > 0.05):
            continue
        if facing and not probe.facing(q):
            continue
        q = np.concatenate([q[:first], (q[first:] + np.pi) % (2 * np.pi) - np.pi])
        if all(np.abs(q - other).max() > 0.3 for other in found):
            found.append(q)
    kin.rest_gain = 0.05
    return found


def reach(kin, probe, home, lines):
    """Share of the approach lines solved in order from `home` without collisions; also the dexterity and stray samples."""
    kin.home = home
    reached, dexterity, stray = 0, [], []
    first = 1 if kin.names[0] == 'slide' else 0
    for start, target, rot in lines:
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
            stray.append(float(np.abs(q[first:] - home[first:]).max()))
        reached += fine
    return reached / max(len(lines), 1), dexterity, stray


def evaluate(base, ws, rng, blade=0.75, slide=True, shape='L', facing=False, stage_margin=0.1):
    model = world_spec(ws, tool=ToolGeometry(shape=shape, blade=blade), base=base, slide=slide).compile()
    kin = Kinematics(model)
    probe = Probe(model, kin)
    b, floor = ws.bounds, ws.bounds[2][0]
    front = b[0][1] + stage_margin
    lines = []
    xs = np.linspace(b[0][0] + 0.06, b[0][1] - 0.06, 5)
    ys = np.linspace(b[1][0] + 0.06, b[1][1] - 0.06, 5)
    for x, y, heading in itertools.product(xs, ys, np.radians([-60, -30, 0, 30, 60])):
        rot = horizontal(heading)
        direction = rot[:, 2]
        crossing = y - direction[1] * (b[0][1] - x) / -direction[0]
        if b[1][0] + 0.02 <= crossing <= b[1][1] - 0.02:  # lines that do not pass the opening are out for any tool
            target = np.array([x, y, floor + 0.05])
            lines.append((target - direction * ((front - x) / -direction[0]), target, rot))
    homes = candidate_homes(kin, probe, np.array([front, 0.0, floor + 0.05]), horizontal(0.0), rng, facing=facing)
    if not homes:
        return None
    coarse = lines[::7]  # pick the home that reaches the most of a subset, then evaluate it on all lines
    home = max(homes, key=lambda h: reach(kin, probe, h, coarse)[0])
    share, dexterity, stray = reach(kin, probe, home, lines)
    return {'home': np.round(home, 2).tolist(), 'reachable': share, 'homes tried': len(homes),
            'dexterity p10': float(np.percentile(dexterity, 10)) if dexterity else None,
            'dexterity median': float(np.median(dexterity)) if dexterity else None,
            'max joint stray from home (rad, median)': float(np.median(stray)) if stray else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--x', type=float, nargs='+', default=[0.6, 0.9, 1.1, 1.3, 1.5])
    parser.add_argument('--facing', action='store_true', help='only homes with the arm facing the cabinet (elbow up, in front)')
    parser.add_argument('--elbow_high', action='store_true', help='with --facing: the elbow above the wrist')
    parser.add_argument('--stage_margin', type=float, default=0.1, help='staging: the tip this far in front of the cabinet (m)')
    parser.add_argument('--shape', choices=['L', 'straight'], default='L')
    parser.add_argument('--no_slide', action='store_true', help='a fixed base (no lateral slide)')
    parser.add_argument('--blade', type=float, nargs='+', default=[0.75], help='blade length (m)')
    parser.add_argument('--y', type=float, nargs='+', default=[0.0], help='sideways offset of the base (slide centre)')
    parser.add_argument('--z', type=float, nargs='+', default=[0.0, 0.25, 0.45])
    args = parser.parse_args()
    ws = Workspace.from_params(SimParams())
    Probe.elbow_high = args.elbow_high
    for blade, x, y, z in itertools.product(args.blade, args.x, args.y, args.z):
        out = evaluate((x, y, z), ws, np.random.default_rng(0), blade, not args.no_slide, args.shape, args.facing, args.stage_margin)
        if out is None:
            print(args.shape + ' blade %.2f base x %.2f y %.2f z %.2f: no home posture' % (blade, x, y, z), flush=True)
            continue
        print(args.shape + ' blade %.2f base x %.2f y %.2f z %.2f: reachable %.2f  dexterity p10 %s median %s  joint stray %s  home %s' % (
            blade, x, y, z, out['reachable'], *(None if out[k] is None else round(out[k], 3) for k in
                                      ('dexterity p10', 'dexterity median', 'max joint stray from home (rad, median)')),
            out['home']), flush=True)


if __name__ == '__main__':
    main()
