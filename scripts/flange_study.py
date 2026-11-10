"""Tool / flange design study (kinematics only): which tool shape lets the UR5e reach the cabinet at every heading?

    scripts/fetch_menagerie.sh                      # once: the Menagerie UR5e (BSD-3) into assets/menagerie
    python scripts/flange_study.py --out runs/flange_study.json

Robot: the MuJoCo Menagerie UR5e on the lateral base slide of this project (base at x = 0.6, as the PyBullet URDF). Cabinet: the
walls of `Scene.build_cabinet`. An 'object envelope' box fills the cabinet from the floor to 18 cm (the tallest objects): an arm
link or a tool bracket inside it would sweep the objects (the horizontal shaft of the tool works at object height by design and
does not count).

Tool designs (in the flange frame, z = flange normal): `L1` along the flange normal, then a bend of `bend_deg`, then `L2` to the
tip. straight: bend 0 (the current tool is straight, 0.445 m); L-bracket: bend 90 (flange normal vertical, the tool horizontal,
joint 6 = heading); 45 degrees in between. The last 5 cm of the tool are its working tip and do not count as collisions.

Task: the tip at (x, y, z) inside the cabinet pointing horizontally into it at heading psi (0 = straight in, along -x), its
'down' axis vertical. A target counts as reachable if the whole approach line from the cabinet front to it is: IK solved in order
along the line (warm started, as a streaming controller would follow it), within the joint limits, without touching the cabinet.
Reported per design: reachable share, share that also stays out of the object envelope, manipulability, joint-limit margin, and
how the joints move when only the heading changes (share of the motion done by joint 6)."""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.sim.workspace import Workspace  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UR5E = os.path.join(ROOT, 'assets', 'menagerie', 'universal_robots_ur5e')
ENVELOPE_HEIGHT = 0.18  # m above the cabinet floor: the tallest objects
TIP_FREE = 0.05  # m of the tool that may touch (its working end)
HOME = np.array([0, 1, -1, 0.5, 0.5, 0.5, 1]) * np.pi  # the project's home (slide + 6 joints)

DESIGNS = {  # name -> (L1, bend_deg, L2)
    'straight 0.445 (current)': (0.0, 0.0, 0.445),
    'straight 0.60': (0.0, 0.0, 0.60),
    'L 0.10 + 0.45': (0.10, 90.0, 0.45),
    'L 0.10 + 0.60': (0.10, 90.0, 0.60),
    'L 0.10 + 0.75': (0.10, 90.0, 0.75),
    'L 0.20 + 0.60': (0.20, 90.0, 0.60),
    '45deg 0.10 + 0.50': (0.10, 45.0, 0.50),
    '45deg 0.10 + 0.65': (0.10, 45.0, 0.65),
}


def tool_frame(L1, bend_deg, L2):
    """(tip position, tip rotation) in the flange frame. Tip axes: z = pointing direction, x = 'down' when the tool is horizontal."""
    b = np.radians(bend_deg)
    zf, xf = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    z_tip = np.cos(b) * zf + np.sin(b) * xf
    x_tip = np.sin(b) * zf - np.cos(b) * xf
    y_tip = np.cross(z_tip, x_tip)
    return L1 * zf + L2 * z_tip, np.column_stack([x_tip, y_tip, z_tip])


def build(design, ws):
    L1, bend, L2 = design
    with open(os.path.join(UR5E, 'ur5e.xml')) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', 'meshdir="%s"' % os.path.join(UR5E, 'assets'))
    xml = xml.replace('<body name="base" quat="0 0 0 -1" childclass="ur5e">',
                      '<body name="slide" pos="0.6 0 0"><joint name="slide" type="slide" axis="0 1 0" range="-0.5 0.5"/>'
                      '<body name="base" quat="1 0 0 0" childclass="ur5e">', 1)
    xml = xml.replace('</worldbody>', '</body></worldbody>', 1)
    spec = mujoco.MjSpec.from_string(xml)
    for key in list(spec.keys):
        spec.delete(key)
    wrist = spec.body('wrist_3_link')
    site = spec.site('attachment_site')
    tool = wrist.add_body(name='tool', pos=list(site.pos), quat=list(site.quat))
    tip_pos, tip_rot = tool_frame(L1, bend, L2)
    bend_point = np.array([0, 0, L1])
    direction = (tip_pos - bend_point) / np.linalg.norm(tip_pos - bend_point)
    if L1 > 0:
        tool.add_geom(name='tool_bracket', type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *bend_point], size=[0.02, 0, 0])
    shaft_end = tip_pos - TIP_FREE * direction
    tool.add_geom(name='tool_shaft', type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*bend_point, *shaft_end], size=[0.015, 0, 0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, tip_rot.flatten())
    tool.add_site(name='tip', pos=list(tip_pos), quat=list(q))
    for g in spec.geoms:  # robot and tool collide with the scene only
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = 2, 0
    b = ws.bounds
    sx, sy = ws.scene_size
    w, height, bottom = 0.2, b[2][1] - b[2][0], b[2][0]
    side_half = height / 2 + bottom / 2
    walls = {'floor': ([sx / 2, sy / 2, bottom / 2], [0, 0, bottom / 2]),
             'back': ([w / 2, sy / 2 + w, side_half], [-sx / 2 - w / 2 - 0.001, 0, side_half]),
             'left': ([sx / 2, w / 2, side_half], [0, -w / 2 - sy / 2 - 0.001, height / 2 + bottom / 2]),
             'right': ([sx / 2, w / 2, side_half], [0, w / 2 + sy / 2 + 0.001, height / 2 + bottom / 2]),
             'ceiling': ([sx / 2 + w / 2, sy / 2 + w, 0.025], [-w / 2, 0, height + bottom + 0.025]),
             'envelope': ([sx / 2, sy / 2, ENVELOPE_HEIGHT / 2], [0, 0, bottom + ENVELOPE_HEIGHT / 2])}
    for name, (half, centre) in walls.items():
        spec.worldbody.add_geom(name='scene_' + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=centre,
                                contype=0, conaffinity=2, rgba=[0.7, 0.7, 0.7, 0.3])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200  # pictures (render_tool_designs.py)
    model = spec.compile()
    return model


class Solver:
    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.site = model.site('tip').id
        self.low, self.high = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
        self.scene = {model.geom('scene_' + n).id: n for n in ('floor', 'back', 'left', 'right', 'ceiling', 'envelope')}
        self.shaft = model.geom('tool_shaft').id

    def ik(self, q0, pos, rot, iterations=300):
        m, d = self.m, self.d
        d.qpos[:] = q0
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target_quat, current_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rot.flatten())
        for _ in range(iterations):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_pos = pos - d.site_xpos[self.site]
            mujoco.mju_mat2Quat(current_quat, d.site_xmat[self.site])
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, target_quat, current_quat)  # in the current tip frame
            err_rot = d.site_xmat[self.site].reshape(3, 3) @ err_rot  # world frame, as the Jacobian
            if np.linalg.norm(err_pos) < 1e-3 and np.linalg.norm(err_rot) < 1e-2:
                return d.qpos.copy(), True
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J, e = np.vstack([jp, jr]), np.concatenate([err_pos, 0.3 * err_rot])
            J[3:] *= 0.3
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            scale = max(1.0, np.abs(dq).max() / 0.2)
            d.qpos[:] = np.clip(d.qpos + dq / scale, self.low, self.high)
        return d.qpos.copy(), False

    def check(self, q):
        """(collides with the cabinet, inside the object envelope, manipulability, joint-limit margin)."""
        m, d = self.m, self.d
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cabinet = envelope = False
        for i in range(d.ncon):
            c = d.contact[i]
            name = self.scene.get(c.geom1) or self.scene.get(c.geom2)
            if name is None or c.dist > 0:
                continue
            if name == 'envelope':
                other = c.geom2 if c.geom1 in self.scene else c.geom1
                envelope |= other != self.shaft  # the horizontal shaft works at object height by design
            else:
                cabinet = True
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, self.site)
        manip = float(np.sqrt(max(np.linalg.det(jp @ jp.T), 0.0)))
        span = self.high - self.low
        margin = float(np.min(np.minimum(q - self.low, self.high - q) / span))
        return cabinet, envelope, manip, margin


def target_rotation(heading):
    """Tip axes for a horizontal tool pointing into the cabinet at `heading` (0: along -x), 'down' axis vertical."""
    z = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    x = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def staging(solver, pos, rot, rng, starts=12):
    """Best collision-free IK solution of a pose from several starts (the home pose first)."""
    best = None
    for k in range(starts):
        q0 = HOME.copy() if k == 0 else rng.uniform(np.maximum(solver.low, -np.pi), np.minimum(solver.high, np.pi))
        q0[0] = np.clip(pos[1], -0.5, 0.5)
        q, ok = solver.ik(q0, pos, rot)
        if not ok:
            continue
        cabinet, _, _, margin = solver.check(q)
        if cabinet:
            continue
        if best is None or margin > best[1]:
            best = (q, margin)
    return None if best is None else best[0]


def approach(solver, target, heading, rng, front_x, steps=5):
    """Solve the approach line from the cabinet front to `target` in order; (joint solutions, envelope hit, manip, margin) or None."""
    rot = target_rotation(heading)
    direction = rot[:, 2]
    distance = (front_x - target[0]) / max(-direction[0], 1e-6)
    start = target - direction * distance
    q = staging(solver, start, rot, rng)
    if q is None:
        return None
    envelope, path = False, [q]
    for s in range(1, steps + 1):
        point = start + (target - start) * s / steps
        q, ok = solver.ik(q, point, rot)
        if not ok:
            return None
        cabinet, inside, manip, margin = solver.check(q)
        if cabinet:
            return None
        envelope |= inside
        path.append(q)
    return path, envelope, manip, margin


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='')
    parser.add_argument('--designs', nargs='*', default=None, help='subset of the design names')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    ws = Workspace.from_params(SimParams())
    b = ws.bounds
    xs = np.linspace(b[0][0] + 0.06, b[0][1] - 0.06, 5)
    ys = np.linspace(b[1][0] + 0.06, b[1][1] - 0.06, 5)
    zs = [b[2][0] + 0.04, b[2][0] + 0.10]
    headings = np.radians([-60, -30, 0, 30, 60])
    front_x = b[0][1] + 0.05
    # geometric bound: the approach line must cross the cabinet front inside the opening (2 cm from the side walls)
    opening = {}
    for heading in headings:
        direction = target_rotation(heading)[:, 2]
        crossings = [y - direction[1] * (b[0][1] - x) / -direction[0] for x, y in itertools.product(xs, ys)]
        opening[round(float(np.degrees(heading)))] = float(np.mean([b[1][0] + 0.02 <= c <= b[1][1] - 0.02 for c in crossings]))
    print('through the opening (upper bound) by heading', {h: round(v, 2) for h, v in opening.items()},
          'all %.2f' % np.mean(list(opening.values())))
    results = {}
    for name, design in DESIGNS.items():
        if args.designs and name not in args.designs:
            continue
        solver, rng = Solver(build(design, ws)), np.random.default_rng(args.seed)
        cells = []
        for x, y, z in itertools.product(xs, ys, zs):
            per_heading = {}
            for heading in headings:
                out = approach(solver, np.array([x, y, z]), heading, rng, front_x)
                per_heading[round(float(np.degrees(heading)))] = out
            # heading changes at the target: joint motion from the 0-degree solution, warm started
            motion = []
            if per_heading[0] is not None:
                q_ref = per_heading[0][0][-1]
                for heading in headings:
                    if heading == 0:
                        continue
                    q, ok = solver.ik(q_ref, np.array([x, y, z]), target_rotation(heading))
                    if ok and not solver.check(q)[0]:
                        dq = np.abs(q - q_ref)
                        motion.append((float(dq.sum()), float(dq[-1] / max(dq.sum(), 1e-9))))
            cells.append({'x': float(x), 'y': float(y), 'z': float(z),
                          'reachable': {h: v is not None for h, v in per_heading.items()},
                          'clear_of_objects': {h: (v is not None and not v[1]) for h, v in per_heading.items()},
                          'manipulability': {h: (v[2] if v else None) for h, v in per_heading.items()},
                          'limit_margin': {h: (v[3] if v else None) for h, v in per_heading.items()},
                          'heading_motion': motion})
        flat = lambda key: [c[key][h] for c in cells for h in c[key]]
        reach, clear = flat('reachable'), flat('clear_of_objects')
        manip = [v for v in flat('manipulability') if v is not None]
        margin = [v for v in flat('limit_margin') if v is not None]
        motion = [m for c in cells for m in c['heading_motion']]
        by_heading = {h: float(np.mean([c['reachable'][h] for c in cells])) for h in cells[0]['reachable']}
        summary = {'reachable': float(np.mean(reach)), 'reachable and clear of objects': float(np.mean(clear)),
                   'reachable by heading': by_heading,
                   'manipulability (mean)': float(np.mean(manip)) if manip else None,
                   'joint-limit margin (mean)': float(np.mean(margin)) if margin else None,
                   'joint motion per heading change (rad, mean)': float(np.mean([m[0] for m in motion])) if motion else None,
                   'share of that motion by joint 6': float(np.mean([m[1] for m in motion])) if motion else None}
        results[name] = {'design': dict(zip(('L1', 'bend_deg', 'L2'), design)), 'summary': summary, 'cells': cells}
        print('%-26s reachable %.2f  clear %.2f  by heading %s  manip %s  heading-change motion %s rad, joint-6 share %s' % (
            name, summary['reachable'], summary['reachable and clear of objects'],
            {h: round(v, 2) for h, v in by_heading.items()},
            None if summary['manipulability (mean)'] is None else round(summary['manipulability (mean)'], 3),
            None if summary['joint motion per heading change (rad, mean)'] is None else round(summary['joint motion per heading change (rad, mean)'], 2),
            None if summary['share of that motion by joint 6'] is None else round(summary['share of that motion by joint 6'], 2)),
            flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'grid': {'x': xs.tolist(), 'y': ys.tolist(), 'z': zs, 'headings_deg': [-60, -30, 0, 30, 60]},
                       'through the opening': opening, 'results': results}, f, indent=1, default=float)


if __name__ == '__main__':
    main()
