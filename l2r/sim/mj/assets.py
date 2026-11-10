"""Assets for the MuJoCo backend, from the same files the PyBullet backend loads.

* The UR5e URDF: cleaned for MuJoCo's stricter parser (comma-separated colours), with every link kept as a body (no fusing of
  fixed joints), so PyBullet link index L is the arm's L-th body after its root.
* `urdf_joint_table`: the joints in URDF order with the fields of PyBullet's getJointInfo (indices, types, limits, effort,
  velocity), so code written against getJointInfo works unchanged.
* YCB models: mass, inertia, visual mesh and the convex parts of the V-HACD collision mesh (MuJoCo makes one convex hull of a
  mesh, so each hull of the decomposition is written to its own OBJ file, cached under ~/.cache/l2r_mujoco)."""
import hashlib
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from l2r.sim.mj.transforms import quat_from_euler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARM_URDF = os.path.join(ROOT, 'assets', 'ur5e', 'ur5e.urdf')
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'l2r_mujoco')

# PyBullet joint type codes (getJointInfo[2])
JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED = 0, 1, 2, 3, 4
_TYPES = {'revolute': JOINT_REVOLUTE, 'continuous': JOINT_REVOLUTE, 'prismatic': JOINT_PRISMATIC, 'fixed': JOINT_FIXED,
          'floating': JOINT_FIXED, 'planar': JOINT_PLANAR}


def _floats(text, default):
    return tuple(float(v) for v in (text or default).replace(',', ' ').split())


def arm_urdf_text(base_position=(0.0, 0.0, 0.0)):
    """The UR5e URDF as MuJoCo-ready text, its root placed at `base_position` (PyBullet's basePosition)."""
    with open(ARM_URDF) as f:
        text = f.read()
    text = re.sub(r'rgba="([^"]*)"', lambda m: 'rgba="%s"' % m.group(1).replace(',', ' '), text)
    meshdir = os.path.dirname(ARM_URDF)
    compiler = ('<mujoco><compiler meshdir="%s" fusestatic="false" discardvisual="false" strippath="false" '
                'balanceinertia="true"/></mujoco>' % meshdir)
    text = re.sub(r'(<robot[^>]*>)', r'\1' + compiler, text, count=1)
    # the root link 'world' becomes MuJoCo's world: carry the base position into the first joint's origin
    root = ET.fromstring(text)
    first = root.find('joint')
    origin = first.find('origin')
    xyz = np.array(_floats(origin.get('xyz') if origin is not None else None, '0 0 0')) + np.asarray(base_position, float)
    name = first.get('name')
    pattern = r'(<joint name="%s"[^>]*>.*?<origin[^>]*xyz=")([^"]*)(")' % re.escape(name)
    return re.sub(pattern, lambda m: m.group(1) + '%.6f %.6f %.6f' % tuple(xyz) + m.group(3), text, count=1, flags=re.S)


def urdf_joint_table(path=ARM_URDF):
    """[(index, name, type, lower, upper, effort, velocity, child link, parent index)] in URDF order (PyBullet's)."""
    root = ET.parse(path).getroot()
    joints = root.findall('joint')
    child_index = {j.find('child').get('link'): i for i, j in enumerate(joints)}
    table = []
    for i, joint in enumerate(joints):
        limit = joint.find('limit')
        kind = _TYPES[joint.get('type')]
        lower = float(limit.get('lower', 0.0)) if limit is not None else 0.0
        upper = float(limit.get('upper', -1.0)) if limit is not None else -1.0
        effort = float(limit.get('effort', 0.0)) if limit is not None else 0.0
        velocity = float(limit.get('velocity', 0.0)) if limit is not None else 0.0
        if kind == JOINT_FIXED and limit is None:
            lower, upper = 0.0, -1.0
        table.append((i, joint.get('name'), kind, lower, upper, effort, velocity, joint.find('child').get('link'),
                      child_index.get(joint.find('parent').get('link'), -1)))
    return table


def _split_obj(path, scale):
    """Write each 'o' group of an OBJ file to its own file (vertices re-indexed); returns the list of paths."""
    with open(path) as f:
        lines = f.readlines()
    key = hashlib.md5((path + repr(scale)).encode()).hexdigest()[:12]
    out_dir = os.path.join(CACHE, key)
    done = os.path.join(out_dir, 'done')
    if os.path.exists(done):
        with open(done) as f:
            return [os.path.join(out_dir, name) for name in f.read().split()]
    vertices, parts, current = [], [], None
    for line in lines:
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('o ') or line.startswith('g '):
            current = []
            parts.append(current)
        elif line.startswith('f '):
            if current is None:
                current = []
                parts.append(current)
            current.append([int(tok.split('/')[0]) for tok in line.split()[1:]])
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for k, faces in enumerate(p for p in parts if p):
        used = sorted({v for face in faces for v in face})
        remap = {v: n + 1 for n, v in enumerate(used)}
        name = 'hull_%02d.obj' % k
        with open(os.path.join(out_dir, name), 'w') as f:
            for v in used:
                f.write(vertices[v - 1])
            for face in faces:
                f.write('f %s\n' % ' '.join(str(remap[v]) for v in face))
        names.append(name)
    with open(done, 'w') as f:
        f.write(' '.join(names))
    return [os.path.join(out_dir, name) for name in names]


def ycb_model(kind):
    """{'mass', 'ipos', 'inertia', 'friction', 'visual': (path, scale), 'hulls': [paths], 'scale'} of a YCB model URDF."""
    directory = os.path.join(ROOT, 'assets', 'ycb_objects', 'Ycb' + kind)
    link = ET.parse(os.path.join(directory, 'model.urdf')).getroot().find('link')
    inertial = link.find('inertial')
    origin = inertial.find('origin')
    inertia = inertial.find('inertia')
    visual_mesh = link.find('visual/geometry/mesh')
    collision_mesh = link.find('collision/geometry/mesh')
    friction = link.find('contact/lateral_friction')
    scale = _floats(collision_mesh.get('scale'), '1 1 1')

    def origin_of(element):
        o = link.find(element + '/origin')
        return (_floats(o.get('xyz') if o is not None else None, '0 0 0'),
                quat_from_euler(_floats(o.get('rpy') if o is not None else None, '0 0 0')))
    return {
        'mass': float(inertial.find('mass').get('value')),
        'ipos': _floats(origin.get('xyz') if origin is not None else None, '0 0 0'),
        'iquat': quat_from_euler(_floats(origin.get('rpy') if origin is not None else None, '0 0 0')),
        'inertia': tuple(float(inertia.get(k)) for k in ('ixx', 'iyy', 'izz')),
        'friction': float(friction.get('value')) if friction is not None else 0.5,
        'visual': (os.path.join(directory, visual_mesh.get('filename')), _floats(visual_mesh.get('scale'), '1 1 1')),
        'hulls': _split_obj(os.path.join(directory, collision_mesh.get('filename')), scale),
        'scale': scale,
        'collision_origin': origin_of('collision'),  # (xyz, quaternion x y z w) of the meshes in the link frame
        'visual_origin': origin_of('visual'),
    }
