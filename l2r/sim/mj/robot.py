"""The robot of the native MuJoCo backend: the MuJoCo Menagerie UR5e on the lateral base slide, with the L-shaped tool.

Tool (flange frame: z = flange normal): a bracket `drop` along the normal, then a horizontal blade of length `blade` along the
flange x axis, a suction cup at its end. With the flange pointing down, joint 6 turns the blade in the floor plane. The blade is
a thin vertical plate (pushes with its end or sideways with its face); the cup is its own body so that the suction (an adhesion
actuator) acts through the cup only. A force / torque sensor sits at the flange (the force between the tool and the wrist, as the
UR5e's built-in sensor measures it).

Geometry and mount from `scripts/mount_study.py` (one posture for the whole cabinet, the tool colliding with the arm): a 0.35 m
blade with the base 0.6 m in front of the cabinet reaches 0.90 of the approach lines through the opening with the arm facing the
cabinet (longer blades need the base further back and reach less: 0.75 m blade 0.57).
`scripts/fetch_menagerie.sh` downloads the arm model (BSD-3)."""
import os
from dataclasses import dataclass

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MENAGERIE = os.path.join(ROOT, 'assets', 'menagerie')
UR5E = os.path.join(MENAGERIE, 'universal_robots_ur5e')
BASE_POSITION = (0.85, 0.0, 0.50)  # 0.6 m in front of the cabinet, on a 0.5 m pedestal (scripts/mount_study.py)
SLIDE_RANGE = 0.5  # m either side
PEDESTAL_RADIUS = 0.1  # m: the pedestal under the robot base (a cylinder) and its mounting plate
ARM_JOINTS = ('slide', 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
              'wrist_3_joint')
STAGE_MARGIN = 0.02  # m: staging puts the tip this far in front of the cabinet opening
# Home (slide, 6 joints): the tip at the staging point in front of the cabinet (x = front + STAGE_MARGIN, centre, 5 cm above the
# cabinet floor), blade pointing in, flange down, the arm facing the cabinet with the elbow up. Every IK solve starts from it and is pulled towards it in the null
# space, so the arm keeps this posture at every heading and depth (no flips between solutions).
HOME = (0.0, 0.7, -2.34, -3.01, 0.63, 1.5708, -0.87)  # arm facing the cabinet, elbow above the wrist (mount_study.py --facing --elbow_high)
# Arms of the Menagerie: model file, the opening tag of the base body and its replacement (orientation in the world), the arm
# joints in order, the flange body / site, landmark bodies (shoulder, elbow, wrist) for posture tests, joint speed limits.
ARMS = {
    'ur5e': dict(dir='universal_robots_ur5e', xml='ur5e.xml',
                 top='<body name="base" quat="0 0 0 -1" childclass="ur5e">', top_new='<body name="base" quat="1 0 0 0" childclass="ur5e">',
                 joints=ARM_JOINTS[1:], flange='wrist_3_link', landmarks=('upper_arm_link', 'forearm_link', 'wrist_1_link'),
                 v_max=(3.14, 3.14, 3.14, 6.28, 6.28, 6.28), home=HOME[1:], base_housing=True),
    'fr3': dict(dir='franka_fr3', xml='fr3.xml',
                top='<body name="base" childclass="fr3">', top_new='<body name="base" quat="0 0 0 1" childclass="fr3">',
                joints=tuple('fr3_joint%d' % i for i in range(1, 8)), flange='fr3_link7',
                landmarks=('fr3_link2', 'fr3_link4', 'fr3_link6'), v_max=(2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26),
                home=None, base_housing=False),
}


def arm_of(model):
    """Name of the arm in a compiled model."""
    for name, arm in ARMS.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, arm['joints'][0]) >= 0:
            return name
    raise ValueError('no known arm in the model')


def joint_names(model):
    """Controlled joints: the slide (if present) and the arm joints."""
    arm = ARMS[arm_of(model)]['joints']
    slide = ('slide',) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide') >= 0 else ()
    return slide + tuple(arm)


ROBOT, SCENE = 2, 1  # collision bits: robot geoms contype ROBOT / conaffinity 0, scene geoms SCENE / SCENE | ROBOT:
# the robot touches the scene, not itself; scene geoms touch each other


@dataclass
class ToolGeometry:
    shape: str = 'L'  # 'L': bracket down then a horizontal blade; 'straight': the blade along the flange normal (no bracket)
    drop: float = 0.10  # m from the flange along its normal to the blade
    blade: float = 0.35  # m, horizontal (with the base at BASE_POSITION: 0.90 of the approach lines, mount_study.py)
    blade_height: float = 0.04
    blade_thickness: float = 0.008
    cup_radius: float = 0.015
    cup_length: float = 0.02
    mass: float = 0.6  # kg, the whole tool

    def axis(self):
        """(blade start, blade direction, the tip's 'x' axis) in the flange frame."""
        if self.shape == 'straight':
            return np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        return np.array([0, 0, self.drop]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])

    def tip(self):
        """Tip (cup face) position and axes in the flange frame: z = pointing direction (x = 'down' with the L flange down)."""
        start, direction, x_tip = self.axis()
        position = start + direction * (self.blade + self.cup_length)
        return position, np.column_stack([x_tip, np.cross(direction, x_tip), direction])


def _menagerie_spec(base=BASE_POSITION, slide=True, arm='ur5e'):
    info = ARMS[arm]
    path = os.path.join(MENAGERIE, info['dir'], info['xml'])
    if not os.path.exists(path):
        raise FileNotFoundError('the Menagerie %s is missing: run scripts/fetch_menagerie.sh' % arm)
    with open(path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', 'meshdir="%s"' % os.path.join(MENAGERIE, info['dir'], 'assets'))
    joint = ('<joint name="slide" type="slide" axis="0 1 0" range="%g %g" damping="50"/>' % (-SLIDE_RANGE, SLIDE_RANGE)
             if slide else '')
    assert info['top'] in xml, 'unexpected %s model' % arm
    xml = xml.replace(info['top'], '<body name="slide_carriage" pos="%g %g %g">%s%s' % (*base, joint, info['top_new']), 1)
    xml = xml.replace('</worldbody>', '</body></worldbody>', 1)
    spec = mujoco.MjSpec.from_string(xml)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def robot_spec(tool=None, suction_force=30.0, base=BASE_POSITION, slide=True, arm='ur5e'):
    """MjSpec with the arm (its base at `base`, on a pedestal when raised), the slide actuator, gravity compensation, the
    tool, the flange sensor and the suction actuator."""
    tool = tool or ToolGeometry()
    spec = _menagerie_spec(base, slide, arm)
    info = ARMS[arm]
    for body in spec.bodies[1:]:
        body.gravcomp = 1.0  # the UR controller compensates gravity
    carriage = spec.body('slide_carriage')
    carriage.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PEDESTAL_RADIUS, 0.01, 0], pos=[0, 0, -0.01],
                      contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.3, 1], mass=10.0)  # mounting plate
    if base[2] > 0.02:  # a cylinder under the robot base only, riding with the slide (its rail on the ground)
        carriage.add_geom(name='pedestal', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PEDESTAL_RADIUS, base[2] / 2 - 0.01, 0],
                          pos=[0, 0, -base[2] / 2 - 0.01], rgba=[0.35, 0.35, 0.38, 1], mass=0.0)
    if slide:
        for side in (-1, 1):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, SLIDE_RANGE + PEDESTAL_RADIUS, 0.01],
                                    pos=[base[0] + side * 0.06, base[1], 0.01], rgba=[0.5, 0.5, 0.52, 1], contype=0, conaffinity=0)
    if slide:
        act = spec.add_actuator(name='slide', target='slide')
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0], act.biasprm[:3] = 20000.0, [0.0, -20000.0, -3000.0]
        act.ctrlrange, act.forcerange = [-SLIDE_RANGE, SLIDE_RANGE], [-1000.0, 1000.0]
        act.ctrllimited = act.forcelimited = 1

    site = spec.site('attachment_site')
    body = spec.body(info['flange']).add_body(name='tool', pos=list(site.pos), quat=list(site.quat))
    body.gravcomp = 1.0
    grey, dark = [0.55, 0.57, 0.6, 1], [0.2, 0.2, 0.22, 1]
    shares = {'bracket': 0.25 if tool.shape == 'L' else 0.0, 'blade': 0.6}
    start, direction, x_tip = tool.axis()
    blade_quat = np.zeros(4)
    mujoco.mju_mat2Quat(blade_quat, np.column_stack([direction, np.cross(x_tip, direction), x_tip]).flatten())
    if tool.shape == 'L':
        body.add_geom(name='tool_bracket', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.02, tool.drop / 2],
                      pos=[0, 0, tool.drop / 2], rgba=grey, mass=tool.mass * shares['bracket'])
    body.add_geom(name='tool_blade', type=mujoco.mjtGeom.mjGEOM_BOX,  # length along `direction`, height along `x_tip`
                  size=[tool.blade / 2, tool.blade_thickness / 2, tool.blade_height / 2], quat=list(blade_quat),
                  pos=list(start + direction * tool.blade / 2), rgba=dark, mass=tool.mass * shares['blade'],
                  friction=[0.4, 0.005, 0.0001])
    body.add_site(name='ft_sensor', pos=[0, 0, 0])
    tip_pos, tip_rot = tool.tip()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, tip_rot.flatten())
    cup_origin = start + direction * tool.blade
    cup = body.add_body(name='cup', pos=list(cup_origin))
    cup.gravcomp = 1.0
    cup_quat = np.zeros(4)  # cylinder axis (its z) along the blade direction
    mujoco.mju_mat2Quat(cup_quat, np.column_stack([x_tip, np.cross(direction, x_tip), direction]).flatten())
    cup.add_geom(name='cup', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[tool.cup_radius, tool.cup_length / 2, 0],
                 pos=list(direction * tool.cup_length / 2), quat=list(cup_quat), rgba=[0.15, 0.35, 0.8, 1],
                 mass=tool.mass * (1 - sum(shares.values())), margin=0.004, gap=0.004, friction=[1.0, 0.005, 0.0001])
    cup.add_site(name='tip', pos=list(tip_pos - cup_origin), quat=list(quat))
    spec.add_sensor(name='ft_force', type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='ft_sensor')
    spec.add_sensor(name='ft_torque', type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname='ft_sensor')
    suction = spec.add_actuator(name='suction', target='cup')
    suction.trntype = mujoco.mjtTrn.mjTRN_BODY
    suction.gaintype, suction.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
    suction.gainprm[0] = float(suction_force)  # N at full vacuum: the cup lets go above it
    suction.ctrlrange, suction.ctrllimited = [0.0, 1.0], 1
    if info['base_housing']:
        spec.body('base').add_geom(name='base_housing', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.076, 0.045, 0],
                                   pos=[0, 0, 0.045], rgba=[0, 0, 0, 0], group=3)  # the Menagerie UR5e base has no collision shape
    tool_geoms = {'tool_bracket', 'tool_blade', 'cup'}
    for g in spec.geoms:
        if g.contype or g.conaffinity or g.name == 'base_housing':
            # arm links touch the scene; the tool also touches the arm (a blade through the robot's own base is a collision)
            g.contype, g.conaffinity = ROBOT, ROBOT if g.name in tool_geoms else 0
    if base[2] > 0.02:
        # solid for the arm, the tool and the objects (it is part of the robot's body tree: robot bits plus the scene's)
        spec.geom('pedestal').contype, spec.geom('pedestal').conaffinity = SCENE | ROBOT, SCENE | ROBOT
    return spec
