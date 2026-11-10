"""The robot of the native MuJoCo backend: the MuJoCo Menagerie UR5e on the lateral base slide, with the L-shaped tool.

Tool (flange frame: z = flange normal): a bracket `drop` along the normal, then a horizontal blade of length `blade` along the
flange x axis, a suction cup at its end. With the flange pointing down, joint 6 turns the blade in the floor plane. The blade is
a thin vertical plate (pushes with its end or sideways with its face); the cup is its own body so that the suction (an adhesion
actuator) acts through the cup only. A force / torque sensor sits at the flange (the force between the tool and the wrist, as the
UR5e's built-in sensor measures it).

The geometry follows `scripts/flange_study.py` (L 0.10 + 0.75 m reaches 98% of the approach lines that pass the cabinet opening).
`scripts/fetch_menagerie.sh` downloads the arm model (BSD-3)."""
import os
from dataclasses import dataclass

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
UR5E = os.path.join(ROOT, 'assets', 'menagerie', 'universal_robots_ur5e')
BASE_POSITION = (0.6, 0.0, 0.0)  # as the PyBullet URDF
SLIDE_RANGE = 0.5  # m either side
ARM_JOINTS = ('slide', 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
              'wrist_3_joint')
ROBOT, SCENE = 2, 1  # collision bits: robot geoms contype ROBOT / conaffinity 0, scene geoms SCENE / SCENE | ROBOT:
# the robot touches the scene, not itself; scene geoms touch each other


@dataclass
class ToolGeometry:
    drop: float = 0.10  # m from the flange along its normal to the blade
    blade: float = 0.75  # m, horizontal
    blade_height: float = 0.04
    blade_thickness: float = 0.008
    cup_radius: float = 0.015
    cup_length: float = 0.02
    mass: float = 0.6  # kg, the whole tool

    def tip(self):
        """Tip (cup face) position and axes in the flange frame: z = pointing direction, x = 'down' with the flange down."""
        position = np.array([self.blade + self.cup_length, 0.0, self.drop])
        x_tip, z_tip = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        return position, np.column_stack([x_tip, np.cross(z_tip, x_tip), z_tip])


def _menagerie_spec():
    path = os.path.join(UR5E, 'ur5e.xml')
    if not os.path.exists(path):
        raise FileNotFoundError('the Menagerie UR5e is missing: run scripts/fetch_menagerie.sh')
    with open(path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', 'meshdir="%s"' % os.path.join(UR5E, 'assets'))
    xml = xml.replace('<body name="base" quat="0 0 0 -1" childclass="ur5e">',
                      '<body name="slide_carriage" pos="%g %g %g"><joint name="slide" type="slide" axis="0 1 0" '
                      'range="%g %g" damping="50"/><body name="base" quat="1 0 0 0" childclass="ur5e">'
                      % (*BASE_POSITION, -SLIDE_RANGE, SLIDE_RANGE), 1)
    xml = xml.replace('</worldbody>', '</body></worldbody>', 1)
    spec = mujoco.MjSpec.from_string(xml)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def robot_spec(tool=None, suction_force=30.0):
    """MjSpec with the arm, the slide actuator, gravity compensation, the tool, the flange sensor and the suction actuator."""
    tool = tool or ToolGeometry()
    spec = _menagerie_spec()
    for body in spec.bodies[1:]:
        body.gravcomp = 1.0  # the UR controller compensates gravity
    carriage = spec.body('slide_carriage')
    carriage.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.08, 0.12, 0.01], pos=[0, 0, -0.01], contype=0, conaffinity=0,
                      rgba=[0.3, 0.3, 0.3, 1], mass=10.0)
    act = spec.add_actuator(name='slide', target='slide')
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0], act.biasprm[:3] = 20000.0, [0.0, -20000.0, -3000.0]
    act.ctrlrange, act.forcerange = [-SLIDE_RANGE, SLIDE_RANGE], [-1000.0, 1000.0]
    act.ctrllimited = act.forcelimited = 1

    site = spec.site('attachment_site')
    body = spec.body('wrist_3_link').add_body(name='tool', pos=list(site.pos), quat=list(site.quat))
    body.gravcomp = 1.0
    grey, dark = [0.55, 0.57, 0.6, 1], [0.2, 0.2, 0.22, 1]
    shares = {'bracket': 0.25, 'blade': 0.6}
    body.add_geom(name='tool_bracket', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.02, tool.drop / 2],
                  pos=[0, 0, tool.drop / 2], rgba=grey, mass=tool.mass * shares['bracket'])
    body.add_geom(name='tool_blade', type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[tool.blade / 2, tool.blade_thickness / 2, tool.blade_height / 2],
                  pos=[tool.blade / 2, 0, tool.drop], rgba=dark, mass=tool.mass * shares['blade'], friction=[0.4, 0.005, 0.0001])
    body.add_site(name='ft_sensor', pos=[0, 0, 0])
    tip_pos, tip_rot = tool.tip()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, tip_rot.flatten())
    cup = body.add_body(name='cup', pos=[tool.blade, 0, tool.drop])
    cup.gravcomp = 1.0
    cup.add_geom(name='cup', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[tool.cup_radius, tool.cup_length / 2, 0],
                 pos=[tool.cup_length / 2, 0, 0], quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0], rgba=[0.15, 0.35, 0.8, 1],
                 mass=tool.mass * (1 - sum(shares.values())), margin=0.004, gap=0.004, friction=[1.0, 0.005, 0.0001])
    cup.add_site(name='tip', pos=list(tip_pos - [tool.blade, 0, tool.drop]), quat=list(quat))
    spec.add_sensor(name='ft_force', type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='ft_sensor')
    spec.add_sensor(name='ft_torque', type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname='ft_sensor')
    suction = spec.add_actuator(name='suction', target='cup')
    suction.trntype = mujoco.mjtTrn.mjTRN_BODY
    suction.gaintype, suction.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
    suction.gainprm[0] = float(suction_force)  # N at full vacuum: the cup lets go above it
    suction.ctrlrange, suction.ctrllimited = [0.0, 1.0], 1
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = ROBOT, 0
    return spec
