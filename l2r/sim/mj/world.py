"""World construction of the native MuJoCo backend: floor, cabinet (the walls of `Scene.build_cabinet`), lights, the camera,
and movable objects, around the robot of `robot.py`."""
import mujoco
import numpy as np

from l2r.sim.mj.robot import ROBOT, SCENE, robot_spec

WALL_THICKNESS, CEILING_THICKNESS = 0.2, 0.05
WALL_RGBA = [186 / 255, 176 / 255, 172 / 255, 1]


def cabinet_walls(ws):
    """{name: (half extents, centre)} of the static walls, as `Scene.build_cabinet`."""
    b = ws.bounds
    sx, sy = ws.scene_size
    w, tw = WALL_THICKNESS, CEILING_THICKNESS
    height, bottom = b[2][1] - b[2][0], b[2][0]
    side_half = height / 2 + bottom / 2
    return {'floor': ([sx / 2, sy / 2, bottom / 2], [0, 0, bottom / 2]),
            'back': ([w / 2, sy / 2 + w, side_half], [-sx / 2 - w / 2 - 0.001, 0, side_half]),
            'left': ([sx / 2, w / 2, side_half], [0, -w / 2 - sy / 2 - 0.001, height / 2.0 + bottom / 2]),
            'right': ([sx / 2, w / 2, side_half], [0, w / 2 + sy / 2 + 0.001, height / 2.0 + bottom / 2]),
            'ceiling': ([sx / 2 + w / 2, sy / 2 + w, tw / 2], [-w / 2, 0, height + bottom + tw / 2])}


def world_spec(ws, tool=None, suction_force=30.0, cabinet_friction=0.5, timestep=1 / 480, iterations=100, base=None,
               slide=True, arm='ur5e'):
    spec = robot_spec(tool, suction_force, slide=slide, arm=arm, **({'base': base} if base is not None else {}))
    spec.option.timestep = timestep
    spec.option.iterations = iterations
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    spec.visual.map.znear, spec.visual.map.zfar = 0.002, 5.0
    spec.stat.extent = 2.0
    world = spec.worldbody
    light = world.add_light(pos=[0.3, 0, 1.6], dir=[0, 0, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    world.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.9, 0.9, 0.9, 1],
                   contype=SCENE, conaffinity=SCENE)  # the robot never reaches the ground
    for name, (half, centre) in cabinet_walls(ws).items():
        world.add_geom(name='wall_' + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=centre, rgba=WALL_RGBA,
                       contype=SCENE, conaffinity=SCENE | ROBOT, friction=[cabinet_friction, 0.005, 0.0001])
    world.add_camera(name='camera', pos=[0.52, 0, ws.bounds[2][0] + 0.38], fovy=56.0)
    return spec


def add_box(spec, name, position, half, mass, friction=0.5, yaw=0.0, rgba=(1, 1, 1, 1)):
    body = spec.worldbody.add_body(name=name, pos=list(position), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), mass=mass, rgba=list(rgba),
                  contype=SCENE, conaffinity=SCENE | ROBOT, friction=[friction, 0.005, 0.0001], condim=4)
    return body


def collision_check(model):
    """`free(q)` for the planner: True if the arm configuration q (the controller's joints) has no contact other than with
    movable objects (pushing an object is the point of a push; the cabinet, the pedestal and the robot itself are obstacles)."""
    from l2r.sim.mj.control import Kinematics
    probe, joints = mujoco.MjData(model), Kinematics(model)
    movable = {b for b in range(model.nbody) if model.body_jntnum[b] and
               model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}

    def free(q, data=None):
        if data is not None:  # the objects where they are now
            probe.qpos[:] = data.qpos
        probe.qpos[joints.qadr] = q
        mujoco.mj_forward(model, probe)
        for c in probe.contact[:probe.ncon]:
            if c.dist < 0 and model.geom_bodyid[c.geom1] not in movable and model.geom_bodyid[c.geom2] not in movable:
                return False
        return True
    return free
