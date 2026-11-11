"""A MuJoCo world behind the subset of the PyBullet client API that the backend-neutral simulator code uses.

Stage 1 of the MuJoCo backend (see `l2r/sim/mj/__init__.py`): the existing `Arm`, `SuctionCup`, `Scene`, `Manipulator` and
`ForceMonitor` run unchanged on it, so the two physics engines can be compared on the same primitives and scenes.

Two phases per episode:

* **building** (after `resetSimulation`): `loadURDF`, `createCollisionShape`, `createVisualShape`, `createMultiBody`,
  `changeDynamics` are *recorded*; `stepSimulation` calls are counted. Ids are handed out in PyBullet's order (the arm first,
  then the plane, the walls, the objects), so `Scene.build_cabinet` / `Scene.spawn_objects` run unchanged and draw their random
  numbers in the same order as on PyBullet: a seed gives the same scene on both engines.
* **running** (after `compile`): one `MjModel` holds everything; the queries (`getJointStates`, `getLinkState`,
  `getContactPoints`, `getClosestPoints`, `calculateInverseKinematics`, constraints, `saveState` ...) are answered from it.

Body ids are the PyBullet-style ids of the building phase, links of the arm are PyBullet link indices (link L = the child of
URDF joint L); poses are PyBullet's (base / link inertial frame for positions of bodies, x y z w quaternions).

Modelling choices that differ from PyBullet (documented, to be compared):

* joint motors: MuJoCo position servos, gains from the joint-space inertia at the home pose (bandwidth `servo_bandwidth`
  rad/s, critically damped), force limited to the URDF effort; the arm is gravity compensated (PyBullet's position motors
  are stiff velocity constraints that hold the arm up);
* suction: a weld equality per object, declared inactive and switched on by `createConstraint` (no force limit: MuJoCo's
  welds are soft constraints without a maximum force; `changeConstraint(maxForce=...)` is ignored);
* joint reaction wrench (`getJointState(...)[2]`): the force the child body receives from its parent (`cfrc_int`, world
  frame; the torque part is about a different reference point than PyBullet's);
* bodies created after `compile` (the paddle, the parallel-jaw gripper) are not supported yet.
"""
import logging

import mujoco
import numpy as np

from l2r.sim.mj import assets
from l2r.sim.mj import transforms as T

log = logging.getLogger(__name__)

ARM_ID = 0  # the arm is the first body loaded (as on PyBullet)


class _Record:
    def __init__(self, kind, mass=0.0, position=(0, 0, 0), orientation=(0, 0, 0, 1), geoms=(), model=None):
        self.kind, self.mass = kind, float(mass)
        self.position, self.orientation = tuple(position), tuple(orientation)
        self.geoms = list(geoms)  # [(PyBullet geom type, size dict, rgba, collides)]
        self.model = model  # YCB model description (assets.ycb_model)
        self.friction = None


class MujocoClient:
    # PyBullet constants used by the shared code
    GEOM_SPHERE, GEOM_BOX, GEOM_CYLINDER, GEOM_MESH, GEOM_PLANE, GEOM_CAPSULE = 2, 3, 4, 5, 6, 7
    JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_FIXED, JOINT_POINT2POINT = 0, 1, 4, 5
    VELOCITY_CONTROL, TORQUE_CONTROL, POSITION_CONTROL = 0, 1, 2
    COV_ENABLE_GUI, COV_ENABLE_SHADOWS, COV_ENABLE_RENDERING = 1, 2, 7
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, ER_TINY_RENDERER, ER_BULLET_HARDWARE_OPENGL = 1, 65536, 131072
    URDF_ENABLE_CACHED_GRAPHICS_SHAPES = 1024

    def __init__(self, servo_bandwidth=80.0, default_friction=0.5):
        self.servo_bandwidth, self.default_friction = float(servo_bandwidth), float(default_friction)
        self.time_step, self.gravity, self.iterations = 1 / 240, (0.0, 0.0, -9.81), None
        self.model = self.data = None
        self._renderers = {}
        self.resetSimulation()

    # ------------------------------------------------------------------ the PyBullet math helpers
    getQuaternionFromEuler = staticmethod(T.quat_from_euler)
    getEulerFromQuaternion = staticmethod(T.euler_from_quat)
    getMatrixFromQuaternion = staticmethod(T.matrix_from_quat)
    multiplyTransforms = staticmethod(T.multiply_transforms)
    invertTransform = staticmethod(T.invert_transform)

    # ------------------------------------------------------------------ options (no-ops where MuJoCo has no equivalent)
    def setAdditionalSearchPath(self, *args, **kwargs):
        pass

    def configureDebugVisualizer(self, *args, **kwargs):
        pass

    def resetDebugVisualizerCamera(self, *args, **kwargs):
        pass

    def enableJointForceTorqueSensor(self, *args, **kwargs):
        pass  # the reaction wrench of every joint is available (cfrc_int)

    def setTimeStep(self, time_step):
        self.time_step = float(time_step)
        if self.model is not None:
            self.model.opt.timestep = self.time_step

    def setGravity(self, x, y, z):
        self.gravity = (float(x), float(y), float(z))
        if self.model is not None:
            self.model.opt.gravity[:] = self.gravity

    def setPhysicsEngineParameter(self, numSolverIterations=None, **kwargs):
        if numSolverIterations:
            self.iterations = int(numSolverIterations)

    # ------------------------------------------------------------------ building phase
    def resetSimulation(self):
        self.records, self.shapes = [], []
        self.model = self.data = None
        self.pending_steps = 0
        self.joint_table = None
        self._renderers = {}
        self._states, self._synced = {}, None
        self._warned = set()

    def _new(self, record):
        self.records.append(record)
        return len(self.records) - 1

    def _building(self, what):
        if self.model is not None:
            raise NotImplementedError('MuJoCo backend: %s after the world was compiled (the paddle and the parallel-jaw '
                                      'gripper create bodies during an episode) is not supported yet' % what)

    def loadURDF(self, path, basePosition=(0, 0, 0), baseOrientation=(0, 0, 0, 1), useFixedBase=False, **kwargs):
        self._building('loadURDF')
        if path.endswith('ur5e.urdf'):
            if any(r.kind == 'arm' for r in self.records):
                raise NotImplementedError('MuJoCo backend: one arm per world')
            self.joint_table = assets.urdf_joint_table()
            return self._new(_Record('arm', position=basePosition, orientation=baseOrientation))
        if path.endswith('plane.urdf'):
            return self._new(_Record('plane', position=basePosition))
        if 'ycb_objects' in path:
            kind = path.replace('\\', '/').split('/')[-2][len('Ycb'):]
            model = assets.ycb_model(kind)
            record = _Record('ycb', mass=model['mass'], position=basePosition, orientation=baseOrientation, model=model)
            record.friction = model['friction']
            return self._new(record)
        raise NotImplementedError('MuJoCo backend: loadURDF(%r)' % path)

    def createCollisionShape(self, shapeType, halfExtents=None, radius=None, height=None, **kwargs):
        self._building('createCollisionShape')
        self.shapes.append((int(shapeType), {'halfExtents': halfExtents, 'radius': radius, 'height': height}, None, True))
        return len(self.shapes) - 1

    def createVisualShape(self, shapeType, halfExtents=None, radius=None, length=None, rgbaColor=None, **kwargs):
        self._building('createVisualShape')
        rgba = tuple(rgbaColor) if rgbaColor is not None else (1.0, 1.0, 1.0, 1.0)
        self.shapes.append((int(shapeType), {'halfExtents': halfExtents, 'radius': radius, 'height': length}, rgba, False))
        return len(self.shapes) - 1

    def createMultiBody(self, baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=(0, 0, 0),
                        baseOrientation=(0, 0, 0, 1), **kwargs):
        self._building('createMultiBody')
        geoms = []
        if baseCollisionShapeIndex >= 0:
            kind, size, _, _ = self.shapes[baseCollisionShapeIndex]
            rgba = self.shapes[baseVisualShapeIndex][2] if baseVisualShapeIndex >= 0 else (1.0, 1.0, 1.0, 1.0)
            geoms.append((kind, size, rgba, True))
        elif baseVisualShapeIndex >= 0:
            kind, size, rgba, _ = self.shapes[baseVisualShapeIndex]
            geoms.append((kind, size, rgba, False))
        kind = 'dynamic' if baseMass > 0 else 'static'
        return self._new(_Record(kind, baseMass, basePosition, baseOrientation, geoms))

    def changeDynamics(self, bodyUniqueId, linkIndex=-1, lateralFriction=None, mass=None, **kwargs):
        if self.model is None:
            record = self.records[bodyUniqueId]
            if lateralFriction is not None:
                record.friction = float(lateralFriction)
            if mass is not None:
                record.mass = float(mass)
            return
        body = self.body_of[bodyUniqueId]
        if lateralFriction is not None:
            for g in np.nonzero(self.model.geom_bodyid == body)[0]:
                self.model.geom_friction[g, 0] = float(lateralFriction)
        if mass is not None:
            self.model.body_mass[body] = float(mass)

    def getDynamicsInfo(self, bodyUniqueId, linkIndex=-1):
        if self.model is None:
            record = self.records[bodyUniqueId]
            return (record.mass, record.friction if record.friction is not None else self.default_friction)
        body = self.body_of[bodyUniqueId]
        geoms = np.nonzero(self.model.geom_bodyid == body)[0]
        return (float(self.model.body_mass[body]), float(self.model.geom_friction[geoms[0], 0]) if len(geoms) else 0.0)

    # ------------------------------------------------------------------ compile
    def _add_geom(self, body, kind, size, rgba, collides, friction):
        g = body.add_geom()
        if kind == self.GEOM_BOX:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, list(size['halfExtents'])
        elif kind == self.GEOM_CYLINDER:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [size['radius'], size['height'] / 2, 0]
        elif kind == self.GEOM_SPHERE:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_SPHERE, [size['radius'], 0, 0]
        else:
            raise NotImplementedError('MuJoCo backend: geom type %d' % kind)
        g.rgba = list(rgba)
        g.contype, g.conaffinity = (1, 3) if collides else (0, 0)
        g.friction = [friction, 0.005, 0.0001]
        return g

    def compile(self, home_joints=None, base_spec=None, settle=None, weld_to=None):
        """Build the MuJoCo model of everything recorded since `resetSimulation`, put the arm at `home_joints` and run the
        simulation steps requested while building (the objects' settling). `base_spec`: an MjSpec the recorded world is added
        to (the native robot of `robot.py`; then no URDF arm is recorded). `settle(model, data)` runs before those steps.
        `weld_to`: the body of `base_spec` that gets one inactive weld per movable object (the native suction cup)."""
        arm = next((r for r in self.records if r.kind == 'arm'), None)
        if base_spec is not None:
            spec = base_spec
        else:
            spec = mujoco.MjSpec.from_string(assets.arm_urdf_text(arm.position)) if arm else mujoco.MjSpec()
        spec.option.timestep = self.time_step
        spec.option.gravity = list(self.gravity)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        if self.iterations:
            spec.option.iterations = self.iterations
        spec.visual.map.znear, spec.visual.map.zfar = 0.002, 5.0  # x the model extent (set below)
        spec.stat.extent = 2.0
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
        if arm:
            for body in spec.bodies[1:]:
                body.gravcomp = 1.0
                for g in body.geoms:
                    if g.contype or g.conaffinity:
                        g.contype, g.conaffinity = 2, 1  # touches the scene, not itself
                        g.friction = [1.0, 0.005, 0.0001]
            for joint in spec.joints:
                act = spec.add_actuator(name='servo_' + joint.name, target=joint.name)
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        world = spec.worldbody
        light = world.add_light()
        light.pos, light.dir = [0.3, 0, 1.5], [0, 0, -1]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        camera = world.add_camera(name='l2r_camera')
        camera.fovy = 45.0
        welds = []
        for uid, record in enumerate(self.records):
            name = 'l2r_body_%d' % uid
            if record.kind == 'arm':
                continue
            if record.kind == 'plane':
                g = world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], pos=list(record.position))
                g.rgba, g.contype, g.conaffinity = [0.9, 0.9, 0.9, 1], 1, 3
                continue
            body = world.add_body(name=name, pos=list(record.position), quat=list(T.to_mj(record.orientation)))
            friction = record.friction if record.friction is not None else self.default_friction
            if record.kind == 'static':
                for kind, size, rgba, collides in record.geoms:
                    self._add_geom(body, kind, size, rgba, collides, friction)
                continue
            body.add_freejoint()
            if record.kind == 'dynamic':
                for kind, size, rgba, collides in record.geoms:
                    self._add_geom(body, kind, size, rgba, collides, friction).mass = record.mass
            else:  # YCB: explicit inertia, convex hulls for contact, the textured mesh for the camera
                m = record.model
                body.explicitinertial = True
                body.mass, body.ipos = record.mass, list(m['ipos'])
                body.iquat, body.inertia = list(T.to_mj(m['iquat'])), list(m['inertia'])
                for k, path in enumerate(m['hulls']):
                    mesh = spec.add_mesh(name='%s_hull%d' % (name, k), file=path, scale=list(m['scale']))
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name)
                    g.pos, g.quat = list(m['collision_origin'][0]), list(T.to_mj(m['collision_origin'][1]))
                    g.contype, g.conaffinity, g.friction = 1, 3, [friction, 0.005, 0.0001]
                    g.rgba, g.group = [1, 1, 1, 0], 3  # contact only
                    g.mass = 0
                path, scale = m['visual']
                mesh = spec.add_mesh(name='%s_visual' % name, file=path, scale=list(scale))
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name)
                g.pos, g.quat = list(m['visual_origin'][0]), list(T.to_mj(m['visual_origin'][1]))
                g.contype = g.conaffinity = 0
                g.rgba, g.mass = [1, 1, 1, 1], 0
            if arm or weld_to:
                welds.append(uid)
        tool = self.joint_table[-1][7] if arm else weld_to
        for uid in welds:  # one suction weld per movable object, off until createConstraint
            eq = spec.add_equality(name='l2r_weld_%d' % uid, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=tool, name2='l2r_body_%d' % uid)
            eq.active = False
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        if arm:
            self._tune_servos(home_joints)
        if settle is not None:
            settle(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        steps, self.pending_steps = self.pending_steps, 0
        for _ in range(steps):
            self.stepSimulation()

    def _index(self):
        m = self.model
        self.body_of = {}  # PyBullet-style id -> MuJoCo body (the root body of the arm: base_link)
        self.owner = np.full(m.nbody, -1, int)  # MuJoCo body -> PyBullet-style id
        self.link_of_body = np.full(m.nbody, -1, int)
        for uid, record in enumerate(self.records):
            if record.kind in ('arm', 'plane'):
                continue
            b = m.body('l2r_body_%d' % uid).id
            self.body_of[uid], self.owner[b] = b, uid
        self.link_body = {}
        if self.joint_table:
            for index, name, *_rest in self.joint_table:
                child = self.joint_table[index][7]
                b = m.body(child).id
                self.link_body[index] = b
                self.owner[b], self.link_of_body[b] = ARM_ID, index
            self.body_of[ARM_ID] = self.link_body[0]
            self.joint_qpos, self.joint_dof, self.joint_act = {}, {}, {}
            for index, name, kind, *_rest in self.joint_table:
                if kind in (assets.JOINT_REVOLUTE, assets.JOINT_PRISMATIC):
                    j = m.joint(name).id
                    self.joint_qpos[index], self.joint_dof[index] = m.jnt_qposadr[j], m.jnt_dofadr[j]
                    self.joint_act[index] = m.actuator('servo_' + name).id
            self.arm_dofs = np.array([self.joint_dof[i] for i in sorted(self.joint_dof)])
        planes = [g for g in range(m.ngeom) if m.geom(g).name.startswith('l2r_body_')]
        self.geom_owner = self.owner[m.geom_bodyid].copy()
        for g in planes:
            self.geom_owner[g] = int(m.geom(g).name[len('l2r_body_'):])
        self.weld_of = {}
        for e in range(m.neq):
            name = m.equality(e).name
            if name.startswith('l2r_weld_'):
                self.weld_of[int(name[len('l2r_weld_'):])] = e
        self._ik_data = mujoco.MjData(m)

    def _tune_servos(self, home_joints):
        m, d = self.model, self.data
        if home_joints is not None:
            for k, index in enumerate(sorted(self.joint_qpos)):
                d.qpos[self.joint_qpos[index]] = home_joints[k]
                d.ctrl[self.joint_act[index]] = home_joints[k]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        try:
            mujoco.mj_fullM(m, d, M)  # MuJoCo >= 4
        except TypeError:
            mujoco.mj_fullM(m, M, d.qM)
        w = self.servo_bandwidth
        for index, a in self.joint_act.items():
            inertia = M[self.joint_dof[index], self.joint_dof[index]]
            kp, kv = inertia * w * w, 2.0 * inertia * w
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, :3] = (0.0, -kp, -kv)
            effort = self.joint_table[index][5]
            if self.joint_table[index][2] == assets.JOINT_PRISMATIC:
                effort = 1000.0  # the URDF's 1 N is a placeholder (see arm_control.joint_limits)
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-effort, effort)
            j = m.joint(self.joint_table[index][1]).id  # the URDF importer also limits the joint itself to the effort
            m.jnt_actfrclimited[j] = 1
            m.jnt_actfrcrange[j] = (-effort, effort)

    # ------------------------------------------------------------------ stepping and state
    def stepSimulation(self):
        if self.model is None:
            self.pending_steps += 1
            return
        mujoco.mj_step(self.model, self.data)
        self._synced = None

    def _sync(self):
        """Kinematics (and the constraint forces) of the current state; the stored ones are from before the last step."""
        if self._synced != self.data.time:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)
            self._synced = self.data.time
            self._rne = False

    def saveState(self):
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        key = max(self._states, default=-1) + 1
        self._states[key] = (state, self.model.eq_data.copy())
        return key

    def restoreState(self, stateId):
        state, eq_data = self._states[stateId]
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.model.eq_data[:] = eq_data
        mujoco.mj_forward(self.model, self.data)
        self._synced = None

    def removeState(self, stateId):
        self._states.pop(stateId, None)

    # ------------------------------------------------------------------ bodies
    def _pose(self, b):
        self._sync()
        return tuple(float(v) for v in self.data.xipos[b]), T.quat_from_matrix(self.data.ximat[b])

    def getBasePositionAndOrientation(self, bodyUniqueId):
        return self._pose(self.body_of[bodyUniqueId])

    def getBaseVelocity(self, bodyUniqueId):
        b = self.body_of[bodyUniqueId]
        j = self.model.body_jntadr[b]
        if self.model.body_jntnum[b] == 0 or self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        v = self.data.qvel[self.model.jnt_dofadr[j]:self.model.jnt_dofadr[j] + 6]
        return tuple(float(x) for x in v[:3]), tuple(float(x) for x in v[3:])

    def getAABB(self, bodyUniqueId, linkIndex=-1):
        self._sync()
        m, d = self.model, self.data
        b = self.link_body[linkIndex] if bodyUniqueId == ARM_ID and linkIndex >= 0 else self.body_of[bodyUniqueId]
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for g in np.nonzero(m.geom_bodyid == b)[0]:
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            centre, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            world_centre = d.geom_xpos[g] + R @ centre
            extent = np.abs(R) @ half
            low, high = np.minimum(low, world_centre - extent), np.maximum(high, world_centre + extent)
        return tuple(low), tuple(high)

    # ------------------------------------------------------------------ the arm
    def getNumJoints(self, bodyUniqueId):
        return len(self.joint_table) if bodyUniqueId == ARM_ID and self.joint_table else 0

    def getJointInfo(self, bodyUniqueId, jointIndex):
        index, name, kind, lower, upper, effort, velocity, child, parent = self.joint_table[jointIndex]
        q = self.joint_qpos.get(index, -1) if self.model is not None else -1
        u = self.joint_dof.get(index, -1) if self.model is not None else -1
        return (index, name.encode(), kind, q, u, 0, 0.0, 0.0, lower, upper, effort, velocity, child.encode(),
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), parent)

    def _reaction(self, index):
        if not self._rne:
            mujoco.mj_rnePostConstraint(self.model, self.data)
            self._rne = True
        f = self.data.cfrc_int[self.link_body[index]]
        return tuple(float(v) for v in f[3:]) + tuple(float(v) for v in f[:3])

    def getJointState(self, bodyUniqueId, jointIndex):
        self._sync()
        if jointIndex in self.joint_qpos:
            q, qd = self.data.qpos[self.joint_qpos[jointIndex]], self.data.qvel[self.joint_dof[jointIndex]]
            torque = self.data.actuator_force[self.joint_act[jointIndex]]
        else:
            q = qd = torque = 0.0
        return float(q), float(qd), self._reaction(jointIndex), float(torque)

    def getJointStates(self, bodyUniqueId, jointIndices):
        return [self.getJointState(bodyUniqueId, j) for j in jointIndices]

    def setJointMotorControlArray(self, bodyIndex=None, jointIndices=(), controlMode=None, targetPositions=None,
                                  positionGains=None, forces=None, bodyUniqueId=None, **kwargs):
        if controlMode != self.POSITION_CONTROL:
            raise NotImplementedError('MuJoCo backend: control mode %s' % controlMode)
        if forces is not None and 'forces' not in self._warned:
            self._warned.add('forces')
            log.info('MuJoCo backend: motor forces come from the URDF effort limits (the forces argument is ignored)')
        for j, target in zip(jointIndices, targetPositions):
            self.data.ctrl[self.joint_act[j]] = target

    def getLinkState(self, bodyUniqueId, linkIndex, computeLinkVelocity=0, computeForwardKinematics=0):
        self._sync()
        b = self.link_body[linkIndex]
        d, m = self.data, self.model
        com = (tuple(float(v) for v in d.xipos[b]), T.quat_from_matrix(d.ximat[b]))
        frame = (tuple(float(v) for v in d.xpos[b]), T.from_mj(d.xquat[b]))
        return com[0], com[1], tuple(float(v) for v in m.body_ipos[b]), T.from_mj(m.body_iquat[b]), frame[0], frame[1]

    def calculateInverseKinematics(self, bodyUniqueId, endEffectorLinkIndex, targetPosition, targetOrientation=None,
                                   lowerLimits=None, upperLimits=None, jointRanges=None, restPoses=None,
                                   maxNumIterations=None, residualThreshold=1e-4, **kwargs):
        """Damped least squares from the current configuration (position and orientation of the link frame), with a weak
        pull towards `restPoses` in the null space and the joint limits enforced; returns the movable joints' positions.
        PyBullet's defaults: 20 iterations, residual 1e-4 (`Arm.solve_ik` repeats until the tool is within IK_TOLERANCE)."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        b = self.link_body[endEffectorLinkIndex]
        movable = sorted(self.joint_qpos)
        qadr = np.array([self.joint_qpos[i] for i in movable])
        low = np.array(lowerLimits if lowerLimits is not None else [-np.inf] * len(movable), float)
        high = np.array(upperLimits if upperLimits is not None else [np.inf] * len(movable), float)
        rest = np.array(restPoses, float) if restPoses is not None else None
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target = np.asarray(targetPosition, float)
        for _ in range(int(maxNumIterations or 20)):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            error = target - d.xpos[b]
            if targetOrientation is not None:
                current = T.from_mj(d.xquat[b])
                delta = T.quat_multiply(targetOrientation, T.quat_conjugate(current))
                if delta[3] < 0:
                    delta = tuple(-v for v in delta)
                rot = 2.0 * np.asarray(delta[:3])
                error = np.concatenate([error, rot])
            if np.linalg.norm(error) < residualThreshold:
                break
            mujoco.mj_jac(m, d, jacp, jacr, d.xpos[b], b)
            J = jacp[:, self.arm_dofs] if targetOrientation is None else np.vstack([jacp, jacr])[:, self.arm_dofs]
            JJt = J @ J.T + 1e-4 * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, error)
            if rest is not None:
                null = np.eye(len(movable)) - J.T @ np.linalg.solve(JJt, J)
                dq += null @ (0.05 * (rest - d.qpos[qadr]))
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, low, high)
        return tuple(float(v) for v in d.qpos[qadr])

    # ------------------------------------------------------------------ contacts
    def _side(self, geom):
        b = self.model.geom_bodyid[geom]
        return int(self.geom_owner[geom]), int(self.link_of_body[b])

    @staticmethod
    def _matches(side, body, link):
        return (body is None or side[0] == body) and (link is None or link == -2 or side[1] == link)

    def getContactPoints(self, bodyA=None, bodyB=None, linkIndexA=-2, linkIndexB=-2):
        m, d = self.model, self.data
        out, force = [], np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.exclude:
                continue
            sides = [self._side(c.geom1), self._side(c.geom2)]
            normal = np.asarray(c.frame[:3])  # from geom1 to geom2
            for a, sign in ((0, -1.0), (1, 1.0)):
                A, B = sides[a], sides[1 - a]
                if self._matches(A, bodyA, linkIndexA) and self._matches(B, bodyB, linkIndexB):
                    mujoco.mj_contactForce(m, d, i, force)
                    on_b = sign * normal  # PyBullet: the normal on B, pointing towards A
                    # PyBullet: position on A = position on B + normal * distance (distance < 0: penetration)
                    pos = np.asarray(c.pos)
                    out.append((0, A[0], B[0], A[1], B[1], tuple(pos + on_b * c.dist / 2), tuple(pos - on_b * c.dist / 2),
                                tuple(on_b), float(c.dist), float(force[0])))
                    break
        return out

    def getClosestPoints(self, bodyA, bodyB, distance, linkIndexA=-2, linkIndexB=-2):
        self._sync()
        m, d = self.model, self.data
        fromto, out = np.zeros(6), []
        geoms_a = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                   and self._matches(self._side(g), bodyA, linkIndexA)]
        geoms_b = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                   and self._matches(self._side(g), bodyB, linkIndexB)]
        for ga in geoms_a:
            for gb in geoms_b:
                dist = mujoco.mj_geomDistance(m, d, ga, gb, distance, fromto)
                if dist <= distance:
                    a, b = fromto[:3].copy(), fromto[3:].copy()
                    n = a - b
                    n = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n
                    A, B = self._side(ga), self._side(gb)
                    out.append((0, A[0], B[0], A[1], B[1], tuple(a), tuple(b), tuple(n), float(dist), 0.0))
        return sorted(out, key=lambda point: point[8])

    def setCollisionFilterPair(self, *args, **kwargs):
        self._building('setCollisionFilterPair')

    # ------------------------------------------------------------------ constraints (suction welds)
    def createConstraint(self, parentBodyUniqueId, parentLinkIndex, childBodyUniqueId, childLinkIndex, jointType,
                         jointAxis, parentFramePosition, childFramePosition, parentFrameOrientation=(0, 0, 0, 1),
                         childFrameOrientation=(0, 0, 0, 1)):
        """A fixed joint between the tool and an object: switches its weld on. The child's frame (in its inertial frame) is
        held at the parent frame (in the parent link's inertial frame), as on PyBullet."""
        if (parentBodyUniqueId != ARM_ID or parentLinkIndex != self.joint_table[-1][0] or jointType != self.JOINT_FIXED
                or childBodyUniqueId not in self.weld_of):
            raise NotImplementedError('MuJoCo backend: only the suction weld (tool link -> object) is supported')
        m = self.model
        e = self.weld_of[childBodyUniqueId]
        as_quat = lambda q: T.quat_from_euler(q) if len(q) == 3 else tuple(q)  # PyBullet takes (0, 0, 0) as no rotation
        parentFrameOrientation, childFrameOrientation = as_quat(parentFrameOrientation), as_quat(childFrameOrientation)
        parent, child = self.link_body[parentLinkIndex], self.body_of[childBodyUniqueId]
        inertial_p = (tuple(m.body_ipos[parent]), T.from_mj(m.body_iquat[parent]))
        inertial_c = (tuple(m.body_ipos[child]), T.from_mj(m.body_iquat[child]))
        pose = T.multiply_transforms(*inertial_p, parentFramePosition, parentFrameOrientation)
        pose = T.multiply_transforms(*pose, *T.invert_transform(childFramePosition, childFrameOrientation))
        pose = T.multiply_transforms(*pose, *T.invert_transform(*inertial_c))  # child body frame in the parent body frame
        m.eq_data[e, :3] = 0.0  # anchor: the child body origin
        m.eq_data[e, 3:6] = pose[0]
        m.eq_data[e, 6:10] = T.to_mj(pose[1])
        m.eq_data[e, 10] = 1.0  # torque scale
        self.data.eq_active[e] = 1
        return int(e)

    def changeConstraint(self, constraintUniqueId, **kwargs):
        if 'maxForce' in kwargs and 'maxForce' not in self._warned:
            self._warned.add('maxForce')
            log.info('MuJoCo backend: constraint force limits are not modelled (welds are soft constraints)')

    def removeConstraint(self, constraintUniqueId):
        self.data.eq_active[constraintUniqueId] = 0

    def getConstraintInfo(self, constraintUniqueId):
        if not self.data.eq_active[constraintUniqueId]:
            raise KeyError('constraint %d is not active' % constraintUniqueId)
        child = int(self.owner[self.model.eq_obj2id[constraintUniqueId]])
        return (ARM_ID, self.joint_table[-1][0], child, -1, self.JOINT_FIXED)

    # ------------------------------------------------------------------ camera
    def render_camera(self, position, rotation, fovy, height, width):
        """(color HxWx3 uint8, depth HxW metres along the optical axis, segmentation HxW PyBullet-style body id or -1) from a
        camera at `position` with the camera-to-world `rotation` (columns: x right, y up, z backwards, MuJoCo's convention)."""
        m, d = self.model, self.data
        cam = m.camera('l2r_camera').id
        m.cam_pos[cam] = position
        m.cam_quat[cam] = T.to_mj(T.quat_from_matrix(rotation))
        m.cam_fovy[cam] = fovy
        self._synced = None
        self._sync()
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(m, height, width)
        renderer = self._renderers[key]
        renderer.update_scene(d, camera=cam)
        color = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera=cam)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera=cam)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        ids = np.full(seg.shape[:2], -1, np.int64)
        is_geom = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[:, :, 0] >= 0)
        ids[is_geom] = self.geom_owner[seg[:, :, 0][is_geom]]
        return color, depth, ids

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
