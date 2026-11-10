"""`Environment` on MuJoCo (`--backend mujoco`): the PyBullet environment's components on a `MujocoClient`.

Everything except the world construction and the camera is inherited: observation, primitives, validity, task state."""
import logging

from l2r.config import SimParams
from l2r.sim.arm import HOME_JOINTS
from l2r.sim.environment import TIME_STEP, Environment
from l2r.sim.mj.camera import MujocoCamera
from l2r.sim.mj.client import MujocoClient

log = logging.getLogger(__name__)


class MujocoEnvironment(Environment):
    def __init__(self, gui=False, time_step=TIME_STEP, num_obj=(9,), params=None, client=None, perception=None):
        params = params or SimParams()
        if gui:
            raise NotImplementedError('MuJoCo backend: no GUI yet (use `python -m mujoco.viewer` on a saved model)')
        if params.end_effector == 'gripper':
            raise NotImplementedError('MuJoCo backend: the parallel-jaw gripper is not ported yet (suction only)')
        if getattr(params, 'pusher', 'tip') == 'paddle':
            raise NotImplementedError('MuJoCo backend: the paddle pusher is not ported yet (--pusher tip)')
        client = client or MujocoClient(servo_bandwidth=getattr(params, 'mj_servo_bandwidth', 80.0))
        super().__init__(False, time_step, num_obj, params, client=client, perception=perception)
        self.camera = MujocoCamera(self.p, self.params, self.ws)

    def reset(self):
        """Same steps and random draws as `Environment.reset`; the world is compiled once everything is recorded, with the
        arm put at its home configuration directly (PyBullet drives it there before the objects exist)."""
        p, params = self.p, self.params
        self.camera.randomize()
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        if params.solver_iterations:
            p.setPhysicsEngineParameter(numSolverIterations=int(params.solver_iterations))
        p.setGravity(0, 0, -9.81)
        self.arm.load()
        self.tool.attach()
        p.loadURDF('plane.urdf', basePosition=(0, 0, 0), useFixedBase=True)
        self.scene.build_cabinet()
        self.scene.spawn_objects(self.num_obj)
        p.compile(home_joints=HOME_JOINTS)
        self.wait_static()
        return self.scene.is_valid()
