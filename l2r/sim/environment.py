"""PyBullet simulation of the confined-space rearrangement task: UR5e + pushing blade / suction cup, a cabinet,
YCB objects and an RGB-D camera. Components live in the sibling modules; this class wires them to one client."""
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from l2r.config import SimParams
from l2r.sim.arm import Arm
from l2r.sim.camera import Camera
from l2r.sim.primitives import Manipulator
from l2r.sim.scene import Scene
from l2r.sim.suction import SuctionCup
from l2r.sim.vision import build_observation, relabel_colormap
from l2r.sim.workspace import Workspace

TIME_STEP = 1 / 480  # s; resetSimulation restores PyBullet's default of 1/240, so it is set on every reset
WAIT_STATIC_MAX_STEPS = 480  # simulation-step budget of wait_static (1 s of simulated time)


class Environment:
    """One simulated cabinet. All randomness comes from `numpy.random` / `random` (seed them for reproducibility)."""

    def __init__(self, gui=False, time_step=TIME_STEP, num_obj=(9,), params=None, client=None):
        """
        Args:
            gui: show the PyBullet viewer (needs a display); headless runs use the CPU renderer.
            num_obj: the number of objects of an episode is sampled from this sequence.
            params: `SimParams` (physics, controller and camera options).
            client: an existing PyBullet client (dependency injection for tests); default: a new one.
        """
        self.params = params or SimParams()
        self.num_obj = list(num_obj)
        self.time_step = time_step
        self.ws = Workspace.from_params(self.params)
        self.p = client or bullet_client.BulletClient(connection_mode=pybullet.GUI if gui else pybullet.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setTimeStep(time_step)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        if gui:
            self.p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=90, cameraPitch=-39.40,
                                              cameraTargetPosition=[0.0, 0, 0.2])
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)
        self.scene = Scene(self.p, self.params, self.ws)
        self.camera = Camera(self.p, self.params, self.ws, gui)
        self.arm = Arm(self.p, self.params, self.ws, time_step)
        self.suction = SuctionCup(self.p, self.params, self.arm, self.scene)
        self.arm.holding = lambda: self.suction.activated
        self.arm.in_contact = self.suction.in_contact
        self.manipulator = Manipulator(self.arm, self.suction, self.scene, self.ws, self.params)

    # ------------------------------------------------------------------ episode
    def reset(self):
        """Rebuild the world with a new random arrangement. Returns False if the scene is not valid (retry)."""
        p, params = self.p, self.params
        self.camera.randomize()
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        engine = {}  # resetSimulation also restores these solver settings
        if params.solver_iterations:
            engine['numSolverIterations'] = int(params.solver_iterations)
        if params.deterministic:
            engine['deterministicOverlappingPairs'] = 1
        if engine:
            p.setPhysicsEngineParameter(**engine)
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # faster scene loading
        self.arm.load()
        if not self.arm.go_home():
            print('Simulation is wrong!')
        p.loadURDF('plane.urdf', basePosition=(0, 0, 0), useFixedBase=True)
        self.scene.build_cabinet()
        self.scene.spawn_objects(self.num_obj)
        self.suction.reset()
        self.wait_static()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return self.scene.is_valid()

    def wait_static(self):
        """Step the simulation until the objects stop moving (or the budget is used up)."""
        self.p.stepSimulation()
        for _ in range(WAIT_STATIC_MAX_STEPS):  # simulation steps, not wall-clock: independent of the CPU speed
            if self.scene.is_static():
                return True
            self.p.stepSimulation()
        return False

    # ------------------------------------------------------------------ observation and task state
    def observe(self):
        """Maps of the current scene: 'colormap' / 'depthmap' (padded) and 'idmap' / 'classmap' / 'freespace'."""
        return build_observation(self.camera.render(), self.scene.names, self.ws, self.scene.target_pixels())

    def is_valid(self):
        return self.scene.is_valid()

    def fraction_sorted(self):
        return self.scene.fraction_sorted()

    def relabel_candidates(self):
        """See `Scene.relabel_candidates`; ids and classes are in `scene.relabel_ids` / `scene.relabel_classes`."""
        return self.scene.relabel_candidates()

    def relabel_colormap(self, colormap, id_map):
        """Recolour the objects of the current relabelling in a colormap (see `vision.relabel_colormap`)."""
        return relabel_colormap(colormap, id_map, self.scene.relabel_ids, self.scene.relabel_classes)

    def target_pixels(self):
        return self.scene.target_pixels()

    # ------------------------------------------------------------------ primitives
    def push(self, xy, angle=0.0, push_length=0.08, speed=0.01):
        return self.manipulator.push(xy, angle, speed, push_length)

    def pick(self, xy, speed=0.01):
        return self.manipulator.pick(xy, speed)

    def place(self, xy, speed=0.01):
        return self.manipulator.place(xy, speed)

    # ------------------------------------------------------------------ diagnostics
    @property
    def force_stops(self):
        return self.arm.force_stops

    @property
    def force_events(self):
        return self.arm.force_events
