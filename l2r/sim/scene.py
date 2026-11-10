"""The cabinet, the objects in it, and the sorting task (target regions, goal and invalid-state checks)."""
import random
from copy import deepcopy

import numpy as np

from l2r.sim.constants import (BLOCK_HALF_EXTENTS, CLASSES, LABELS, MAX_OBJECTS_PER_CLASS, OBJECT_CATALOGUE,
                               TARGET_COLORS, TARGET_ORDER, target_region_choices)

WALL_THICKNESS = 0.2
CEILING_THICKNESS = 0.05
WALL_COLOR = [186 / 255, 176 / 255, 172 / 255, 1]
MIN_OBJECT_DISTANCE = 0.1  # m between object centres at spawn (shrinks if the cabinet is too full)


class Scene:
    """Static cabinet + movable objects + target regions. Everything is created in `reset`."""

    def __init__(self, p, params, workspace):
        self.p, self.params, self.ws = p, params, workspace
        self.object_ids = []  # PyBullet ids of the movable objects
        self.names = {}  # object id -> 'class YcbName'
        self.regions = {}  # class -> [y_min, y_max] (m) of its target region
        self.relabel_ids, self.relabel_classes = [], []  # filled by `relabel_candidates`
        # until the first reset: the first three candidate intervals, unshuffled
        self.regions = {key: list(interval) for key, interval in zip(TARGET_ORDER, target_region_choices(self.ws.bounds))}

    # ------------------------------------------------------------------ queries
    @staticmethod
    def class_of(name):
        """'blue CrackerBox' -> 'blue'."""
        return name.split(' ')[0]

    def class_name(self, object_id):
        return self.class_of(self.names[object_id])

    def poses(self):
        """{object id: (position, orientation quaternion)}."""
        return {i: self.p.getBasePositionAndOrientation(i) for i in self.object_ids}

    def is_static(self):
        """True if no object moves any more."""
        velocities = [self.p.getBaseVelocity(i) for i in self.object_ids]
        return all(np.linalg.norm(v[0]) < 1e-2 and np.linalg.norm(v[1]) < 1e-2 for v in velocities)

    def in_target(self, y, name):
        low, high = self.regions[self.class_of(name)]
        return low <= y <= high

    def target_pixels(self):
        """Rows of the target regions in the heightmap: (lower rows, upper rows, colours), in TARGET_ORDER."""
        low, high, colors = [], [], []
        for key in self.regions:
            high.append(self.ws.y_to_row(self.regions[key][1]))
            low.append(self.ws.y_to_row(self.regions[key][0]))
            colors.append(TARGET_COLORS[key])
        return low, high, colors

    # ------------------------------------------------------------------ task state
    def is_valid(self, ignore=()):
        """False if an object fell off the cabinet floor or toppled over (the invalid state of the paper). `ignore`: object
        ids left out of the check (an object held in the air by a gripper may hang tilted; it is checked once released)."""
        for _ in range(5):
            self.p.stepSimulation()
        poses = {k: v for k, v in self.poses().items() if k not in ignore}
        if not poses:
            return True
        z = np.array([pos[2] for pos, _ in poses.values()])
        roll = [abs(self.p.getEulerFromQuaternion(q)[0]) for _, q in poses.values()]
        pitch = [abs(self.p.getEulerFromQuaternion(q)[1]) for _, q in poses.values()]
        upright = max(roll) < 1.2 and max(pitch) < 1.2
        return bool(np.all(z > self.ws.bounds[2][0] - 0.1) and upright)

    def invalid_objects(self):
        """Ids of the objects that make the scene invalid now (fell off the floor or toppled), without stepping the simulation."""
        out = []
        for object_id, (pos, quat) in self.poses().items():
            roll, pitch, _ = self.p.getEulerFromQuaternion(quat)
            if pos[2] <= self.ws.bounds[2][0] - 0.1 or abs(roll) >= 1.2 or abs(pitch) >= 1.2:
                out.append(int(object_id))
        return out

    def progress(self):
        """Task progress without stepping the simulation: (fraction of the objects inside the region of their class,
        mean distance in m of the objects to their region, 0 when inside)."""
        poses = self.poses()
        inside, distance = 0, []
        for object_id, (pos, _) in poses.items():
            low, high = self.regions[self.class_name(object_id)]
            inside += low <= pos[1] <= high
            distance.append(max(low - pos[1], pos[1] - high, 0.0))
        return inside / len(poses), float(np.mean(distance))

    def positions(self):
        """(N,3) object positions, in the order of `object_ids`."""
        return np.array([self.p.getBasePositionAndOrientation(i)[0] for i in self.object_ids])

    def fraction_sorted(self):
        """Fraction of the objects inside the target region of their own class."""
        self.p.stepSimulation()
        poses = self.poses()
        return sum(self.in_target(pos[1], self.names[i]) for i, (pos, _) in poses.items()) / len(poses)

    def relabel_candidates(self):
        """Hindsight relabelling support: returns the fraction of objects lying inside *some* target region and
        stores their ids and the class of that region (the class they would have if the scene were sorted)."""
        self.relabel_ids, self.relabel_classes = [], []
        poses = self.poses()
        for object_id, (pos, _) in poses.items():
            for key in self.regions:
                low, high = self.regions[key]
                if low <= pos[1] <= high:
                    self.relabel_ids.append(object_id)
                    self.relabel_classes.append(LABELS[key])
                    break
        return len(self.relabel_ids) / len(poses)

    # ------------------------------------------------------------------ construction
    def randomize_regions(self):
        """Assign the three target intervals to the classes in a random order."""
        shuffled = random.sample(target_region_choices(self.ws.bounds), 3)
        self.regions = {key: [shuffled[k][0], shuffled[k][1]] for k, key in enumerate(TARGET_ORDER)}

    def build_cabinet(self):
        """Static walls: floor, back, left, right and ceiling."""
        b = self.ws.bounds
        sx, sy = self.ws.scene_size
        w, tw = WALL_THICKNESS, CEILING_THICKNESS
        height, bottom = b[2][1] - b[2][0], b[2][0]
        side_half = height / 2 + bottom / 2
        walls = {  # name -> (half extents, centre)
            'floor': ([sx / 2, sy / 2, bottom / 2], [0, 0, bottom / 2]),
            'back': ([w / 2, sy / 2 + w, side_half], [-sx / 2 - w / 2 - 0.001, 0, side_half]),
            'left': ([sx / 2, w / 2, side_half], [0, -w / 2 - sy / 2 - 0.001, height / 2.0 + bottom / 2]),
            'right': ([sx / 2, w / 2, side_half], [0, w / 2 + sy / 2 + 0.001, height / 2.0 + bottom / 2]),
            'ceiling': ([sx / 2 + w / 2, sy / 2 + w, tw / 2], [-w / 2, 0, height + bottom + tw / 2]),
        }
        self.wall_ids = {}
        for name, (half, centre) in walls.items():
            shape = self.p.createCollisionShape(shapeType=self.p.GEOM_BOX, halfExtents=half)
            visual = self.p.createVisualShape(shapeType=self.p.GEOM_BOX, halfExtents=half, rgbaColor=WALL_COLOR)
            self.wall_ids[name] = self.p.createMultiBody(0, baseCollisionShapeIndex=shape, baseVisualShapeIndex=visual,
                                                         basePosition=centre, baseOrientation=[0, 0, 0, 1])  # static
        self.p.changeDynamics(self.wall_ids['floor'], -1, lateralFriction=self.params.cabinet_friction)

    def _draw_object_names(self, count):
        """Names of the objects of an episode: three distinct YCB objects, one per class, drawn at random."""
        ycb = random.sample(list(self.params.object_types or OBJECT_CATALOGUE), 3)
        names = [f'{c} {o}' for c, o in zip(CLASSES, ycb)]
        counts, chosen = dict.fromkeys(CLASSES, 0), []
        while len(chosen) < count:
            name = np.random.choice(names)
            cls = self.class_of(name)
            if counts[cls] < MAX_OBJECTS_PER_CLASS:
                counts[cls] += 1
                chosen.append(name)
        return chosen

    def spawn_objects(self, num_obj):
        """Create `num_obj` (a list to sample the count from) objects at random, well separated positions."""
        p, b = self.p, self.ws.bounds
        count = min(int(np.random.choice(num_obj)), MAX_OBJECTS_PER_CLASS * len(CLASSES))
        names = self._draw_object_names(count)
        self.randomize_regions()
        self.names, self.object_ids = {}, []
        centres = np.zeros((0, 2))
        for name in names:
            min_dist, attempts = MIN_OBJECT_DISTANCE, 0
            while True:  # rejection sampling; the spacing shrinks if the cabinet gets too full
                attempts += 1
                if attempts % 100 == 0:
                    min_dist *= 0.8
                x = np.random.uniform(b[0, 0] + 0.05, b[0, 1] - 0.1)
                y = np.random.uniform(b[1, 0] + 0.02, b[1, 1] - 0.02)
                xyz = np.float32([x, y, b[2][0] + 0.0]).reshape(1, 3)
                xy = np.float32([x, y]).reshape(1, 2)
                if len(centres) == 0 or np.min(np.linalg.norm(centres - xy, axis=1)).squeeze() > min_dist:
                    centres = np.concatenate((centres, xy), axis=0)
                    break
            cls, kind = name.split(' ')
            dims = deepcopy(BLOCK_HALF_EXTENTS[cls])
            dims[0] += np.random.uniform(-0.005, 0.01)
            dims[1] += np.random.uniform(-0.005, 0.01)
            dims[2] += np.random.uniform(-0.01, 0.01)
            dims[2] = max(dims[2], 0.03)
            position = xyz.squeeze()
            object_id = self._create_object(kind, position, dims)
            if self.params.randomize:  # domain randomisation of friction and mass
                mass = p.getDynamicsInfo(object_id, -1)[0]
                p.changeDynamics(object_id, -1, lateralFriction=np.random.uniform(*self.params.friction_range),
                                 mass=mass * np.random.uniform(*self.params.mass_scale_range))
            self.names[object_id] = name
            for _ in range(20):
                p.stepSimulation()  # let it settle
            self.object_ids.append(object_id)

    def _create_object(self, kind, position, dims):
        """YCB model by name, or a 'block' / 'cylinder' primitive of random size."""
        p = self.p
        if kind == 'block':
            position[2] += dims[2] + 0.002  # rest on the floor instead of starting half inside it
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=dims)
            return p.createMultiBody(self.params.block_mass, shape, visual, basePosition=position)
        if kind == 'cylinder':
            radius = 0.03 + np.random.uniform(-0.005, 0.005)
            position[2] += dims[2] + 0.002
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=2 * dims[2])
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=2 * dims[2])
            return p.createMultiBody(self.params.cylinder_mass, shape, visual, basePosition=position)
        orientation = p.getQuaternionFromEuler([0, 0, np.random.uniform(-3.14, 3.14)])
        return p.loadURDF(f'assets/ycb_objects/Ycb{kind}/model.urdf', basePosition=position, baseOrientation=orientation)
