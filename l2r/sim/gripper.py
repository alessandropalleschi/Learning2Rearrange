"""Parallel-jaw gripper (wide-stroke, 20 cm) as an alternative end effector to the suction cup.

Two finger pads on prismatic joints close along the tool's lateral axis (world y): the objects stand on the cabinet floor, so a
vertical pinch is impossible and the approach is from the front, along the tool's z axis. The jaws are a programmatic multibody
attached to the tool link by a fixed constraint (collisions with the arm are disabled); the grasp is held by friction between
the force-limited pads and the object: no weld is involved.
"""
import numpy as np

OPENING = 0.20  # m between the pads when open (a wide-stroke gripper: the scene objects are up to ~17 cm across)
PAD_THICKNESS = 0.012
PAD_HEIGHT = 0.04  # extent along the tool x axis (vertical in the world)
FINGER_LENGTH = 0.06
PALM_HALF = (0.012, 0.115, 0.008)
REACH = PALM_HALF[2] + FINGER_LENGTH  # tool origin -> fingertips (m)
GRIP_DEPTH = PALM_HALF[2] + FINGER_LENGTH / 2  # tool origin -> centre of the pads: this point is put at the object centre
CLOSE_FORCE = 80.0  # N per finger
CLOSE_SPEED = 0.15  # m/s
MIN_WIDTH = 0.02  # m: closing (nearly) completely means nothing was grasped
SETTLE_STEPS = 240
OBJECT_SPINNING_FRICTION = 0.5


class ParallelJawGripper:
    """Same interface towards `Manipulator` as `SuctionCup`: `activated` (an object is held), `release`, `holds`, `in_contact`,
    `place_offsets`; plus `close` / `open` and the geometry (`REACH`)."""

    is_gripper = True
    reach = REACH
    grip_depth = GRIP_DEPTH

    def __init__(self, p, params, arm, scene):
        self.p, self.params, self.arm, self.scene = p, params, arm, scene
        self.body = None
        self.reset()

    def reset(self):
        self.activated = False
        self.held = None
        self.held_in_tool = None  # position of the held object in the tool frame
        self.width = OPENING

    # ------------------------------------------------------------------ construction
    def attach(self):
        """Create the jaws at the tool pose and fix them to the tool link. Call after the arm is loaded, every reset."""
        p = self.p
        self.reset()
        box = lambda half: p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        look = lambda half, rgba: p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        pad = (PAD_HEIGHT / 2, PAD_THICKNESS / 2, FINGER_LENGTH / 2)
        offset = OPENING / 2 + PAD_THICKNESS / 2
        z = PALM_HALF[2] + FINGER_LENGTH / 2
        pos, orn = self.arm.tool_frame()
        self.body = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=box(PALM_HALF), baseVisualShapeIndex=look(PALM_HALF, [0.2, 0.2, 0.2, 1]),
            basePosition=pos, baseOrientation=orn,
            linkMasses=[0.05, 0.05], linkCollisionShapeIndices=[box(pad)] * 2,
            linkVisualShapeIndices=[look(pad, [0.9, 0.5, 0.1, 1])] * 2,
            linkPositions=[(0, offset, z), (0, -offset, z)], linkOrientations=[(0, 0, 0, 1)] * 2,
            linkInertialFramePositions=[(0, 0, 0)] * 2, linkInertialFrameOrientations=[(0, 0, 0, 1)] * 2,
            linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_PRISMATIC] * 2, linkJointAxis=[(0, -1, 0), (0, 1, 0)])
        for link in (0, 1):
            p.changeDynamics(self.body, link, jointLowerLimit=0, jointUpperLimit=OPENING / 2, lateralFriction=2.0,
                             spinningFriction=0.05, contactStiffness=3e4, contactDamping=1e3)
        constraint = p.createConstraint(self.arm.body, self.arm.ee_link, self.body, -1, p.JOINT_FIXED, (0, 0, 0),
                                        (0, 0, 0), (0, 0, 0))
        p.changeConstraint(constraint, maxForce=500)
        for arm_link in range(-1, p.getNumJoints(self.arm.body)):  # the jaws never collide with the arm itself
            for link in (-1, 0, 1):
                p.setCollisionFilterPair(self.arm.body, self.body, arm_link, link, 0)
        self._command(0.0)

    # ------------------------------------------------------------------ jaws
    def _command(self, closing):
        """Target position of both pads: 0 = open, OPENING / 2 = fully closed."""
        for link in (0, 1):
            self.p.setJointMotorControl2(self.body, link, self.p.POSITION_CONTROL, targetPosition=closing,
                                         force=CLOSE_FORCE, maxVelocity=CLOSE_SPEED)

    def _pad_positions(self):
        return [self.p.getJointState(self.body, link)[0] for link in (0, 1)]

    def open(self):
        self._command(0.0)
        for _ in range(SETTLE_STEPS):
            self.p.stepSimulation()
            if max(self._pad_positions()) < 1e-3:
                break
        self.width = OPENING - sum(self._pad_positions())

    def release(self):
        """Open the jaws (drops the held object)."""
        self.open()
        self.activated, self.held, self.held_in_tool = False, None, None

    def close(self):
        """Close until the pads stop. A grasp is an object touched by both pads with the gap not fully closed."""
        p = self.p
        self._command(OPENING / 2)
        for _ in range(SETTLE_STEPS):
            p.stepSimulation()
            if max(abs(p.getJointState(self.body, link)[1]) for link in (0, 1)) < 1e-3:
                break
        self.width = OPENING - sum(self._pad_positions())
        touched = [set(c[2] for c in p.getContactPoints(bodyA=self.body, linkIndexA=link)) & set(self.scene.object_ids)
                   for link in (0, 1)]
        common = touched[0] & touched[1]
        if self.width > MIN_WIDTH and common:
            self.held = next(iter(common))
            # PyBullet combines the two bodies' coefficients multiplicatively: without torsional friction on the object the
            # pinch contact is a pivot and the held object swings about the pad axis
            p.changeDynamics(self.held, -1, spinningFriction=OBJECT_SPINNING_FRICTION)
            self.activated = True
            self.held_in_tool = self._object_in_tool(self.held)
        return self.activated

    # ------------------------------------------------------------------ queries
    def _tool_rotation(self, orn):
        return np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    def _object_in_tool(self, object_id):
        tool_pos, tool_orn = self.arm.tool_frame()
        object_pos, _ = self.p.getBasePositionAndOrientation(object_id)
        return self._tool_rotation(tool_orn).T @ (np.array(object_pos) - np.array(tool_pos))

    def in_contact(self):
        """True if the jaws (or the held object) touch something."""
        p = self.p
        bodies = [self.body] + ([self.held] if self.held is not None else [])
        return any(len([c for c in p.getContactPoints(bodyA=b) if c[2] not in (self.body, self.arm.body)]) for b in bodies)

    def holds(self):
        """True if the object is still between the pads, where it was when it was grasped."""
        if not self.activated or self.held is None:
            return False
        moved = np.linalg.norm(self._object_in_tool(self.held) - self.held_in_tool)
        return bool(moved < self.params.suction_hold_tolerance and OPENING - sum(self._pad_positions()) > MIN_WIDTH)

    def place_offsets(self, orientation):
        """(offset_x, offset_y) for `Manipulator.place` at the tool orientation `orientation`: the held object sits at
        `held_in_tool` in the tool frame, so at that orientation it is at R * held_in_tool from the tool."""
        if self.held_in_tool is None:
            return 0.0, 0.0
        world = self._tool_rotation(orientation) @ self.held_in_tool
        return float(world[0]), float(-world[1])
