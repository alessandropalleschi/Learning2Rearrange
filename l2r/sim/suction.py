"""Suction cup model: a rigid constraint to the object touched by the tool, limited in force."""
import numpy as np


class SuctionCup:
    """Vacuum tool simulated as a fixed constraint (`SimParams.suction_max_force`, `suction_erp`) between the tool
    link and an object it touches. The tool pose it uses is the link frame of the arm's `ee_link`."""

    MAX_LATERAL_OFFSET = 0.017  # m: contact must be within this of the cup axis (y) to seal
    CLOSE_DISTANCE = 0.015  # m: fallback search radius when there is no contact point

    def __init__(self, p, params, arm, scene):
        self.p, self.params, self.arm, self.scene = p, params, arm, scene
        self.reset()

    def reset(self):
        self.activated = False
        self.constraint = None
        self.offset_x = self.offset_y = 0.0  # where the grasped object sits relative to the cup (used by `place`)
        self.grip_body = self.grip_distance = None

    # ------------------------------------------------------------------ queries
    def in_contact(self):
        """True if the tool (or, while holding, the held object) touches something."""
        p, body, link = self.p, self.arm.body, self.arm.ee_link
        if self.activated and self.constraint is not None:
            try:
                info = p.getConstraintInfo(self.constraint)
                body, link = info[2], info[3]
            except Exception:  # constraint vanished
                self.constraint = None
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.arm.body]  # ignore contacts with the robot itself
        return bool(points)

    def holds(self):
        """True if the constraint is alive and the object is still where it was when it was grasped."""
        if not self.activated or self.constraint is None or self.grip_body is None:
            return False
        tool_pos, _ = self.arm.tool_frame()
        obj_pos, _ = self.p.getBasePositionAndOrientation(self.grip_body)
        distance = np.linalg.norm(np.asarray(obj_pos) - np.asarray(tool_pos))
        return abs(distance - self.grip_distance) < self.params.suction_hold_tolerance

    # ------------------------------------------------------------------ actions
    def release(self):
        """Switch the vacuum off (drop the constraint). Only does something while activated."""
        if not self.activated:
            return
        self.activated = False
        if self.constraint is not None:
            try:
                self.p.removeConstraint(self.constraint)
                self.constraint = None
            except Exception:  # already removed
                pass
            self.grip_body = self.grip_distance = None

    def activate(self):
        """Switch the vacuum on: attach to the best contacting object (or, without contact, the closest object
        within CLOSE_DISTANCE). Does nothing if already activated or nothing is close; check `activated`."""
        if self.activated:
            return
        contacts = self.p.getContactPoints(bodyA=self.arm.body, linkIndexA=self.arm.ee_link)
        if contacts:
            self._attach_to_contact(contacts)
        else:
            self._attach_to_closest()

    # ------------------------------------------------------------------ internals
    def _object_in_tool_frame(self, object_id):
        """Pose of an object in the tool frame: (position, orientation)."""
        tool_pos, tool_orn = self.arm.tool_frame()
        inverse = self.p.invertTransform(tool_pos, tool_orn)
        obj_pos, obj_orn = self.p.getBasePositionAndOrientation(object_id)
        return self.p.multiplyTransforms(inverse[0], inverse[1], obj_pos, obj_orn)

    def _attach_to_contact(self, contacts):
        """Constraint to the nearest contacting object whose contact is on the cup axis, in front of the tool and
        facing it (surface normal aligned with the cup axis within params.suction_min_alignment)."""
        tool_pos, _ = self.arm.tool_frame()
        best, best_distance = None, 1000.0
        for point in contacts:
            object_id, link, contact_pos, normal = point[2], point[4], point[5], point[7]
            if object_id not in self.scene.object_ids:
                continue
            distance = np.linalg.norm(np.asarray(self._object_in_tool_frame(object_id)[0]))
            if (distance < best_distance and np.abs(contact_pos[1] - tool_pos[1]) < self.MAX_LATERAL_OFFSET
                    and contact_pos[0] < tool_pos[0] and normal[0] > self.params.suction_min_alignment):
                best, best_distance = (object_id, link, contact_pos), distance
        if best is None:
            return
        object_id, link, contact_pos = best
        obj_to_tool = self._object_in_tool_frame(object_id)
        offset = contact_pos[0] - tool_pos[0] if tool_pos[0] > contact_pos[0] else 0  # object-side contact
        anchor = (obj_to_tool[0][0] + offset, obj_to_tool[0][1], obj_to_tool[0][2])
        self._create_constraint(object_id, link, anchor, obj_to_tool)
        self.offset_y, self.offset_x = obj_to_tool[0][1], -anchor[0]

    def _attach_to_closest(self):
        """No contact point: constraint to the closest object that is within CLOSE_DISTANCE, laterally on the cup axis
        and behind the contact point."""
        tool_pos, _ = self.arm.tool_frame()
        best, best_distance = None, 1000
        for object_id in self.scene.object_ids:
            close = self.p.getClosestPoints(bodyA=self.arm.body, linkIndexA=self.arm.ee_link,
                                            distance=self.CLOSE_DISTANCE, bodyB=object_id)
            if not close:
                continue
            first = close[0]
            obj_pos = self.p.getBasePositionAndOrientation(object_id)[0]
            distance = np.linalg.norm(np.asarray(self._object_in_tool_frame(object_id)[0]))
            if np.abs(first[5][1] - tool_pos[1]) < 0.01 and distance < best_distance and obj_pos[0] > first[5][0]:
                best, best_distance = close, distance
        if best is None:
            return
        object_id, link = best[-1][2], best[-1][4]
        obj_to_tool = self._object_in_tool_frame(object_id)
        self._create_constraint(object_id, link, obj_to_tool[0], obj_to_tool)
        self.offset_x, self.offset_y = -obj_to_tool[0][0], obj_to_tool[0][1]

    def _create_constraint(self, object_id, link, anchor, obj_to_tool):
        self.constraint = self.p.createConstraint(
            parentBodyUniqueId=self.arm.body, parentLinkIndex=self.arm.ee_link, childBodyUniqueId=object_id,
            childLinkIndex=link, jointType=self.p.JOINT_FIXED, jointAxis=(0, 0, 0), parentFramePosition=anchor,
            parentFrameOrientation=obj_to_tool[1], childFramePosition=(0, 0, 0), childFrameOrientation=(0, 0, 0))
        self.p.changeConstraint(self.constraint, gearRatio=-1, erp=self.params.suction_erp,
                                maxForce=self.params.suction_max_force)
        self.activated = True
        self.grip_body = object_id
        self.grip_distance = float(np.linalg.norm(np.asarray(obj_to_tool[0])))
