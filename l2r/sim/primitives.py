"""The three motion primitives: push with the blade, pick with the end effector (suction cup or parallel jaws), place the
held object."""
import logging

import numpy as np

from l2r.sim.arm import HOME_JOINTS

log = logging.getLogger(__name__)

FRONT_MARGIN = 0.1  # m in front of the cabinet where every primitive starts / ends (place: 0.15)
PUSH_HEIGHT = 0.05  # m above the cabinet floor: low, to avoid toppling objects (paper: 5 cm)
PICK_HEIGHT = 0.04
GRASP_HEIGHT = PICK_HEIGHT  # jaw centre above bounds[2][0] (the cabinet floor): pads at mid-height of the objects
PUSH_QUATERNION = (np.sqrt(0.5), 0, -np.sqrt(0.5), 0)  # blade pointing into the cabinet
PLACE_QUATERNION = (0.7088025212287903, 0.0003121407644357532, -0.7054068446159363, 0.0003104391216766089)
WALL_MARGIN = 0.008  # m: a push stops this far from the side walls


class Manipulator:
    """Push / pick / place built from the arm's Cartesian moves and the suction cup."""

    def __init__(self, arm, tool, scene, workspace, params):
        """tool: the end effector, a `SuctionCup` or a `ParallelJawGripper`."""
        self.arm, self.suction, self.scene, self.ws, self.params = arm, tool, scene, workspace, params

    # ------------------------------------------------------------------ helpers
    def _front(self, y, z, margin=FRONT_MARGIN):
        """Staging point in front of the cabinet (outside it, at the given y / z)."""
        return (self.ws.bounds[0][1] + margin, y, z)

    def _retreat_from(self, pos, front, quat, speed):
        """Leave towards the staging point on a straight line from `pos`, then go home."""
        self.arm.move_straight(pos, front, quat, speed)
        self.arm.go_home()

    # ------------------------------------------------------------------ push
    def push(self, start_xy, angle=0.0, speed=0.01, push_length=0.08):
        """Push along y from `start_xy` by `push_length` m (direction sin(angle) = -1 / +1). True if the motion
        completed and the scene is still valid."""
        arm, b = self.arm, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        start = np.array([start_xy[0], start_xy[1], b[2][0] + PUSH_HEIGHT])
        end = [start[0], start[1] + push_length * np.sin(angle), start[2]]
        end[1] = max(min(end[1], b[1, 1] - WALL_MARGIN), b[1, 0] + WALL_MARGIN)
        quat = PUSH_QUATERNION
        front = self._front(start[1], start[2])

        success = arm.move_to_pose((front, quat), speed)
        if success:
            success = arm.move_straight(front, start, quat, speed, detect_force=True)  # approach
        else:
            pos = np.asarray(arm.tool_pose()[0])
            self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
            return success
        if success:
            success = arm.move_straight(start, end, quat, 0.1 * speed, detect_force=True)  # the push itself
        else:
            log.debug('push: approach stopped, going back home')
            pos = np.asarray(arm.tool_pose()[0])
            self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
            return success
        pos = np.asarray(arm.tool_pose()[0])
        arm.move_to_pose((start, quat), speed)
        # retreat: after a completed push from the start point, after a force stop from where it stopped
        self._retreat_from(start if success else pos, self._front(start[1], start[2]), quat, speed)
        return success & self.scene.is_valid()

    # ------------------------------------------------------------------ pick
    def pick(self, xy, speed=0.01, yaw=0.0):
        """Pick the object at `xy` with the end effector: `_pick_with_cup`, or with the jaws (rotated by `yaw` about the
        vertical) `_pick_with_jaws`."""
        if getattr(self.suction, 'is_gripper', False):
            return self._pick_with_jaws(xy, speed, yaw)
        return self._pick_with_cup(xy, speed)

    def _pick_with_cup(self, xy, speed=0.01):
        """Touch the object at `xy` from the front with the cup, switch the vacuum on and lift it out of the
        cabinet. True if an object is held (and, with params.suction_verify, still held after the lift)."""
        arm, suction, b = self.arm, self.suction, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        target = np.array([xy[0], xy[1], b[2][0] + PICK_HEIGHT])
        pos, quat = arm.tool_pose()
        suction.release()
        front = self._front(target[1], target[2])

        success = arm.move_to_pose((front, quat), speed)
        if not success:
            arm.move_straight(pos, front, quat, speed)
            arm.go_home()
            return success
        success = arm.move_straight(front, target, quat, speed=0.5 * speed, detect_contact=True, detect_force=True)
        pos = np.asarray(arm.tool_pose()[0])
        if np.linalg.norm(np.float32(target) - np.float32(pos)) < 0.05:  # touching the object (or at the target)
            suction.activate()
            success = True
        if success:
            suction.activate()
            lift = self._front(pos[1], pos[2], margin=FRONT_MARGIN + 0.2)
            arm.move_straight(pos, lift, quat, speed)
            arm.go_home()
            grasped = suction.activated
            if grasped and self.params.suction_verify and not suction.holds():
                suction.release()  # the object slipped off during the lift
                grasped = False
            success &= grasped
        else:
            pos = np.asarray(arm.tool_pose()[0])
            arm.move_straight(pos, front, quat, speed)
            arm.go_home()
        arm.go_home()
        return success & self.scene.is_valid()

    def _pick_with_jaws(self, xy, speed=0.01, yaw=0.0):
        """Approach from the front with open jaws rotated by `yaw` about the vertical (fingertips at `xy`, low above the
        floor), close, pull the object out of the cabinet. True if an object is held after the lift (and still between the
        pads with params.suction_verify)."""
        arm, gripper, b = self.arm, self.suction, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        pos, home_quat = arm.tool_pose()
        quat = self.arm.p.multiplyTransforms([0, 0, 0], self.arm.p.getQuaternionFromEuler([0, 0, yaw]), [0, 0, 0], home_quat)[1]
        height = b[2][0] + GRASP_HEIGHT
        front = self._front(xy[1], height)
        target = (xy[0] + gripper.grip_depth, xy[1], height)  # tool origin: the middle of the pads goes to the object centre
        success = arm.move_to_pose((front, quat), speed)
        if success:
            success = arm.move_straight(front, target, quat, 0.5 * speed, detect_force=True)
        grasped = bool(success) and gripper.close()
        pos = np.asarray(arm.tool_pose()[0])
        arm.move_straight(pos, self._front(pos[1], pos[2], margin=FRONT_MARGIN + 0.2), quat, speed)
        arm.go_home()
        if grasped and self.params.suction_verify and not gripper.holds():
            gripper.release()
            grasped = False
        arm.go_home()
        return grasped & self.scene.is_valid(ignore=(gripper.held,) if grasped else ())

    # ------------------------------------------------------------------ place
    def place(self, xy, speed=0.01):
        """Carry the held object to `xy` (the cup position that puts the object there), release and retreat."""
        arm, suction, b = self.arm, self.suction, self.ws.bounds
        quat = PLACE_QUATERNION
        offset_x, offset_y = suction.place_offsets(quat)
        position = np.array([xy[0] - offset_x, xy[1] + offset_y, b[2][0] + PUSH_HEIGHT])
        front = self._front(position[1], position[2], margin=0.15)

        reached = arm.move_to_pose((front, quat), speed)
        if not reached:
            reached = arm.move_to_pose((front, quat), speed)  # retry once
            reached = reached or not self.params.strict_place  # legacy: carry on even if the retry failed
        placed = reached and arm.move_straight(front, position, quat, speed, detect_force=True)
        suction.release()  # drop the object where the tool is (also after a failed approach)
        pos = np.asarray(arm.tool_pose()[0])
        retreated = arm.move_straight(pos, front, quat, speed)
        arm.go_home()
        arm.go_home()
        valid = self.scene.is_valid()
        suction.release()
        return bool(placed and retreated and valid)
