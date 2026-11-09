from enum import unique
from json import tool
import time
import pybullet as p
import pybullet_data
import numpy as np
from utils import *
import os
import random
from scipy import ndimage
from copy import deepcopy
import cv2
from config import SimParams
from random import sample
PIXEL_SIZE = 0.005

BOUNDS = np.float32([[-0.25, 0.25], [-0.3, 0.3], [0.4, 0.75]])  # (X, Y, Z)
image_size = (480,640)
config_camera = {'position': (0.52, 0, BOUNDS[2][0]+0.38),
                  'rotation': (np.pi / 3, np.pi, -np.pi / 2),
                  'intrinsics': (450., 0, image_size[1]/2, 0, 450., image_size[0]/2, 0, 0, 1),
                  'image_size': image_size,
                  'zrange': (0.01, 10.0),
                  'noise': False}




PICK_TARGETS = {
  'blue block': None,
  'red block': None,
  'green block': None,
  'orange block': None,
  'yellow block': None,
  'purple block': None,
  'pink block': None,
  'cyan block': None,
  'brown block': None,
  'gray block': None,
}

COLORS = {
  'blue':   (78/255,  121/255, 167/255, 255/255),
  'red':    (255/255,  87/255,  89/255, 255/255),
  'green':  (89/255,  169/255,  79/255, 255/255),
  'orange': (242/255, 142/255,  43/255, 255/255),
  'yellow': (237/255, 201/255,  72/255, 255/255),
  'purple': (176/255, 122/255, 161/255, 255/255),
  'pink':   (255/255, 157/255, 167/255, 255/255),
  'cyan':   (118/255, 183/255, 178/255, 255/255),
  'brown':  (156/255, 117/255,  95/255, 255/255),
  'gray':   (186/255, 176/255, 172/255, 255/255),
}

LABELS = {
  'blue':   1,
  'red':    2,
  'green':  3,
  'orange': 4,
  'yellow': 5,
  'purple': 6,
  'pink':   7,
  'cyan':   8,
  'brown':  9,
  'gray':   10,
}


DIMENSIONS = {
  'blue':   [0.025, 0.025, 0.05],
  'red':    [0.03, 0.03, 0.07],
  'green':  [0.025, 0.025, 0.05],
  'orange': (242/255, 142/255,  43/255, 255/255),
  'yellow': [0.025, 0.025, 0.05],
  'purple': (176/255, 122/255, 161/255, 255/255),
  'pink':   (255/255, 157/255, 167/255, 255/255),
  'cyan':   (118/255, 183/255, 178/255, 255/255),
  'brown':  (156/255, 117/255,  95/255, 255/255),
  'gray':   (186/255, 176/255, 172/255, 255/255),
}





PLACE_TARGETS = {
  'blue': [ BOUNDS[1][0]+0.1, BOUNDS[1][0]+0.15 ],
  'yellow': [ (BOUNDS[1][1]-0.15), BOUNDS[1][1]-0.1 ],
  'green': [ -0.05, 0.05 ],
}
PLACE_TARGETS_ORG = PLACE_TARGETS
PLACE_TARGETS_COLORS = {
  'blue': [ 0, 0,139 ],
  'green': [ 0,110,51],
  'yellow': [ 246,190,0 ],
}




PLACE_TARGETS_LIST = [ [ BOUNDS[1][0]+0.01, BOUNDS[1][0]+0.16 ], [ (BOUNDS[1][1]-0.18), BOUNDS[1][1]-0.01 ], [ -0.075, 0.085 ]]
PLACE_TARGETS_LIST_1 = [ [ BOUNDS[1][0]+0.04, BOUNDS[1][0]+0.15 ], [ (BOUNDS[1][1]-0.15), BOUNDS[1][1]-0.04 ], [ -0.055, 0.055 ]]
PLACE_TARGETS_LIST_2 = [ [ BOUNDS[1][0]+0.01, BOUNDS[1][0]+0.14 ], [ (BOUNDS[1][1]-0.14), BOUNDS[1][1]-0.03 ], [ -0.065, 0.057 ]]
PLACE_TARGETS_LIST_3 = [ [ BOUNDS[1][0]+0.02, BOUNDS[1][0]+0.16 ], [ (BOUNDS[1][1]-0.16), BOUNDS[1][1]-0.02 ], [ -0.065, 0.037 ]]

PLACE_CHOICES = [PLACE_TARGETS_LIST]

COLORS_CLASS = {
  '1':   (78/255,  121/255, 167/255, 255/255),
  '2':    (255/255,  87/255,  89/255, 255/255),
  '4':  (89/255,  169/255,  79/255, 255/255),
  '3': (242/255, 142/255,  43/255, 255/255),
  '5': (237/255, 201/255,  72/255, 255/255),
  '6': (176/255, 122/255, 161/255, 255/255),
  '7':   (255/255, 157/255, 167/255, 255/255),
  '8':   (118/255, 183/255, 178/255, 255/255),
  '9':  (156/255, 117/255,  95/255, 255/255),
  '10':   (186/255, 176/255, 172/255, 255/255),
}

colors = ['blue GelatinBox','green MustardBottle', 'yellow PottedMeatCan']
colors = ['blue TomatoSoupCan','green MasterChefCan', 'yellow PottedMeatCan']
colors = ['blue ChipsCan','green MasterChefCan', 'yellow PottedMeatCan']
colors = ['blue CrackerBox','green MasterChefCan', 'yellow PottedMeatCan']
colors = ['blue','green', 'yellow']
objects  = ['GelatinBox', 'CrackerBox', 'PottedMeatCan', 'MasterChefCan', 'SugarBox', 'ChipsCan', 'MustardBottle']
objects  = ['PottedMeatCan', 'CrackerBox', 'ChipsCan', 'MustardBottle']

def configure_cabinet(params):
    """Resize the workspace (module-level BOUNDS is updated in place; call in every process that uses it)."""
    BOUNDS[0] = (-params.cabinet_depth / 2, params.cabinet_depth / 2)
    BOUNDS[2, 1] = BOUNDS[2, 0] + params.cabinet_height


class Environment:
    MOVE_JOINTS_MAX_STEPS = 1000   # simulation-step budget of move_joints
    WAIT_STATIC_MAX_STEPS = 480    # simulation-step budget of wait_static (1 s of simulated time)

    def __init__(self, gui=False, time_step=1 / 480, num_obj=[9], params=None):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """
        self.params = params or SimParams()
        configure_cabinet(self.params)
        self.camera_config = dict(config_camera)
        self.num_obj = num_obj
        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.cam_intrinsics = config_camera['intrinsics']
        self.cam_pos = config_camera['position']
        self.cam_rot = config_camera['rotation']
        self.bounds = BOUNDS
        self.home_joints = np.array([0,1, -1, 0.5, 0.5, 0.5, 1]) * np.pi
        self.ik_rest_joints = np.array([0,1, -1, 0.5, 0.5, 0.5, 1]) * np.pi
        self.final_scene_size = [self.bounds[0][1]-self.bounds[0][0]+0.005, self.bounds[1][1]-self.bounds[1][0]+0.015]      
        # Start PyBullet.
        self.contact_constraint = None
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        if gui:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-39.40,
                cameraTargetPosition=[0.0,0,0.2],

            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)        
        self.offset_y = 0
        self.offset_x = 0

        self.activated = False
        self.config = []
        # target regions are per-environment state (randomised in randomize()), not module globals
        self.place_targets = deepcopy(PLACE_TARGETS_ORG)
    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
        w = [np.linalg.norm(p.getBaseVelocity(i)[1]) for i in self.obj_ids["rigid"]]
        return all(np.array(v) < 1e-2) and all(np.array(w) < 1e-2)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                obj_class = self.obj_id_to_class[obj_id]
                #dist_sort = np.norm(pos[:2]-) 
                info[obj_id] = (pos, rot, dim, obj_class)
        return info

    def log_video(self, start,base_directory,task_name=0):
        """
        Logs video of each task being executed
        """
        path = os.path.join(base_directory,"video_logs")
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            p.stopStateLogging(self.curr_recording)
            self.video_log_key = task_name
        except Exception:
            print("No Video Currently Being Logged")
        if(start):
            self.curr_recording = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                                  path +
                                                  str(task_name) + ".mp4") 

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def check_sim(self):
        for _ in range(5):
            p.stepSimulation(self.client_id)
        scene_info = self.info
        z_value = np.array([value[0][2] for key, value in scene_info.items()])
        
      
        roll = [abs(p.getEulerFromQuaternion(value[1])[0]) for key, value in scene_info.items()]
        pitch = [abs(p.getEulerFromQuaternion(value[1])[1]) for key, value in scene_info.items()]
        not_fall = max(roll)<1.2 and max(pitch)<1.2
        # NOTE: an object is invalid if it fell below the cabinet floor or toppled over
        # (the y-bounds check that used to be computed here was never used in the result).
        return bool(np.all(z_value > self.bounds[2][0]-0.1) and not_fall)


    def release(self):
      """Release gripper object, only applied if gripper is 'activated'.
      If suction off, detect contact between gripper and objects.
      If suction on, detect contact between picked object and other objects.
      To handle deformables, simply remove constraints (i.e., anchors).
      Also reset any relevant variables, e.g., if releasing a rigid, we
      should reset init_grip values back to None, which will be re-assigned
      in any subsequent grasps.
      """
      if self.activated:
        self.activated = False

        # Release gripped rigid object (if any).
        if self.contact_constraint is not None:
          try:
            p.removeConstraint(self.contact_constraint)
            self.contact_constraint = None
          except:  # pylint: disable=bare-except
            pass
          self.init_grip_distance = None
          self.init_grip_item = None



    def activate(self):
      """Simulate suction using a rigid fixed constraint to contacted object."""
      # TODO(andyzeng): check deformables logic.
      # del def_ids
      dist = 1000.0
      body_pose = p.getLinkState(self.ur5e, self.ur5e_ee_id)
      if not self.activated:
        points = p.getContactPoints(bodyA=self.ur5e, linkIndexA=self.ur5e_ee_id)
        if points:
          # Handle contact between suction with a rigid object.
          body_pose = p.getLinkState(self.ur5e, self.ur5e_ee_id)
          obj_id = -1
          for point in points:
            obj_id_i, contact_link_i, dist_c, posonCup, normal_i = point[2], point[4], point[8], point[5], point[7]

            if obj_id_i in self.obj_ids['rigid']:

              obj_pose = p.getBasePositionAndOrientation(obj_id_i)
              world_to_body = p.invertTransform(body_pose[0], body_pose[1])
              obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                world_to_body[1],
                                                obj_pose[0], obj_pose[1])
              dot_i = normal_i[0]
              if (np.linalg.norm(np.asarray(obj_to_body[0]))<dist) and np.abs(posonCup[1]-body_pose[0][1])<0.017 and posonCup[0]<body_pose[0][0] and dot_i>0.5:
                obj_id = obj_id_i
                contact_link = contact_link_i
                dist = np.linalg.norm(np.asarray(obj_to_body[0]))
                contact_normal = normal_i
                contact_pos = posonCup
          
          if obj_id in self.obj_ids['rigid']:
            body_pose = p.getLinkState(self.ur5e, self.ur5e_ee_id)
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            world_to_body = p.invertTransform(body_pose[0], body_pose[1])
            obj_to_body = p.multiplyTransforms(world_to_body[0],
                                              world_to_body[1],
                                              obj_pose[0], obj_pose[1])
            if body_pose[0][0]>contact_pos[0]:
              offset = contact_pos[0]-body_pose[0][0]
            else:
              offset = 0 
            pos_constrs = (obj_to_body[0][0]+offset,obj_to_body[0][1], obj_to_body[0][2] )
            self.contact_constraint = p.createConstraint(
                parentBodyUniqueId=self.ur5e,
                parentLinkIndex=self.ur5e_ee_id,
                childBodyUniqueId=obj_id,
                childLinkIndex=contact_link,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
            parentFramePosition=((pos_constrs)),
            parentFrameOrientation=((obj_to_body[1])),
            childFramePosition=(0, 0, 0),
            childFrameOrientation=((0, 0, 0)),)
            self.offset_y = obj_to_body[0][1]
            self.offset_x = -pos_constrs[0]
            p.changeConstraint(self.contact_constraint,gearRatio=-1, erp=self.params.suction_erp,maxForce=self.params.suction_max_force)
            self.activated = True
        else:
             closest = []
             min_dist = 1000
             points = []
             for i in self.body_ids:
               close = p.getClosestPoints(bodyA=self.ur5e, linkIndexA=self.ur5e_ee_id,distance=0.015,bodyB=i)
               if close:
                 points_2 = close
                 close = close[0]
                 obj_pose = p.getBasePositionAndOrientation(i)
                 world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                 obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                 world_to_body[1],
                                                 obj_pose[0], obj_pose[1])
                 contact_pos = close[5]
                 if(np.abs(contact_pos[1]-body_pose[0][1])<0.01):
                    closest = close[2] if np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and obj_pose[0][0]>contact_pos[0] else closest
                    points = points_2 if np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and obj_pose[0][0]>contact_pos[0] else points
                    min_dist = np.linalg.norm(np.asarray(obj_to_body[0])) if np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and obj_pose[0][0]>contact_pos[0] else min_dist

             if points:

                # Handle contact between suction with a rigid object.
                for point in points:
                  obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids['rigid']:
                  body_pose = p.getLinkState(self.ur5e, self.ur5e_ee_id)
                  obj_pose = p.getBasePositionAndOrientation(obj_id)
                  world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                  obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                    world_to_body[1],
                                                    obj_pose[0], obj_pose[1])
                
                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.ur5e,
                    parentLinkIndex=self.ur5e_ee_id,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                parentFramePosition=(obj_to_body[0]),
            parentFrameOrientation=((obj_to_body[1])),
                 # childFrameOrientation=((0, 0, 0)),)
            childFramePosition=(0, 0, 0),
            childFrameOrientation=((0,0,0)),)
                p.changeConstraint(self.contact_constraint,gearRatio=-1, erp=self.params.suction_erp,maxForce=self.params.suction_max_force)
                self.offset_x = -obj_to_body[0][0]

                self.activated = True              
                self.offset_y = obj_to_body[0][1]


    def get_target_px(self): 
      info_scene = self.info
      name_color = []
      Y_up = []
      Y_down = []
      Color = []
      for id in info_scene:
        BOUND_TARG_UP = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][1]
        BOUND_TARG_DOWN = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][0]
        Yup = int(np.round((BOUND_TARG_UP-BOUNDS[1,0]) / PIXEL_SIZE))
        Ydown = int(np.round((BOUND_TARG_DOWN-BOUNDS[1,0])  / PIXEL_SIZE))
        color = PLACE_TARGETS_COLORS[self.obj_id_to_name[id].split(' ')[0]]
        if(not self.obj_id_to_name[id].split(' ')[0] in name_color):
          name_color.append(self.obj_id_to_name[id].split(' ')[0])
          Color.append(color)
          Y_up.append(Yup)
          Y_down.append(Ydown)
      return Y_down, Y_up, Color

    def add_target_pixels(self):
       Y_up = []
       Y_down = []
       Color = []
       for key in self.place_targets:
        BOUND_TARG_UP = self.place_targets[key][1]
        BOUND_TARG_DOWN = self.place_targets[key][0]
        Yup = int(np.round((BOUND_TARG_UP-BOUNDS[1,0]) / PIXEL_SIZE))
        Ydown = int(np.round((BOUND_TARG_DOWN-BOUNDS[1,0])  / PIXEL_SIZE))
        color = PLACE_TARGETS_COLORS[key]
        Color.append(color)
        Y_up.append(Yup)
        Y_down.append(Ydown)        
       return Y_down, Y_up, Color

    def check_relabel(self):
      info_scene = self.info
      self.id_list = []
      self.new_class = []
      sorted = 0
      for id in info_scene:
        Y_pos = info_scene[id][0][1]
        skip = False
        for key in self.place_targets:
          if(skip):
            break
          BOUND_TARG_UP = self.place_targets[key][1]
          BOUND_TARG_DOWN = self.place_targets[key][0]
          Yup = int(np.round((BOUND_TARG_UP-BOUNDS[1,0]) / PIXEL_SIZE))
          Ydown = int(np.round((BOUND_TARG_DOWN-BOUNDS[1,0])  / PIXEL_SIZE))
          if(BOUND_TARG_DOWN<=Y_pos<=BOUND_TARG_UP):
              self.id_list.append(id)
              self.new_class.append(LABELS[key])
              sorted+=1
              skip = True
        
      return sorted/len(info_scene)

    def relabel_image(env, color_image, id_image, id_lists, new_class_list):
      class_to_color = {LABELS[k]: np.multiply(COLORS[k][:3], 255) for k in ('blue', 'green', 'yellow')}
      relabeled = []
      for (images, id_images) in zip(color_image, id_image):
        # Work on a copy: the original (un-relabeled) transition is stored in the replay
        # buffer and shares these arrays, so it must not be modified in place.
        image = images.copy()
        for (id, class_new) in zip(id_lists, new_class_list):
          if class_new not in class_to_color:
            continue
          image[id_images == id] = class_to_color[class_new]
        relabeled.append(image)
      return relabeled


    def check_sorted(self):
        for _ in range(1):
            p.stepSimulation(self.client_id)

        info_scene = self.info
        sorted = 0
        for id in info_scene:
            Y_pos = info_scene[id][0][1]
            BOUND_TARG_UP = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][1]
            BOUND_TARG_DOWN = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][0]
            if(BOUND_TARG_DOWN<=Y_pos<=BOUND_TARG_UP):
                sorted+=1
        return sorted/len(info_scene)

    def get_distance_to_goal(self):
        for _ in range(5):
            p.stepSimulation(self.client_id)
        d = 0
        info_scene = self.info
        sorted = 0
        d = 0
        for id in info_scene:
            Y_pos = info_scene[id][0][1]
            Y_pos = min(Y_pos,BOUNDS[1][1])
            Y_pos = max(Y_pos,BOUNDS[1][0])
            BOUND_TARG_UP = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][1]
            BOUND_TARG_DOWN = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][0]
            if(BOUND_TARG_DOWN<=Y_pos<=BOUND_TARG_UP):
              d = max(d,0)
            else:
              d = max(min(abs(Y_pos - BOUND_TARG_UP), abs(Y_pos - BOUND_TARG_DOWN)),d)
        return d

    def check_sort_status(self):
        info_scene = self.info
        sorted = 0
        for id in info_scene:
            Y_pos = info_scene[id][0][1]
            Y_pos = min(Y_pos,BOUNDS[1][1])
            Y_pos = max(Y_pos,BOUNDS[1][0])
            BOUND_TARG_UP = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][1]
            BOUND_TARG_DOWN = self.place_targets[self.obj_id_to_name[id].split(' ')[0]][0]
            if(BOUND_TARG_DOWN<=Y_pos<=BOUND_TARG_UP):
                sorted+=1
        return sorted

    def add_objects(self):
        """List of (fixed, rigid) objects in env."""
        NO = min(int(np.random.choice(self.num_obj)), 15)  # at most 5 objects per class, 3 classes
        obj = sample(objects,3)
        obj = [x+' '+y for x,y in zip(colors,obj)]
        print(obj)

        obj_names = []
        yellow = 0
        green = 0
        blue = 0
        while len(obj_names)<NO:
          col = np.random.choice(obj)
          if col.split(' ')[0] == 'blue' and blue < 5:
            blue +=1
            obj_names.append(col)
          elif col.split(' ')[0] == 'yellow' and yellow < 5:
            yellow +=1
            obj_names.append(col)
          elif col.split(' ')[0] == 'green' and green < 5:
            green +=1
            obj_names.append(col)
        self.randomize()
        self.obj_name_to_id = {}
        self.obj_id_to_color = {}
        self.obj_id_to_name = {}
        self.obj_id_to_class = {}
        self.obj_id_to_target = {}
        self.body_ids = []
        obj_xyz = np.zeros((0, 2))
        ind = 0
        yy = [-0.02, 0.02]
        for obj_name in obj_names:
          if (1):
            BOUND_TARG = BOUNDS[1, :]
 
            # Random position at least min_dist from the other objects (spacing shrinks if the cabinet is too full).
            min_dist, attempts = 0.1, 0
            while True:
              attempts += 1
              if attempts % 100 == 0:
                min_dist *= 0.8
              rand_x = np.random.uniform(BOUNDS[0, 0] +0.05, BOUNDS[0, 1] - 0.1)
              rand_y = np.random.uniform(BOUNDS[1, 0] + 0.02, BOUNDS[1,1] - 0.02)


              rand_xyz = np.float32([rand_x, rand_y, BOUNDS[2][0]+0.0]).reshape(1, 3)
              rand_xy = np.float32([rand_x, rand_y]).reshape(1, 2)
              if len(obj_xyz) == 0:
                obj_xyz = np.concatenate((obj_xyz, rand_xy), axis=0)
                break
              else:
                nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xy, axis=1)).squeeze()
                if nn_dist > min_dist:
                  obj_xyz = np.concatenate((obj_xyz, rand_xy), axis=0)
                  break

            object_color = deepcopy(COLORS[obj_name.split(' ')[0]])
            object_dim = deepcopy(DIMENSIONS[obj_name.split(' ')[0]])
            object_dim[0] += np.random.uniform(-0.005,0.01)
            object_dim[1] += np.random.uniform(-0.005,0.01)

            object_dim[2] += np.random.uniform(-0.01,0.01)
            object_dim[2] = max(object_dim[2],0.03)

            object_type = obj_name.split(' ')[1]
            object_position = rand_xyz.squeeze()
            if object_type == 'block':
              object_position[2] += object_dim[2] + 0.002  # rest on the floor instead of starting half inside it
              object_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=object_dim)
              object_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=object_dim)
              object_id = p.createMultiBody(self.params.block_mass, object_shape, object_visual, basePosition=object_position)
            elif object_type == 'bowl':
              object_position[2] = 0
              object_id = p.loadURDF("assets/ycb_objects/YcbBowl/model.urdf", object_position, useFixedBase=1)
            elif object_type == 'cylinder':
              radius = 0.03+np.random.uniform(-0.005,0.005)
              object_position[2] += object_dim[2] + 0.002
              object_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height = 2*object_dim[2])
              object_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length = 2*object_dim[2])
              object_id = p.createMultiBody(self.params.cylinder_mass, object_shape, object_visual, basePosition=object_position)
            else:
                path = "assets/ycb_objects/Ycb"+object_type+"/model.urdf"
                object_id = p.loadURDF(path, basePosition=object_position,baseOrientation=p.getQuaternionFromEuler([0,0,np.random.uniform(-3.14,3.14)]))
            if self.params.randomize:
                mass = p.getDynamicsInfo(object_id, -1)[0]
                p.changeDynamics(object_id, -1, lateralFriction=np.random.uniform(*self.params.friction_range),
                                 mass=mass * np.random.uniform(*self.params.mass_scale_range))
            self.obj_id_to_name[object_id] = obj_name
            self.obj_id_to_color[object_id] = object_color
            self.obj_id_to_class[object_id] = LABELS[self.obj_id_to_name[object_id].split(' ')[0]]
            self.obj_id_to_target[object_id] = self.place_targets[self.obj_id_to_name[object_id].split(' ')[0]]
            for _ in range(20):
                p.stepSimulation(self.client_id)

            self.body_ids.append(object_id)
            self.add_object_id(object_id, category="rigid")
            ind +=1

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.ur5e, self.ur5e_ee_id
        if self.activated and self.contact_constraint is not None:
          try:
            info = p.getConstraintInfo(self.contact_constraint)
            body, link = info[2], info[3]
          except:  # pylint: disable=bare-except
            self.contact_constraint = None
            pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
          points = [point for point in points if point[2] != self.ur5e]  # ignore contacts with the robot itself
        if points:

          return True
        else:
            return False
            closest = []
            min_dist = 1000
            points = []
            body_pose = p.getLinkState(self.ur5e, self.ur5e_ee_id)

            for i in self.body_ids:
              close = p.getClosestPoints(bodyA=self.ur5e, linkIndexA=self.ur5e_ee_id,distance=0.01,bodyB=i)
              if close:
                points_2 = close
                close = close[0]
                obj_pose = p.getBasePositionAndOrientation(i)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                world_to_body[1],
                                                obj_pose[0], obj_pose[1])
                closest = close[2] if (np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and abs(obj_to_body[0][0])>0) else closest
                points = points_2 if (np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and abs(obj_to_body[0][0])>0) else points
                min_dist = np.linalg.norm(np.asarray(obj_to_body[0])) if (np.linalg.norm(np.asarray(obj_to_body[0])) < min_dist and abs(obj_to_body[0][0])>0) else min_dist

            if closest:
              return True

        return False

    def wait_static(self, timeout=1):
        """Step simulator asynchronously until objects settle."""
        p.stepSimulation(self.client_id)
        for _ in range(self.WAIT_STATIC_MAX_STEPS):  # simulation steps, not wall-clock => machine independent
            if self.is_static:
                return True
            p.stepSimulation(self.client_id)
        return False




    def randomize(self):
        rand_int = np.random.randint(0,len(PLACE_CHOICES))
        PLACE_TARGETS_RAND = PLACE_CHOICES[rand_int]
        PLACE_SELECT = random.sample(PLACE_TARGETS_RAND,3)

        self.place_targets = deepcopy(PLACE_TARGETS_ORG)
        n_obj = 0
        for key in self.place_targets:
            self.place_targets[key][0] = PLACE_SELECT[n_obj][0]
            self.place_targets[key][1] = PLACE_SELECT[n_obj][1]
            n_obj+=1
        return self.place_targets  


    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.use_gripper = False
        self._randomize_camera()
        p.resetSimulation()
        p.setTimeStep(self.time_step)  # resetSimulation restores the default 1/240 s
        p.setGravity(0, 0, -9.81)
        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Load UR5e
        self.ur5e = p.loadURDF("assets/ur5e/ur5e.urdf", basePosition=(0.6, 0, 0), useFixedBase=True)
        self.ur5e_joints = []
        for i in range(p.getNumJoints(self.ur5e)):
            info = p.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "fixFrame":
                self.ur5e_ee_id = joint_id
            if joint_name == "wrist_3_joint":
                self.wrist_id = joint_id
            if joint_name == "wrist_2_joint":
                self.wrist_2_id = joint_id

            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.ur5e_joints.append(joint_id)
        p.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)
        p.enableJointForceTorqueSensor(self.ur5e, 11, 1)
        p.enableJointForceTorqueSensor(self.ur5e, 10, 1)


        # Move robot to home joint configuration.
        success = self.go_home()

        if not success:
            print("Simulation is wrong!")
        self.plane = p.loadURDF("plane.urdf", basePosition=(0, 0, -0.00), useFixedBase=True)

        self.createCabinet()

        self.add_objects()
        self.wait_static()
        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return self.check_sim()


    def step(self, act):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        if act['name']=='push':
            pose0 = act['pose0']
            pose1 = act['pose1']
            success = self.push(pose0, pose1)
            # Exit early if action times out.
        if act['name']=='pick':
            pose0 = act['pose0']
            success = self.pick(pose0)
        # Step simulator asynchronously until objects settle.
        if act['name']=='place':
            pose0 = act['pose0']
            success = self.place(pose0)

        while not self.is_static:
            p.stepSimulation(self.client_id)

        # Get RGB-D camera image observations.
        obs = self.get_observation()
        return obs, success

    def _randomize_camera(self):
        """Domain randomisation of the camera pose and sensor noise (no-op unless params.randomize)."""
        cfg = dict(config_camera)
        if self.params.randomize:
            cfg['position'] = tuple(np.asarray(config_camera['position']) +
                                    np.random.uniform(-self.params.cam_pos_jitter, self.params.cam_pos_jitter, 3))
            cfg['rotation'] = tuple(np.asarray(config_camera['rotation']) +
                                    np.deg2rad(np.random.uniform(-self.params.cam_rot_jitter_deg,
                                                                 self.params.cam_rot_jitter_deg, 3)))
        cfg['noise'] = bool(self.params.sensor_noise)
        self.camera_config = cfg

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def get_observation(self):
      observation = {}

      # Render current image.
      color, depth, segm, position, orientation, intrinsics = self.render_camera(self.camera_config)
      observation['color'] = color.copy()
      observation['depth'] = depth.copy()
      # Get heightmaps and colormaps.
      masks = None
      ID_Mask = None
      segm_ids = np.unique(segm)
      sgms_binary = segm.copy()
      segm_plane = np.zeros_like(color)
      Color_Mask = None
      Obj_ID = None
      for sid in segm_ids:
        if sid not in self.body_ids:
          color[segm==sid] = [[255,255,255]]
          segm_plane[segm==sid] = [[255,255,255]]
          segm[segm == sid] = 0
          sgms_binary[sgms_binary==sid] = 0
        else:
          color[segm==sid] = [np.multiply(COLORS[self.obj_id_to_name[sid].split(' ')[0]][:3],255)]
          segm_obj = np.zeros_like(color)
          segm_obj[segm==sid] = [np.multiply(COLORS[self.obj_id_to_name[sid].split(' ')[0]][:3],255)]
          segm_obj = np.expand_dims(segm_obj, axis=0)
          obj_class = None
          obj_class = LABELS[self.obj_id_to_name[sid].split(' ')[0]]
          obj_class  = np.expand_dims(obj_class,axis=0)
          obj_id  = np.expand_dims(sid,axis=0)
          obj_color = [np.multiply(COLORS[self.obj_id_to_name[sid].split(' ')[0]][:3],255)]
          if(masks is None):
            masks = segm_obj
            ID_Mask = obj_class
            Color_Mask = obj_color
            Obj_ID = obj_id
          else:
            masks = np.concatenate((masks,segm_obj), axis = 0)
            ID_Mask = np.concatenate((ID_Mask,obj_class),axis=0)
            Color_Mask = np.concatenate((Color_Mask,obj_color),axis=0)
            Obj_ID = np.concatenate((Obj_ID,obj_id),axis=0)
          
          segm[segm==sid] = LABELS[self.obj_id_to_name[sid].split(' ')[0]]
          sgms_binary[sgms_binary==sid] = 1
      observation['mask'] = segm
      
      
      points = self.get_pointcloud(depth, intrinsics)
      position = np.float32(position).reshape(3, 1)
      rotation = p.getMatrixFromQuaternion(orientation)
      rotation = np.float32(rotation).reshape(3, 3)
      transform = np.eye(4)
      transform[:3, :] = np.hstack((rotation, position))
      points = self.transform_pointcloud(points, transform)
      heightmap, colormap = self.get_heightmap(points, color, segm_plane, BOUNDS, PIXEL_SIZE)

      Y_down, Y_up, _ = self.add_target_pixels()

      heightmap_on_target = np.zeros_like(heightmap)
      binary_mask_map = np.zeros_like(heightmap)
      class_mask_map = np.zeros_like(heightmap)
      id_mask_map = np.zeros_like(heightmap)

      masks_heightmap = None
      id_mask = 0
      if masks is not None:
        for m in masks:
          heightmap_mask, cmask = self.get_heightmap_mask(points, m, BOUNDS, PIXEL_SIZE)
          obj_mask_id = Obj_ID[id_mask]

          color_mask = Color_Mask[id_mask]
          object_pixels = np.where((cmask[:,:,0] == color_mask[0]) & (cmask[:,:,1] == color_mask[1]) & (cmask[:,:,2] == color_mask[2]))
          print(object_pixels)
          mask = np.zeros(heightmap_mask.shape[:2], np.uint8)
          mask[object_pixels] = 1
          M = cv2.moments(mask,binaryImage = True)
          obj_on_target = False
          if M['m00'] == 0:
              id_mask +=1
              continue
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          obj_class = ID_Mask[id_mask]
          if (obj_class == 1):
            obj_on_target = Y_down[0] <= cy <= Y_up[0]
          elif obj_class==5:
            obj_on_target = Y_down[1] <= cy <= Y_up[1]
          elif obj_class==3:
            obj_on_target = Y_down[2] <= cy <= Y_up[2]

          id_mask_map[object_pixels] = obj_mask_id

          binary_mask_map[object_pixels] = 1
          class_mask_map[object_pixels] = obj_class
          if obj_on_target:
            heightmap_on_target[object_pixels] = -1

          id_mask +=1
  
      
      free_space = np.ones_like(heightmap)
      # free_space[Y_down[0]:Y_up[0],:] = 1+ 
      # free_space[Y_down[1]:Y_up[1],:] = 

      object_pixels = np.where((class_mask_map!=0) | ((colormap[:,:,0]==105) & (colormap[:,:,1]==105) & (colormap[:,:,2]==105)))
      binary_mask_map_original = deepcopy(binary_mask_map)

      object_pixels = np.where( (binary_mask_map==1) & (heightmap_on_target!=-1))
      binary_mask_map[object_pixels[0],:] = 1
      object_pixels = np.where( (binary_mask_map==1) & (heightmap_on_target==-1))

      res_R = ndimage.interpolation.shift(binary_mask_map, [-2,0], order=0)
      binary_mask_map = cv2.bitwise_or(binary_mask_map, res_R)
      res_L = ndimage.interpolation.shift(binary_mask_map, [2,0], order=0)

      binary_mask_map = cv2.bitwise_or(binary_mask_map, res_L)
      for i in range(4):
        res_R = np.pad(binary_mask_map,((0,0),(1,0)), mode='constant')[:,:-1]      
        binary_mask_map = cv2.bitwise_or(binary_mask_map, res_R)
      for i in range(4):
        res_R = np.pad(binary_mask_map,((0,0),(0,1)), mode='constant')[:,1:]      
        binary_mask_map = cv2.bitwise_or(binary_mask_map, res_R)





      if np.all(binary_mask_map==1):
        binary_mask_map = binary_mask_map_original
      heightmap_on_target[free_space!=0] = -10

      heightmap_on_target[0:8,:] = -10
      heightmap_on_target[heightmap_on_target.shape[0]-8:heightmap_on_target.shape[0],:] = -10
      heightmap_on_target[:,heightmap_on_target.shape[1]-8:heightmap_on_target.shape[1]] = -10
      occluded = np.where((colormap[:,:,0]==105) & (colormap[:,:,1]==105) & (colormap[:,:,2]==105))
      heightmap_on_target[occluded] = -10

      free_space_original = free_space
      free_space = free_space-binary_mask_map
      free_space[free_space<0] = 0
      free_space[:,0:8] = 0
      free_space[:,-8:] = 0
      free_space[0:5,:] = 0
      free_space[-5:,:] = 0
      if np.all(free_space==0):
        binary_mask_map = binary_mask_map_original
        free_space = free_space_original
        free_space = free_space-binary_mask_map
        free_space[free_space<0] = 0
        free_space[:,0:8] = 0
        free_space[:,-6:] = 0
        free_space[0:4,:] = 0
        free_space[-4:,:] = 0

      padding_width = ((int(4),int((4))), ((4,0)))       
      color_heightmap_sq_r =  np.pad(colormap[:,:,0], padding_width, 'constant', constant_values=0)
      color_heightmap_sq_r.shape = (color_heightmap_sq_r.shape[0], color_heightmap_sq_r.shape[1], 1)
      color_heightmap_sq_g =  np.pad(colormap[:,:,1], padding_width, 'constant', constant_values=0)
      color_heightmap_sq_g.shape = (color_heightmap_sq_g.shape[0], color_heightmap_sq_g.shape[1], 1)
      color_heightmap_sq_b =  np.pad(colormap[:,:,2], padding_width, 'constant', constant_values=0)
      color_heightmap_sq_b.shape = (color_heightmap_sq_b.shape[0], color_heightmap_sq_b.shape[1], 1)
      colormap = np.concatenate((color_heightmap_sq_r, color_heightmap_sq_g, color_heightmap_sq_b), axis=2)
      heightmap =  np.pad(heightmap, padding_width, 'constant', constant_values=0.3)

      observation['freespace'] = free_space

      observation["classmap"] = class_mask_map
      observation['idmap'] = id_mask_map
      observation["colormap"] = colormap
      observation["ontargetmap"] = heightmap_on_target
      observation["depthmap"] = heightmap
      return observation


    def render_camera(self, config_camera):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        quaternion = p.getQuaternionFromEuler(config_camera["rotation"])
        rotation = p.getMatrixFromQuaternion(quaternion)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config_camera["position"] + lookdir
        focal_len = config_camera["intrinsics"][0]
        znear, zfar = config_camera["zrange"]
        viewm = p.computeViewMatrix(config_camera["position"], lookat, updir)
        fovh = (config_camera["image_size"][0] / 2) / focal_len
        fovh = (np.arctan(fovh) * 2 / np.pi) * 180 
        intrinsics = np.float32(config_camera["intrinsics"]).reshape(3, 3)
        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config_camera["image_size"][1] / config_camera["image_size"][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config_camera["image_size"][1],
            height=config_camera["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # OpenGL needs a display; the CPU renderer works headless (containers, CI)
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
        )

        # Get color image.
        color_image_size = (config_camera["image_size"][0], config_camera["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config_camera["noise"]:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config_camera["image_size"][0], config_camera["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config_camera["noise"]:
            depth += np.random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        return color, depth, segm, config_camera["position"], quaternion, intrinsics 

    def __del__(self):
        p.disconnect()

    def get_link_pose(self, body, link):
        result = p.getLinkState(body, link)
        return result[4], result[5]

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.02, timeout=0.5):
        """Move UR5e to target joint configuration."""
        for _ in range(self.MOVE_JOINTS_MAX_STEPS):  # simulation steps, not wall-clock
            current_joints = np.array([p.getJointState(self.ur5e, i)[0] for i in self.ur5e_joints])
            diff_joints = target_joints - current_joints
            diff_joints[-1] = 0
            if all(np.abs(diff_joints) < 1e-3):
                # give time to stop
                for _ in range(1):
                    p.stepSimulation(self.client_id)
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = 100*diff_joints
            step_joints = current_joints + v*speed
            step_joints[-1] = np.pi
            p.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=0.4*np.ones(len(self.ur5e_joints)),
            )
            p.stepSimulation(self.client_id)

        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints, success = self.solve_ik(pose, speed=speed)
        return success     

    def accurateCalculateInverseKinematics(self, pose, threshold=0.002, maxIter=5, speed = 0.03):
      closeEnough = False
      iter = 0
      while (not closeEnough and iter < maxIter):
        jointPoses = p.calculateInverseKinematics(bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
             targetOrientation=pose[1]) #lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283, -10000]*3,
            # jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566, 20000]*3)

        success = self.move_joints(jointPoses,speed)

        if(not success):
          return jointPoses, False
        ls = p.getLinkState(self.ur5e, self.ur5e_ee_id)
        targetPos = pose[0]
        newPos = ls[4]
        diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
        diff = np.array(diff)
        closeEnough = np.linalg.norm(diff) < threshold
        dist2 = closeEnough
        iter = iter + 1
      return jointPoses, True
    
    
    
    def solve_ik(self, pose, speed):
        """Calculate joint configuration with inverse kinematics."""
        joints, success = self.accurateCalculateInverseKinematics(pose, speed=speed)

        joints = np.array(joints, dtype=np.float32)
        return joints, success

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, detect_contact=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.002  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        if(length==0):
            return True
        vec = vec / length
        n_push = np.int32(np.floor(np.linalg.norm(pose1 - pose0) / step_distance))  # every 1 cm

        success = True
        for n in range(n_push):
            target = pose0 + n * vec * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            for _ in range(1):
              p.stepSimulation(self.client_id)

            if not success:
                 return success
            if self.activated:
              start_sensing = 0.2
            else:
              start_sensing = 0.5 

            if detect_force and target[0]<start_sensing:
                force = np.sum(np.abs(np.array(p.getJointState(self.ur5e, self.ur5e_ee_id)[2][:3])))
                force2 = np.sum(np.abs(np.array(p.getJointState(self.ur5e, 11)[2][:3])))
                force3 = np.sum(np.abs(np.array(p.getJointState(self.ur5e, 10)[2][:3])))
                
                if force > max_force or force2> max_force or force3 > max_force:
                    target = pose0 +(n-1)* vec * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False
            if detect_contact:

              if(self.detect_contact()):
                if np.linalg.norm(np.float32(pose1)-np.float32(target))<0.01:
                  return True
                else:
                  pass

        success &= self.move_ee_pose((pose1, rot), speed)
        return success 

    def push(self, pose0, angle=0., speed=0.01):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """
        success = self.move_joints(self.ik_rest_joints)

        # Adjust push start and end positions.
        pos0 = np.zeros((3))
        posxy = np.asarray(pose0).copy()
        pos0[0] = posxy[0]
        pos0[1] = posxy[1]
        pos0[2] = self.bounds[2][0] + 0.05
        push_length = 0.08
        pos1 = [pos0[0], pos0[1]+push_length*np.sin(angle), pos0[2]]
        pos1[1] = min(pos1[1],self.bounds[1,1]-0.008)
        pos1[1] = max(pos1[1],self.bounds[1,0]+0.008)

        vec = pos1 - pos0
        # Align against push direction.
        grasp_location_margin = 0.1
        angle = 0
        quat = (0.7,0,-0.7,0)
        pos, ori = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)

        location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos0[1], pos0[2])
        tool_position = location_front_grasp_target
        success = self.move_ee_pose((tool_position, quat), speed)
        # Execute push.
        if success:
            success = self.straight_move(tool_position, pos0, quat,speed, detect_force=True)
        else:
            pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
            pos = np.asarray(pos)
            location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos[1], pos[2])
            tool_position = location_front_grasp_target            

            self.straight_move(pos, tool_position, quat, speed)
            self.go_home()
            return success
        if success:
            success = self.straight_move(pos0, pos1, quat, 0.1*speed, detect_force=True)

        else:

            pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
            pos = np.asarray(pos)
            print("Error, Going back home")
            location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos[1], pos[2])
            tool_position = location_front_grasp_target            

            self.straight_move(pos, tool_position, quat, speed)
            self.go_home()
            return success
        if success:
            pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
            pos = np.asarray(pos)
            self.move_ee_pose((pos0, quat), speed)
            location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos0[1], pos0[2])
            tool_position = location_front_grasp_target            
            self.straight_move(pos0, tool_position, quat, speed)
            self.go_home()
        else:
            pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
            pos = np.asarray(pos)
            self.move_ee_pose((pos0, quat), speed)
            location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos0[1], pos0[2])
            tool_position = location_front_grasp_target            
            self.straight_move(pos, tool_position, quat, speed)
            self.go_home()


        success &= self.check_sim()
        return success

    def pick(self, position, speed=0.01):

      pos0 = np.zeros((3))
      posxy = np.asarray(position).copy()
      pos0[0] = posxy[0]
      pos0[1] = posxy[1]
      pos0[2] = self.bounds[2][0] + 0.04
      success = self.move_joints(self.ik_rest_joints)
      success = False
      pos, ori = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
      quat = ori
      pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id) 
      self.release()
      grasp_location_margin = 0.1
      location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, pos0[1], pos0[2])

      tool_position = location_front_grasp_target

      pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
      pos = np.asarray(pos)
      success = self.move_ee_pose((tool_position, quat), speed=speed)
    
      if success:
        success = self.straight_move(tool_position,pos0, quat,speed=0.5*speed, detect_contact=True, detect_force=True)
        pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)
        if (np.linalg.norm(np.float32(pos0)-np.float32(pos)))<0.05:
          self.activate()
          success = True
      else:
        self.straight_move(pos,tool_position, quat,speed)
        self.go_home()
        return success     
      if success:
        self.activate()
        location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin+0.2, pos[1], pos[2]+0.0)

        tool_position = location_front_grasp_target

        self.straight_move(pos,tool_position, quat,speed)
        self.go_home()

        grasp_sucess = self.activated
        success &= grasp_sucess
      else:
        pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)
        self.straight_move(pos,tool_position, quat,speed)
        self.go_home()

      

      self.go_home()

      success &= self.check_sim()

      return success
    
    def place(self, pos, heightmap_rotation_angle=0, speed=0.01):
      
      
      position = np.zeros((3))
      posxy = np.asarray(pos).copy()
      position[0] = posxy[0]
      position[1] = posxy[1]

      position[2] = self.bounds[2][0] + 0.05
      position[0] -= self.offset_x
      quat = (0.7088025212287903, 0.0003121407644357532, -0.7054068446159363, 0.0003104391216766089)
      position[1] +=self.offset_y
      grasp_location_margin = 0.15
      location_front_grasp_target = (self.bounds[0][1]+grasp_location_margin, position[1], position[2])

      tool_position = location_front_grasp_target

      pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
      pos = np.asarray(pos)




      




      success = self.move_ee_pose((tool_position, quat), speed=speed)
      if not success:
        success = self.move_ee_pose((tool_position, quat), speed=speed)
        success = True

      if success:
        success = self.straight_move(tool_position,position, quat,speed, detect_force=True)

      else:

        self.release()
        pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)
        success = self.straight_move(pos,tool_position, quat,speed, detect_force=False)
        self.go_home()

      if success:
        self.release()
        pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)
        success = self.straight_move(pos,tool_position, quat,speed, detect_force=False)
        self.go_home()
      else:
        self.release()
        success = False
        pos, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.asarray(pos)
        success = self.straight_move(pos,tool_position, quat,speed, detect_force=False)
        self.go_home()

      self.go_home()
      success &= self.check_sim()
      self.release()

      return success
 
 


    
    def createCabinet(self): 
      ori = [0,0,0,1]
      color= [186/255, 176/255, 172/255, 1]
      mass = 0  # static walls
      width = 0.2
      height = BOUNDS[2][1]-BOUNDS[2][0]
      bottom_width = BOUNDS[2][0]
      top_width = 0.05
      x_disp = BOUNDS[0][0]
      # Bottom 
      half_extent = [self.final_scene_size[0]/2, self.final_scene_size[1]/2, bottom_width/2]
      wall_cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extent)
      wall_vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
      self.bottom_wall_id = p.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
          basePosition=[0,0, bottom_width/2],
          baseOrientation=ori)

      p.changeDynamics(self.bottom_wall_id, -1, lateralFriction=self.params.cabinet_friction)

      # Back
      half_extent = [width/2, self.final_scene_size[1]/2+width, height/2+bottom_width/2]
      wall_cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extent)
      wall_vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
      self.back_wall_id = p.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
          basePosition=[-self.final_scene_size[0]/2-width/2-0.001,0,height/2+bottom_width/2],
          baseOrientation=ori)

      # Left
      half_extent = [self.final_scene_size[0]/2, width/2, height/2+bottom_width/2]
      wall_cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extent)
      wall_vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
      self.left_wall_id = p.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
          basePosition=[0,-width/2-self.final_scene_size[1]/2-0.001,height/2.0 + bottom_width/2],
          baseOrientation=ori)

      # Right
      half_extent = [self.final_scene_size[0]/2, width/2, height/2+bottom_width/2]
      wall_cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extent)
      wall_vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
      self.right_wall_id = p.createMultiBody(mass, baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
          basePosition=[0,width/2+self.final_scene_size[1]/2+0.001,height/2.0 + bottom_width/2],
          baseOrientation=ori)

      # Top
      half_extent = [self.final_scene_size[0]/2 + width/2, self.final_scene_size[1]/2 + width, top_width/2]
      wall_cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extent)
      wall_vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extent, rgbaColor=color)
      self.top_wall_id = p.createMultiBody(0,baseCollisionShapeIndex=wall_cid, baseVisualShapeIndex=wall_vid, 
          basePosition=[-width/2,0, height + bottom_width+top_width/2],
          baseOrientation=ori)
      

      self.scene_height = bottom_width





    def get_pointcloud(self, depth, intrinsics):
      """Get 3D pointcloud from perspective depth image.
      Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
      Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
      """
      height, width = depth.shape
      xlin = np.linspace(0, width - 1, width)
      ylin = np.linspace(0, height - 1, height)
      px, py = np.meshgrid(xlin, ylin)
      px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
      py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
      points = np.float32([px, py, depth]).transpose(1, 2, 0)
      return points

    def transform_pointcloud(self, points, transform):
      """Apply rigid transformation to 3D pointcloud.
      Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
      Returns:
        points: HxWx3 float array of transformed 3D points.
      """
      padding = ((0, 0), (0, 0), (0, 1))
      homogen_points = np.pad(points.copy(), padding,
                              'constant', constant_values=1)
      for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
      return points

    def get_heightmap(self, points, colors, segm_plane, bounds, pixel_size):
      """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
      Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
          region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
      Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        xyzmap: HxWx3 float array of XYZ points in world coordinates.
      """

      width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
      height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
      heightmap = np.zeros((height, width), dtype=np.float32)
      colormap = np.full((height, width, colors.shape[-1]),(105,105,105), dtype=np.uint8)
      # if 1 < 0.5:
      # else:
      Y2, Y1, CT = self.add_target_pixels()
      # for ind in range(y2,y1):
      #   if colormap[ind,0:224]==[255,255,255]:
      # Filter out 3D points that are outside of the predefined bounds.
      ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] <= bounds[0, 1])
      iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] <= bounds[1, 1])
      iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] <= bounds[2, 1])
      valid = ix & iy & iz
      points = points[valid]
      colors = colors[valid]

      # Sort 3D points by z-value, which works with array assignment to simulate
      # z-buffering for rendering the heightmap image.
      iz = np.argsort(points[:, -1])
      points, colors = points[iz], colors[iz]
      px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
      py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
      px = np.clip(px, 0, width - 1)
      py = np.clip(py, 0, height - 1)
      z_bottom = bounds[2][0]
      heightmap[py, px] = points[:, 2] - bounds[2][0]
      heightmap[heightmap < 0.008] = 0
      heightmap[heightmap == -z_bottom] = 0

      for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
      xv, yv = np.meshgrid(np.linspace(bounds[0, 0], bounds[0, 1], height),
                          np.linspace(bounds[1, 0], bounds[1, 1], width))
      

      
      
      for i in range(len(Y2)):
        y2 = Y2[i]
        y1 = Y1[i]
        cc = CT[i]
        for ind in range(y2,y1):
          for xx in range(width):
            if np.array_equal(colormap[ind,xx], [255,255,255]):
              colormap[ind,xx] = cc

      return heightmap, colormap


    def get_heightmap_mask(self, points, mask, bounds, pixel_size, no_pad = False):
      """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
      Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
          region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
      Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        xyzmap: HxWx3 float array of XYZ points in world coordinates.
      """
      width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
      height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
      heightmap = np.zeros((height, width), dtype=np.float32)
      colormap = np.zeros((height, width, mask.shape[-1]), dtype=np.uint8)

      # Filter out 3D points that are outside of the predefined bounds.
      ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
      iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
      iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
      valid = ix & iy & iz
      points = points[valid]
      mask = mask[valid]
      
      # Sort 3D points by z-value, which works with array assignment to simulate
      # z-buffering for rendering the heightmap image.
      iz = np.argsort(points[:, -1])
      points = points[iz]
      mask = mask[iz]
      px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
      py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
      px = np.clip(px, 0, width - 1)
      py = np.clip(py, 0, height - 1)
      heightmap[py, px] = mask[:, 0]
      for c in range(mask.shape[-1]):
        colormap[py, px, c] = mask[:, c]

      if no_pad:
        return heightmap, colormap
      
      return heightmap, colormap

    def xyz_to_pix(self,position):
      """Convert from 3D position to pixel location on heightmap."""
      u = int(np.round((position[1] - BOUNDS[1, 0]) / PIXEL_SIZE))
      v = int(np.round((position[0] - BOUNDS[0, 0]) / PIXEL_SIZE))
      return (u, v)

    def px_to_xyz(self,px,py,depth_map):
      """Convert from 3D position to pixel location on heightmap."""
      height = int(np.round((self.bounds[1, 1] - self.bounds[1, 0]) / PIXEL_SIZE))
      width = int(np.round((self.bounds[0, 1] - self.bounds[0, 0]) / PIXEL_SIZE))
      if px>width:
        px = width
      finger_width = 0.03
      safe_kernel_width = int(np.round((finger_width)/PIXEL_SIZE))
     
      
      local_patch_width = 0.03
      local_patch_width = int(np.round((local_patch_width/2)/PIXEL_SIZE)) 
      local_region = depth_map[max(py-local_patch_width,0):min(py+local_patch_width+1,height),max(px-local_patch_width,0):min(px+local_patch_width+1,width)]      
      if(local_region.size==0):
        z_safe = self.bounds[2][0]
      else:
        z_safe = np.max(local_region) + self.bounds[2][0]   

      xyz = [px * PIXEL_SIZE + self.bounds[0][0],
                    py * PIXEL_SIZE + self.bounds[1][0],
                    z_safe]
      return xyz
    
    def px_to_xy(self,px,py):
      """Convert from 3D position to pixel location on heightmap."""
      height = int(np.round((self.bounds[1, 1] - self.bounds[1, 0]) / PIXEL_SIZE))
      width = int(np.round((self.bounds[0, 1] - self.bounds[0, 0]) / PIXEL_SIZE))
     
      xy = [px * PIXEL_SIZE + self.bounds[0][0],
                    py * PIXEL_SIZE + self.bounds[1][0]]
      return xy