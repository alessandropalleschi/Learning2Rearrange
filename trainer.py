import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import PushFCN, PickFCN
from memory import *
from scipy import ndimage
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, is_testing=False, load_snapshot=False, snapshot_push=[], snapshot_pick=[],snapshot_push_target=[],snapshot_pick_target=[]):
        
        self.buffer = ReplayMemory(8000)
        self.episode = ReplayMemory(50)
        self.double = True
        self.iteration = 0
        self.model_push = PushFCN()
        self.model_push_target = PushFCN()        
        self.model_pick = PickFCN()
        self.model_pick_target = PickFCN()  
        self.push_length_px = 8
        self.future_reward_discount = 0.9
        self.max_dim = 160
        self.offset_px = 4
        if load_snapshot:
            self.model_push.load_state_dict(torch.load(snapshot_push))
            self.model_pick.load_state_dict(torch.load(snapshot_pick))
            self.model_push_target.load_state_dict(torch.load(snapshot_push_target))
            self.model_pick_target.load_state_dict(torch.load(snapshot_pick_target))

            print('Pre-trained model snapshot')
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = True
        if not load_snapshot:
            self._synchronize_q_networks(self.model_push_target, self.model_push)        
            self._synchronize_q_networks(self.model_pick_target, self.model_pick)        
        
        
        self.model_push = self.model_push.cuda()
        self.model_pick = self.model_pick.cuda()
        self.model_push_target = self.model_push_target.cuda()
        self.model_pick_target = self.model_pick_target.cuda()

        # Set model to training mode
        self.model_push.train()
        self.model_pick.train()

        l_r = 1e-4

        self.criterion_push = torch.nn.SmoothL1Loss(reduction='none').cuda() # Huber loss
        self.criterion_grasp = torch.nn.SmoothL1Loss(reduction='none').cuda() # Huber loss
        
        self.optimizer_push = torch.optim.Adam(self.model_push.parameters(), lr=l_r, weight_decay=2e-5)
        self.optimizer_pick = torch.optim.Adam(self.model_pick.parameters(), lr=l_r, weight_decay=2e-5)

    @staticmethod
    def _soft_update_q_network_parameters(q_network_1: nn.Module,
                                          q_network_2: nn.Module,
                                          alpha: float) -> None:
        """In-place, soft-update of q_network_1 parameters with parameters from q_network_2."""
        for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    def _synchronize_q_networks(self,q_network_1: nn.Module, q_network_2: nn.Module) -> None:
        """In place, synchronization of q_network_1 and q_network_2."""
        _ = q_network_1.load_state_dict(q_network_2.state_dict())


    def preprocessQmaps(self, map, depth, primitve):

        depth_input_crop = depth[self.offset_px:depth.shape[0]-self.offset_px,self.offset_px:depth.shape[1]]
        object_pixels = (depth_input_crop>0.008).astype(np.uint8)

        if primitve == 'grasp':
            
            map = np.ma.masked_array(map, np.broadcast_to(1-object_pixels, map.shape, subok=True))        
            map = map.filled(fill_value=-1000)
        elif primitve == 'push':
            kernel = np.ones((self.push_length_px, self.push_length_px), np.uint8)
            contactable_regions = cv2.dilate(object_pixels, kernel, iterations=1)
            contactable_regions[np.nonzero(object_pixels)] = 0
            for i in range(map.shape[0]):
                mapf = np.ma.masked_array(map[i,:,:], np.broadcast_to(1-contactable_regions, map[i,:,:].shape, subok=True))        
                map[i,:,:] = mapf.filled(fill_value=-1000)
        return map

    def forward(self, color_input, depth_input, is_volatile=True, specific_rotation=-1,network='policy'):

        color_heightmap = color_input.copy()
        depth_heightmap = depth_input.copy()

        color_heightmap = color_heightmap.astype(float)/255.
        depth_heightmap = depth_heightmap.astype(float)/0.4


        max_dim = self.max_dim

        padding_width_scal_x = int((max_dim-color_input.shape[0])/2)
        padding_width_scal_y = int((max_dim-color_input.shape[1])/2)

        padding_width = ((int(padding_width_scal_x), int(padding_width_scal_x)), (int(padding_width_scal_y), int(padding_width_scal_y)))
            # color_heightmap_2x = np.pad(color_heightmap_2x, padding_width, 'constant', constant_values=0)

        color_heightmap_r =  np.pad(color_heightmap[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_r.shape = (color_heightmap_r.shape[0], color_heightmap_r.shape[1], 1)
        color_heightmap_g =  np.pad(color_heightmap[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_g.shape = (color_heightmap_g.shape[0], color_heightmap_g.shape[1], 1)
        color_heightmap_b =  np.pad(color_heightmap[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_b.shape = (color_heightmap_b.shape[0], color_heightmap_b.shape[1], 1)
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)


        padding_width = ((int(padding_width_scal_x), int(padding_width_scal_x)), (int(padding_width_scal_y), int(padding_width_scal_y)))

        depth_heightmap =  np.pad(depth_heightmap, padding_width, 'constant', constant_values=0)    

        depth_heightmap = np.tile(depth_heightmap.reshape(
                            depth_heightmap.shape[0], depth_heightmap.shape[1], 1), (1, 1, 3))

        depth_heightmap.shape = (depth_heightmap.shape[0], depth_heightmap.shape[1], depth_heightmap.shape[2], 1)
        color_heightmap.shape = (color_heightmap.shape[0], color_heightmap.shape[1], color_heightmap.shape[2], 1)


            # Construct minibatch of size 1 (b,c,h,w)

        input_color_data = torch.from_numpy(depth_heightmap.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(color_heightmap.astype(np.float32)).permute(3,2,0,1)


        input = torch.cat((input_color_data,input_depth_data),axis=1)

        if network=='policy':
            output_prob_push = self.model_push.forward(input=input,specific_rotation=specific_rotation,is_volatile=is_volatile)
            output_prob_pick = self.model_pick.forward(input=input,is_volatile=is_volatile)
        elif network=='target':
            output_prob_push = self.model_push_target.forward(input=input,specific_rotation=specific_rotation,is_volatile=is_volatile)
            output_prob_pick = self.model_pick_target.forward(input=input,is_volatile=is_volatile)



        push_predictions = output_prob_push.cpu().detach().numpy()[:,0,int(padding_width_scal_x):int(color_heightmap.shape[0] - padding_width_scal_x),int(padding_width_scal_y):int(color_heightmap.shape[1] - padding_width_scal_y)]
        push_predictions = push_predictions[:,self.offset_px:color_input.shape[0]-self.offset_px,self.offset_px:color_input.shape[1]]

        pick_predictions = output_prob_pick.cpu().detach().numpy()[:,0,int(padding_width_scal_x):int(color_heightmap.shape[0] - padding_width_scal_x),int(padding_width_scal_y):int(color_heightmap.shape[1] - padding_width_scal_y)]
        pick_predictions = pick_predictions[:,self.offset_px:color_input.shape[0]-self.offset_px,self.offset_px:color_input.shape[1]]
        pick_predictions = self.preprocessQmaps(pick_predictions,depth_input,'grasp')
        
        
        return push_predictions, pick_predictions


    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):


        size = color_heightmap.shape
        max_dim = self.max_dim
        padding_width_scal_x = int((max_dim-color_heightmap.shape[0])/2)
        padding_width_scal_y = int((max_dim-color_heightmap.shape[1])/2)


        label = np.zeros((1,size[0]+2*padding_width_scal_x,size[1]+2*padding_width_scal_y))
        action_area = np.zeros((size[0],size[1]))
        action_area[best_pix_ind[1]+self.offset_px][best_pix_ind[2]+self.offset_px] = 1
        tmp_label = np.zeros((size[0],size[1]))

        tmp_label[action_area > 0] = label_value
        label[0,padding_width_scal_x:(size[0]+padding_width_scal_x),padding_width_scal_y:(size[1]+padding_width_scal_y)] = tmp_label

                # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((size[0],size[1]))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,padding_width_scal_x:(size[0]+padding_width_scal_x),padding_width_scal_y:(size[1]+padding_width_scal_y)] = tmp_label_weights

        self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        loss = 0
        if primitive_action == 'push':
            loss = self.criterion_push(self.model_push.push_prob[0].view(1,max_dim,max_dim), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            loss = loss.sum()
        elif primitive_action == 'grasp':
            loss = self.criterion_grasp(self.model_pick.pick_prob[0].view(1,max_dim,max_dim), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            loss = loss.sum()


        return loss    

    def get_prediction_vis(self, predictions, color_input, best_pix_ind=[], final_pix=[]):

        canvas = None
        color_heightmap = color_input.copy()
        color_heightmap= color_heightmap[self.offset_px:color_heightmap.shape[0]-self.offset_px,self.offset_px:color_heightmap.shape[1]]
        num_rotations = predictions.shape[0]
        final_pix = np.array(final_pix)
        best_pix_ind = np.array(best_pix_ind)
        size_final = final_pix.shape
        size_init = best_pix_ind.shape
        if num_rotations > 1 and len(best_pix_ind)>0:
            rot_selected = best_pix_ind[0]
        else:
            rot_selected = 0
        for canvas_row in range(int(num_rotations/num_rotations)):
            tmp_row_canvas = None
            for canvas_col in range(predictions.shape[0]):
                rotate_idx = canvas_row+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()

                prediction_vis = prediction_vis+1
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == rot_selected and len(best_pix_ind)>0:
                    # prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[size_init[0]-1]), int(best_pix_ind[size_init[0]-2])), 7, (0,0,255), 2)
                    # prediction_vis = cv2.circle(prediction_vis, (int(final_pix[size_final[0]-1]), int(final_pix[size_final[0]-2])), 7, (255,0,0), 2)
                    prediction_vis = cv2.arrowedLine(prediction_vis, (int(best_pix_ind[size_init[0]-1]), int(best_pix_ind[size_init[0]-2])),(int(final_pix[size_final[0]-1]), int(final_pix[size_final[0]-2])), (0,255,0), 2)
                # if(num_rotations == 1):
                #     prediction_vis = cv2.rectangle(prediction_vis, (int(best_pix_ind[size_init[0]-1])-self.model_pick_and_place.crop_size//2, int(best_pix_ind[size_init[0]-2]-self.model_pick_and_place.crop_size//2)) , (int(best_pix_ind[size_init[0]-1])+self.model_pick_and_place.crop_size//2, int(best_pix_ind[size_init[0]-2]+self.model_pick_and_place.crop_size//2)), (0,0,255), 2)

                prediction_vis = ndimage.zoom(prediction_vis, zoom=[2,2,1], order=0)
                color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0) 
                size = prediction_vis.shape
       
                top_pad = int((320-size[0])/2)
                left_pad = int((320-size[1])/2)

                prediction_vis=cv2.copyMakeBorder(prediction_vis.copy(),top_pad,top_pad,left_pad,left_pad,cv2.BORDER_CONSTANT,value=[0,0,0])
                rot = cv2.getRotationMatrix2D((int(320 // 2),int(320 // 2)), np.radians(rotate_idx*(360/(num_rotations))-90) * 180 / np.pi, 1.0)
 
 
                prediction_vis = cv2.warpAffine(prediction_vis, rot, (320, 320))
                # prediction_vis = ndimage.rotate(prediction_vis,(rotate_idx*(180/num_rotations)), reshape=False, order=0)
                size = color_heightmap_2x.shape
                
                top_pad = int((320-size[0])/2)
                left_pad = int((320-size[1])/2)

                color_heightmap_2x =cv2.copyMakeBorder(color_heightmap_2x.copy(),top_pad,top_pad,left_pad,left_pad,cv2.BORDER_CONSTANT,value=[0,0,0])

                background_image = cv2.warpAffine(color_heightmap_2x, rot, (320, 320))
                   
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if(rotate_idx == 1):
                    prediction_vis = prediction_vis[::-1,:,:]

                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas
