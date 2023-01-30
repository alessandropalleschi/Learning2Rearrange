import numpy as np
from env import Environment
from trainer import Trainer
import argparse
import os
import torch
import cv2
import matplotlib.pyplot as plt
from memory import *
import utils
import time
from skimage.draw import line
from operator import itemgetter

BATCH_SIZE = 32
NUM_EPOCHS = 1


def relable_episode(episode,env):
    state_color_list = list(map(itemgetter('colormap'), Transition(*zip(*episode.memory)).state))
    state_id_list = list(map(itemgetter('idmap'), Transition(*zip(*episode.memory)).state))
    state_color_heightmap_new = env.relabel_image(state_color_list,state_id_list,env.id_list,env.new_class)    
    next_state_color_list = list(map(itemgetter('colormap'), Transition(*zip(*episode.memory)).next_state))
    next_state_id_list = list(map(itemgetter('idmap'), Transition(*zip(*episode.memory)).next_state))
    next_state_color_heightmap_new = env.relabel_image(next_state_color_list,next_state_id_list,env.id_list,env.new_class)    

    for (transitions,state_color,next_state_color) in zip(episode.memory,state_color_heightmap_new,next_state_color_heightmap_new):
        state = transitions.state
        state['colormap'] = state_color
        next_state = transitions.next_state
        next_state['colormap'] = next_state_color

        transitions = transitions._replace(state=state)
        transitions = transitions._replace(next_state=next_state)
    
    return episode      


def check_push(env, depth, pix_x, pix_y, angle):

    # object_cent = depth[pix_y,0:pix_x]
    # object_cent = np.argwhere( (object_cent>0.01) & (object_cent<0.15))
    # if(object_cent.size>1):
    #     feas = False
    # else:
    #     feas = True
    push_length = 0.04
    primitive_position_f = env.px_to_xy(pix_x,pix_y)
    primitive_position_f[1] = primitive_position_f[1]+push_length*np.sin(angle)
    depth_new = depth.copy()
    depth_new = depth_new[4:depth_new.shape[0]-4,4:depth_new.shape[1]]
    pixel_final = env.xyz_to_pix(primitive_position_f)
    end_px = min(pixel_final[0],depth_new.shape[0]-1)
    rr, cc = line(pix_y, pix_x, end_px, pix_x)
    local_region_push = depth_new[rr,pix_x:]
    local_region_push = local_region_push[local_region_push>0.02]
    if(local_region_push.size==0):
        local_region_push = 0
    else:
        local_region_push = len(local_region_push)
    feasibile = local_region_push>2
    return feasibile




def optimize_model(trainer):

    buffer = trainer.buffer
    if len(buffer) > BATCH_SIZE*NUM_EPOCHS:
        print("Training")
                # mini_batch, idxs, is_weights = trainer.prioritized_memory.sample(BATCH_SIZE)
                # mini_batch = np.array(mini_batch).transpose()
        for j in range(NUM_EPOCHS):
            transitions = buffer.sample(BATCH_SIZE)   
            batch = Transition(*zip(*transitions))
                    # #             i = sample_iteration
                    # batch_loss = 0
            errors = []
            losses_push = []
            losses_grasp = []
            losses = []
            for i in range(BATCH_SIZE):
                    #             if 1:
                    # color_state = color_log[sample_iteration][0]
                    #                 depth_state = depth_log[sample_iteration][0]
                    #                 color_next_state = next_color_log[sample_iteration][0]
                    #                 depth_next_state = next_depth_log[sample_iteration][0]
                        # color_state = mini_batch[0][i]['colormap']
                        # depth_state = mini_batch[0][i]['depthmap']
                        # color_next_state = mini_batch[4][i]['colormap']
                        # depth_next_state = mini_batch[4][i]['depthmap']
                        # primitive_action_sample = mini_batch[1][i]
                        # action_sample = mini_batch[2][i]
                        # reward = mini_batch[3][i]
                color_state = batch.state[i]['colormap']
                depth_state = batch.state[i]['depthmap']
                color_next_state = batch.next_state[i]['colormap']
                depth_next_state = batch.next_state[i]['depthmap']
                reward = batch.reward[i]
                primitive_action_sample = batch.primitive_action[i]
                action_sample = batch.action[i]
                done = batch.done[i]
                        # if primitive_action_sample == 'push':
                        #     push_pred = trainer.forward_push(color_state,depth_state,is_volatile = True, network='policy')
                        #     pred = push_pred[action_sample[0],action_sample[1],action_sample[2]]

                        # else:
                        #     place_pred = trainer.forward_pick_and_place(color_state,depth_state,is_volatile = True, network='policy')
                        #     pred = place_pred[action_sample[0],action_sample[1],action_sample[2]]

                    #                 # grasp_success_sample = batch.grasp_success[i]
                    #                 # best_pix_ind_grasp_sample = batch.best_pix_ind_grasp[i]
                        # reward = mini_batch[3][i]
                if done:
                    label = reward
                else:
                    with torch.no_grad():
                        push_new_sample, place_new_sample = trainer.forward(color_next_state,depth_next_state,is_volatile=True,network='policy')
                    if trainer.double:
                        with torch.no_grad():
                           push_pred_target, place_pred_target = trainer.forward(color_next_state,depth_next_state,is_volatile=True,network='target')

                        greedy_action = 'grasp' if np.max(place_new_sample) > np.max(push_new_sample) else 'push'
                        if greedy_action=='grasp':
                            index = np.unravel_index(np.argmax(place_new_sample),place_new_sample.shape)
                            future_reward_samp = place_pred_target[index]
                        else:
                            index = np.unravel_index(np.argmax(push_new_sample),push_new_sample.shape)
                            future_reward_samp = push_pred_target[index]
                    else:
                        future_reward_samp = max(np.max(place_new_sample), np.max(push_new_sample))

                    label = reward+trainer.future_reward_discount*future_reward_samp
                        
                        

                if primitive_action_sample == 'grasp':
                    loss_grasp = trainer.backprop(color_state,depth_state, primitive_action_sample, action_sample, label)
                    losses.append(loss_grasp)
                    loss_grasp.cpu().detach()
                elif primitive_action_sample == 'push':
                    loss_push = trainer.backprop(color_state,depth_state, primitive_action_sample, action_sample, label)

                    losses.append(loss_push)
                    loss_push.cpu().detach()

            trainer.optimizer_push.zero_grad()
            trainer.optimizer_pick.zero_grad()
            losses =   torch.stack(losses).mean()
            losses.backward()                    
            trainer.optimizer_push.step()
            trainer.optimizer_pick.step()

            print("Total loss: %f" %(losses.cpu().detach()))

        if trainer.double:
            trainer._soft_update_q_network_parameters(trainer.model_push_target,
                                                trainer.model_push,
                                                alpha=1e-3)
            trainer._soft_update_q_network_parameters(trainer.model_pick_target,
                                                trainer.model_pick,
                                                alpha=1e-3)   
                    # logger.save_backup_model(trainer.model_push, trainer.model_pick_and_place,trainer.model_push_target,trainer.model_pick_and_place_target, method)            

def train_short_memory(obs,next_obs,reward, action_sample, primitive_action_sample,done,trainer):


    color_state = obs['colormap']
    depth_state = obs['depthmap']
    color_next_state = next_obs['colormap']
    depth_next_state = next_obs['depthmap']
                    #                 # grasp_success_sample = batch.grasp_success[i]
                    #                 # best_pix_ind_grasp_sample = batch.best_pix_ind_grasp[i]
                        # reward = mini_batch[3][i]
    if done:
        label = reward
    else:
        with torch.no_grad():
            push_new_sample, place_new_sample = trainer.forward(color_next_state,depth_next_state,is_volatile=True,network='policy')
        if trainer.double:
            with torch.no_grad():
                push_pred_target, place_pred_target = trainer.forward(color_next_state,depth_next_state,is_volatile=True,network='target')

            greedy_action = 'grasp' if np.max(place_new_sample) > np.max(push_new_sample) else 'push'
            if greedy_action=='grasp':
                index = np.unravel_index(np.argmax(place_new_sample),place_new_sample.shape)
                future_reward_samp = place_pred_target[index]
            else:
                index = np.unravel_index(np.argmax(push_new_sample),push_new_sample.shape)
                future_reward_samp = push_pred_target[index]
        else:
            future_reward_samp = max(np.max(place_new_sample), np.max(push_new_sample))
                                # print(future_reward_samp)

        label = reward+trainer.future_reward_discount*future_reward_samp

    trainer.optimizer_push.zero_grad()
    trainer.optimizer_pick.zero_grad()
    loss = trainer.backprop(color_state,depth_state, primitive_action_sample, action_sample, label)
                            #     trainer.prioritized_memory.update(idx, errors[i])
                            # if not losses_push:
                            #     losses_push.append(torch.tensor(0.).float().cuda())
                            # if not losses_grasp:
                            #     losses_grasp.append(torch.tensor(0.).float().cuda())
    loss.backward()                    
    trainer.optimizer_push.step()
    trainer.optimizer_pick.step()
    print("Short Memory loss: %f" %(loss.cpu().detach()))


def get_freq(x, exclude):
    values,counts = np.unique(x[x!=0],return_counts= True)
    # print(values)
   
    if counts.size == 0:
        return exclude
    else:
        iz = np.argsort(counts)
        return values[iz[-1]]

def select_placement(Yup,Ydow,freespace,no_obj=False):
    freespace[:,0:5] = 0
    freespace[:,-5:] = 0
    visual = np.zeros_like(freespace)
    ind = np.where(freespace>0)
    if (no_obj):
        dist_y = np.abs(ind[0]-freespace.shape[0] // 2) + 1/ind[1]
        pixel_ind = np.nanargmin(dist_y)
        return ind[1][pixel_ind], ind[0][pixel_ind]
    else: 
        mid = (Yup + Ydow) // 2 
        # if np.random.uniform() < 0.5:
        # print(1/ind[1])
        # time.sleep(2)
        dist_y = np.abs(ind[0]-mid) + ind[1]
        # else: 
        #     dist_y = np.abs(ind[0]-mid)
        # if np.random.uniform() < 0.5:

        #     pixels = np.where(dist_y<20)[0]
        #     # print(pixels)
        #     if len(pixels)>0:
        #         pixel_ind = np.random.choice(pixels)
        #     else:
        #         pixel_ind = np.nanargmin(dist_y)
        visual[ind] = 1/dist_y
        # else: 
        try:
            pixel_ind = np.nanargmin(dist_y)
        except ValueError:
            print(ind)

        return ind[1][pixel_ind], ind[0][pixel_ind]



def get_place_pose(env,obj_class,freespace):
    Y_down, Y_up, _ = env.add_target_pixels()
    if obj_class==1:
        return select_placement(Y_up[0],Y_down[0],freespace)
    if obj_class==5:
        return select_placement(Y_up[1],Y_down[1],freespace)
    if obj_class==3:
        return select_placement(Y_up[2],Y_down[2],freespace)

def check_feasibility_grasp(classmap, pix_x, pix_y):

    class_obj = get_object_class(classmap, pix_x, pix_y)

    return class_obj!=0

def get_object_class(classmap, pix_x, pix_y):
    
    return classmap[pix_y,pix_x]

def select_action(trainer,env,obs_old,push_predictions,grasp_predictions,explore_prob,is_testing):

    push_predictions = trainer.preprocessQmaps(push_predictions,obs_old['depthmap'],'push')
    grasp_predictions = trainer.preprocessQmaps(grasp_predictions,obs_old['depthmap'],'grasp')

    obs = obs_old.copy()
    best_push_conf = np.max(push_predictions)
    best_grasp_conf = np.max(grasp_predictions)
    primitive_action = 'grasp'

    explore_actions = False
    if best_push_conf > best_grasp_conf:
        primitive_action = 'push'

    explore_actions = np.random.uniform() < explore_prob and not is_testing
    if explore_actions:
        if np.random.randint(0,2) == 1:
            primitive_action = 'push' if np.random.randint(0,2) == 0 else 'grasp'
    if primitive_action=='push':

        if explore_actions:
            while True:
                if np.random.uniform(0,1) < 0.8:
                    indices = np.where(push_predictions.flatten() != -1000)[0]
                                                
                    best_px = np.unravel_index(np.random.choice(indices),push_predictions.shape)
                    if check_push(env,obs['depthmap'], best_px[2],best_px[1],np.deg2rad(best_px[0]*(360.0/trainer.model_push.nr_rotations)-90)):
                        break
                else:
                    indices = np.where(push_predictions.flatten() != -2000)[0]
                    best_px = np.unravel_index(np.random.choice(indices),push_predictions.shape)

        else:
            best_px = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        predicted_value = push_predictions[best_px[0],best_px[1],best_px[2]]
        best_px_x = best_px[2]
        best_px_y = best_px[1]
        best_rotation = np.deg2rad(best_px[0]*(360.0/trainer.model_push.nr_rotations)-90)
    elif primitive_action=='grasp':
        if explore_actions:
            if np.random.uniform(0,1) < 0.8:
                indices = np.where(grasp_predictions.flatten() != -1000)[0]
            else: 
                indices = np.where(grasp_predictions.flatten() != -2000)[0]
                            
            best_px = np.unravel_index(np.random.choice(indices),grasp_predictions.shape)
        else:
            best_px = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        
        predicted_value = grasp_predictions[best_px[0],best_px[1],best_px[2]]
        best_px_x = best_px[2]
        best_px_y = best_px[1]
        best_rotation = np.deg2rad(best_px[0]*(360.0/trainer.model_push.nr_rotations)-90)

    primitive_position = env.px_to_xy(best_px_x,best_px_y)
    success = False
    classmap = obs['classmap']

    if primitive_action == 'push' and check_push(env,obs['depthmap'], best_px[2],best_px[1],best_rotation):
        success = env.push(primitive_position, best_rotation)
        push_pred_vis = trainer.get_prediction_vis(push_predictions,obs['colormap'], best_px, best_px)
        cv2.imwrite('visualization.push.png', push_pred_vis)

    elif primitive_action=='grasp' and check_feasibility_grasp(classmap,best_px_x,best_px_y):
        success = env.pick(primitive_position)
        if success:
            class_pick = get_object_class(classmap, best_px_x, best_px_y)
            obs_after_pick = env.get_observation()
            freespace = obs_after_pick['freespace']
            px_place, py_place = get_place_pose(env,class_pick,freespace)
            primitive_position = env.px_to_xy(px_place,py_place)
            success = env.place(primitive_position)
            grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, obs['colormap'], best_px,[py_place,px_place])
            cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

    
    
    return primitive_action, best_px, success, predicted_value

def main(args):


    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    random_seed = args.random_seed

    # ------------- Algorithm options -------------
    # heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?

    # -------------- Testing options --------------
    is_testing = args.is_testing
    training_iterations = 0
    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_push = os.path.abspath(args.snapshot_push)  if load_snapshot else None
    snapshot_pick = os.path.abspath(args.snapshot_pick)  if load_snapshot else None
    snapshot_push_target = os.path.abspath(args.snapshot_push_target)  if load_snapshot else None
    snapshot_pick_target = os.path.abspath(args.snapshot_pick_target)  if load_snapshot else None

    logging_directory = os.path.join(os.path.abspath('logs'),args.logging_directory)
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


    # Set random seed
    np.random.seed(random_seed)
    
    # Initialize pick-and-place system (camera and robot)
    env = Environment(gui=True,num_obj=[2])
    while True:
        if(env.reset()):
            if(env.check_sorted()<1):
                no_change_count = 0
                relable = True

                break
    prev_sort_status = env.check_sort_status()
    # Initialize trainer
    most_recent_success_rate = deque(maxlen=50)   
    most_recent_scores = deque(maxlen=50)
    reward = 0
    trainer = Trainer(is_testing=is_testing, load_snapshot=load_snapshot, snapshot_push=snapshot_push, snapshot_pick=snapshot_pick, snapshot_pick_target=snapshot_pick_target, snapshot_push_target=snapshot_push_target)
    
    while True:
        if not env.check_sim() or no_change_count>39 or env.check_sorted()==1:
    
            print("Restoring Simulation")
            print("Restoring Simulation")
            reward = reward if env.check_sim() else -100
            print("Episode Reward:", reward)
            most_recent_scores.append(reward)
            most_recent_success_rate.append(1 if (env.check_sorted()==1 and env.check_sim()) else 0)
            average_success = sum(most_recent_success_rate) / len(most_recent_success_rate)
            average_score = sum(most_recent_scores) / len(most_recent_scores)
            print("Average Reward", average_score)
            print("Average Success Rate", average_success)
            rewward = 0
            while True:
                if(env.reset()):
                    if(env.check_sorted()<1):
                        no_change_count = 0
                        relable = True
                        trainer.episode.memory.clear()
                        break
                continue



        
        
        explore_prob = max(0.5 * np.power(0.9998, trainer.iteration),0.1)

        print('Observing Scene')
        obs = env.get_observation()
        prev_relable = env.check_relabel() == 1
        if prev_relable:
            prev_new_class = env.new_class
        color_heightmap = obs['colormap']
        depth_heightmap = obs['depthmap']
        color_hist = color_heightmap.copy()
        depth_hist = depth_heightmap.copy()

        with torch.no_grad():
            push_predictions, grasp_predictions = trainer.forward(color_hist, depth_hist, is_volatile=True)
        primitive_action, best_index, success, predicted_value = select_action(trainer=trainer,env=env,obs_old=obs,push_predictions=push_predictions,grasp_predictions=grasp_predictions,explore_prob=explore_prob,is_testing=is_testing)


        env.wait_static()
        done = env.check_sorted()==1
        ok_status = env.check_sim()
        reward_value = -1
        if not done and ok_status:
            next_obs = env.get_observation()
        else:
            next_obs = obs

        change_detected,diff = utils.check_env_depth_change(obs['depth'],next_obs['depth'])

        if not done:
            reward_value = reward_value if change_detected else -5

        if done and ok_status:
            reward_value = 0
        reward_value = reward_value if ok_status else -50
        reward+=reward_value
        done_episode = done or (not ok_status)
        train_short_memory(obs,next_obs,reward_value, best_index, primitive_action,done_episode,trainer)
        trainer.buffer.push(obs.copy(), best_index, next_obs.copy(), reward_value, primitive_action,done_episode)
        reward_relable = 0 if env.check_relabel()==1 else reward_value
        if relable and reward_value==-1 and env.check_relabel()==1:
            if prev_relable:
                if prev_new_class != env.new_class:

                    done_episode_relabel = True
                    trainer.episode.push(obs, best_index, next_obs, reward_relable, primitive_action, done_episode_relabel)
                    print("Relabeling")
                    trainer.episode = relable_episode(trainer.episode,env)
                    trainer.buffer.memory+=trainer.episode.memory
                    trainer.episode.memory.clear()
            else:
                done_episode_relabel = True
                trainer.episode.push(obs, best_index, next_obs, reward_relable, primitive_action, done_episode_relabel)
                print("Relabeling")
                trainer.episode = relable_episode(trainer.episode,env)
                trainer.buffer.memory+=trainer.episode.memory
                trainer.episode.memory.clear()

        optimize_model(trainer=trainer)
        trainer.iteration +=1

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True,                                    help='run in simulation?')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=13953,                      help='random seed for simulation and neural net initialization')

    # ------------- Algorithm options -------------
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=True)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_push', dest='snapshot_push', action='store', default='/scr1/ObjectSorting/logs/train_resnet_123/models/push_snapshot-001400.reinforcement.pth')
    parser.add_argument('--snapshot_pick', dest='snapshot_pick', action='store', default='/scr1/ObjectSorting/logs/train_resnet_123/models/pick_and_place_snapshot-001400.reinforcement.pth')
    parser.add_argument('--snapshot_push_target', dest='snapshot_push_target', action='store', default='/scr1/ObjectSorting/logs/train_resnet_123/models/target_push_snapshot-001400.reinforcement.pth')
    parser.add_argument('--snapshot_pick_target', dest='snapshot_pick_target', action='store', default='/scr1/ObjectSorting/logs/train_resnet_123/models/target_pick_and_place_snapshot-001400.reinforcement.pth')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store', default='train_resnet_123_test')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)