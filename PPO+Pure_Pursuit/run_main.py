import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import matplotlib.pyplot as plt
from pure_pursuit import a_PIDControl, d_PIDControl, pure_pursuit_control, calc_target_index, nearest_point_distance
from vehicle_update import update, kinematic_update, dynamic_update, update_cv, dynamic_update_cv
from plot_animation import show_step_animation, show_episode_plot, save_plot, save_sim, strategy_3d_map, plot_action
from PPO import PPO
import time
import csv
import sys
import os
sys.path.append("CubicSpline")

try:
    import cubic_spline_planner
except:
    raise


# k = 0.1  # look forward gain
# Lfc = 2.0  # look-ahead distance
# Kp = 1.0  # speed proportional gain
# Ki = 0.1
dt = 0.1  # [s]
L = 2.9  # [m] wheel base of vehicle
max_reward = 0

# old_nearest_point_index = None
# show_animation = True

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((L / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((L / 2) * math.sin(self.yaw))

        self.vx = 0.0001
        self.vy = 0
        self.omega = 0

def target_course(v,rc,it):
    cx = np.arange(0, 1000, 0.1)
    cy = [-25+math.cos(ix / rc) * 25 for ix in cx] # 0.2, 0.1, 0.05 / 5 10 20
    if it == 1:
        cy = [-iy for iy in cy]
    else:
        pass
    cv = np.ones(int(len(cx)))*v/3.6 # [m/s]
    cyaw = 0

    return cx, cy, cv, cyaw

def target_dis_dir(x_current, y_current, yaw_current, x_target, y_target): # calculate target distance and direction
    x_diff = x_target - x_current
    y_diff = y_target - y_current

    # Restrict alpha and beta (angle differences) to the range
    # [-pi, pi] to prevent unstable behavior e.g. difference going
    # from 0 rad to 2*pi rad with slight turn

    target_dis = np.sqrt(x_diff**2 + y_diff**2) # target distance: distance between current position to next target point
    target_angle = (np.arctan2(y_diff, x_diff)- yaw_current + np.pi) % (2 * np.pi) - np.pi
    if target_angle < -3.14:
        target_angle = 0
    # print(target_angle)

    return target_dis, target_angle

def control_blender(a_PID, d_pure_pursuit, a_RL, d_RL,blender):
    sigma_a = 0.0 # 0.2
    sigma_d = blender # 0.5
    a_RL_correction = sigma_a*a_RL
    d_RL_correction = sigma_d*d_RL
    a_final = a_PID + a_RL_correction
    d_final = d_pure_pursuit + d_RL_correction
    # print('d_final',d_final)
    return a_final, d_final, a_RL_correction, d_RL_correction

def reward_function(state, cx, cy, a_PID, d_pure_pursuit, a_RL, d_RL, target_dis, target_angle, target_speed, state_v):
    dis_nor = nearest_point_distance(state, cx, cy) 
    # print(dis_nor)
    a_nor = abs(a_PID-a_RL)/1
    d_nor = abs(d_pure_pursuit-d_RL)/1
    # dis_nor = abs(target_dis)/1
    ang_nor =  abs(target_angle)/1
    spd_nor = abs(target_speed-state_v)/1

    # instant_reward = 20*0.5**a_nor + 20*0.5**d_nor 
    # instant_reward = 5*0.5**a_nor + 5*0.5**d_nor + 30*0.3**dis_nor + 30*0.5**ang_nor + 30*0.5**spd_nor
    # instant_reward = 20*0.5**a_nor + 20*0.5**d_nor + 20*0.8**dis_nor + 20*0.8**ang_nor + 20*0.8**spd_nor
    # instant_reward = 40*0.8**dis_nor + 20*0.8**ang_nor + 40*0.8**spd_nor
    # instant_reward = 10*0.5**a_nor + 10*0.5**d_nor + 50*0.8**dis_nor + 20*0.8**ang_nor + 10*0.8**spd_nor
    # instant_reward = 5*0.5**a_nor + 5*0.5**d_nor + 80*0.95**dis_nor + 5*0.8**ang_nor + 5*0.8**spd_nor
    instant_reward = 80*0.8**dis_nor + 20*0.8**ang_nor 

    return instant_reward


def train(train_epi, v, rc, it, blender):
    global max_reward

    history = {'episode': [], 'Episode_reward': []}
    #  target course
    cx, cy, cv, cyaw = target_course(v,rc,it)

    T = 3500/v  # max simulation time 50

    total_episode = 20
    sim_episode = []
    sim_average_reward = []
    sim_a_RL_percentage = []
    sim_d_RL_percentage = []

    for i in range(total_episode):

        # initial state
        state = State(x=-0.0, y=0.0, yaw=0.0, v=0.0)

        lastIndex = len(cx) - 1
        time = 0.0
        sim_t = []
        sim_x = []
        sim_y = []
        sim_yaw = []
        sim_v = []
        sim_a_PID = []
        sim_d_PID = []
        sim_d_pure_pursuit = []
        sim_a_final = []
        sim_d_final = []
        sim_a_RL = []
        sim_d_RL = []
        sim_a_RL_correction = []
        sim_d_RL_correction = []
        sim_a_RL_or_not = []
        sim_d_RL_or_not = []
        sim_reward = []
        sim_target_dis = []
        sim_target_angle = []
        d_PID = 0
        a_final = 0
        d_final = 0
        target_speed = cv[0]

        # observation = np.array([target_dis, target_angle, target_speed, state.v, a_final, d_final])

        target_ind = calc_target_index(state, cx, cy, 1)
        target_dis, target_angle = target_dis_dir(state.x, state.y, state.yaw, cx[target_ind], cy[target_ind])

        # observation = np.array([target_dis, target_angle, target_speed, state.v, a_final, d_final])
        # observation = np.array([target_angle])
        observation = np.array([target_angle, target_speed])
        # print(observation)
        states, actions, rewards = [], [], []
        episode_reward = 0
        j = 0

        while T >= time and lastIndex > target_ind:
            # print('observation',observation)

            target_speed = cv[target_ind]

            # a_PID = a_PIDControl(target_speed, state.vx)
            a_PID = a_PIDControl(target_speed, state.v)
            # print(target_speed, state.v)
            # if abs(a_PID) > 5:
            #     a_PID = np.sign(a_PID)*5

            # a_PID = PIDControl(target_speed, state.v)
            d_pure_pursuit, target_ind = pure_pursuit_control(state, cx, cy, target_ind, 0)
            # print(d_pure_pursuit)
            d_PID = d_PIDControl(target_angle,d_PID)
            # print(target_angle)


            action_RL = model.choose_action(observation)
            # print(observation, action_RL)
            a_RL = 0 # acceleration
            d_RL = action_RL[0] # delta_steer
            if math.isnan(d_RL):
                # print('observation',observation)
                # print('d_RL is none')
                d_RL = 0
                j=999999999
                break
                # continue
            # print('d_RL',d_RL)

            # Control command mechanism selection
            # switch
            # a_final, d_final, a_RL_or_not, d_RL_or_not = control_switch(a_PID, d_pure_pursuit, a_RL, d_RL)
            # adjust
            # a_final, d_final = control_blender(a_PID, d_pure_pursuit, a_RL, d_RL)
            # pure ppo
            # a_final, d_final = a_RL, d_RL
            # pure conventional
            # a_final, d_final = a_PID, d_pure_pursuit
            # a_final, d_final, a_RL_correction, d_RL_correction = control_blender(a_PID, d_PID, a_RL, d_RL)
            a_final, d_final, a_RL_correction, d_RL_correction = control_blender(a_PID, d_pure_pursuit, a_RL, d_RL, blender)


            # state = update(state, a_final, d_final)
            # state = update_cv(state, target_speed, d_final)
            state = dynamic_update_cv(state, target_speed, d_final)
            # state = update(state, a_RL, d_RL)
            # state = kinematic_update(state, a_final, d_final)
            # state = dynamic_update(state, a_PID, d_pure_pursuit)
            # state = dynamic_update(state, a_final, d_final)

            target_dis, target_angle = target_dis_dir(state.x, state.y, state.yaw, cx[target_ind], cy[target_ind])
            # print('target_dis, target_angle',target_dis, target_angle)

            # next_observation = np.array([target_dis/50, target_angle, target_speed/50, state.v/50, a_final/10, d_final/20])
            next_observation = np.array([target_angle, target_speed])
            reward = reward_function(state, cx, cy, a_PID, d_pure_pursuit, a_RL, d_RL, target_dis, target_angle, target_speed, state.v)
            episode_reward += reward

            states.append(observation)
            actions.append(action_RL)
            rewards.append(reward)

            observation = next_observation
            # print(observation)

            sim_t.append(time)
            sim_x.append(state.x)
            sim_y.append(state.y)
            sim_yaw.append(state.yaw)
            sim_v.append(state.v)
            sim_a_PID.append(a_PID)
            sim_d_PID.append(d_PID)
            sim_d_pure_pursuit.append(d_pure_pursuit)
            sim_a_RL.append(a_RL)
            sim_d_RL.append(d_RL)
            sim_a_RL_correction.append(a_RL_correction)
            sim_d_RL_correction.append(d_RL_correction)
            sim_a_final.append(a_final)
            sim_d_final.append(d_final)
            sim_target_dis.append(target_dis)
            sim_target_angle.append(target_angle)
            # sim_a_RL_or_not.append(a_RL_or_not)
            # sim_d_RL_or_not.append(d_RL_or_not)
            sim_reward.append(reward)
            # show_step_animation(cx, cy, sim_x, sim_y, sim_yaw, state.x, state.y, state.yaw, state.v, target_ind)
            # w = model.get_weights(i)

            if (j + 1) % model.batch == 0:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = model.discount_reward(states, rewards, next_observation)
                    # d_reward = [reward]
                    # print('states',states,'actions',actions,'rewards',rewards)
                    # print(d_reward)

                    model.update(states, actions, d_reward, j)
                    # w = model.get_weights(i)

                    states, actions, rewards = [], [], []


            j += 1
            time = time + dt

        # print(len(sim_d_RL_or_not))
        average_reward = episode_reward / j
        sim_episode.append(i)
        sim_average_reward.append(average_reward)
        # sim_a_RL_percentage.append(np.sum(sim_a_RL_or_not)/len(sim_a_RL_or_not)*100)
        # sim_d_RL_percentage.append(np.sum(sim_d_RL_or_not)/len(sim_d_RL_or_not)*100)

        # history['episode'].append(i)
        # history['Episode_reward'].append(episode_reward)
        # print('Episode: {} | Episode reward: {:.2f}'.format(i, average_reward))
        # model.save_history(history, 'ppo2.csv')

        # w = model.get_weights(i)
        # print(w)


        if train_epi == 0:
            if i == 0:
                max_reward = average_reward
                print('max_reward:',round(max_reward,2))
                save_io = 0
                # save_plot(i,average_reward)
                save_sim(i,average_reward,cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
                show_episode_plot(cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, sim_a_RL_or_not, sim_d_RL_or_not, sim_a_RL_percentage, sim_d_RL_percentage, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
                plot_action(model, save_io,i,average_reward, target_speed)
                strategy_3d_map(model)
            if average_reward > max_reward:
                save_io = 1
                max_reward = average_reward
                print('Better! Save Model! max_reward:',round(max_reward,2))
                model.save_learning()
                show_episode_plot(cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, sim_a_RL_or_not, sim_d_RL_or_not, sim_a_RL_percentage, sim_d_RL_percentage, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
                plot_action(model, save_io,i,average_reward, target_speed)
                strategy_3d_map(model)
        else:
            if average_reward > max_reward: 
                save_io = 1
                max_reward = average_reward
                print('Better! Save Model! max_reward:',round(max_reward,2))
                # print('Better! Save Model!')
                model.save_learning()
                # save_plot(i,average_reward)
                save_sim(i,average_reward,cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
                show_episode_plot(cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, sim_a_RL_or_not, sim_d_RL_or_not, sim_a_RL_percentage, sim_d_RL_percentage, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
                plot_action(model, save_io,i,average_reward, target_speed)
                strategy_3d_map(model)
            else:
            	save_io = 0

        # show_episode_plot(cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
        #     sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
        #     sim_reward, sim_episode, sim_average_reward, sim_a_RL_or_not, sim_d_RL_or_not, sim_a_RL_percentage, sim_d_RL_percentage, \
        #     sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)
        # plot_action(model, save_io,i,average_reward, target_speed)
        # strategy_3d_map(model)
        
        
        # save_sim(i,average_reward,cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
        # sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
        # sim_reward, sim_episode, sim_average_reward, \
        # sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle)

if __name__ == '__main__':
    model = PPO(50, 'ppo2') # 32
    # train()
    it = 0 # 1/0, inverse track, 1 inverse
    blender = 1 # 1/0, 1 with PPO

    for v in range(30,60,15): # 15, 30, 45
        for rc in range(20,10,-20): # rc=20, r=16, 60 40 20
            # global max_reward 
            max_reward = 20
            print('train for (v,rc) = (', v,',', rc, ')')
            for train_epi in range(1000):
                print('train_epi:',train_epi)
                # print(max_reward)
                train(train_epi, v, rc, it, blender)
                model.load_weights()





