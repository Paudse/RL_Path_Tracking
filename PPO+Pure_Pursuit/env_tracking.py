"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
# import cvxpy
import math
import numpy as np
import sys
sys.path.append("../CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 10.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.1  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True


class Car:
    """
    vehicle state class
    """

    def __init__(self):
        self.episode = 0
        self.sim_ave_reward = []
        self.sim_j = []
        self.sim_episode = []

        self.dl = 1.0  # course tick
        self.cx, self.cy, self.cyaw, self.ck = self.get_straight_course(self.dl)
        # self.cx, self.cy, self.cyaw, self.ck = self.get_straight_course2(self.dl)
        # self.cx, self.cy, self.cyaw, self.ck = self.get_straight_course3(self.dl)
        # self.cx, self.cy, self.cyaw, self.ck = self.get_forward_course(self.dl)
        # self.cx, self.cy, self.cyaw, self.ck = self.get_switch_back_course(self.dl)

        self.cyaw = self.smooth_yaw(self.cyaw)
        self.sp = self.calc_speed_profile(self.cx, self.cy, self.cyaw, TARGET_SPEED)
        self.goal = [self.cx[-1], self.cy[-1]]

    def n_action(self):
        self.A_DIM = 2
        return self.A_DIM

    def n_state(self):
        self.S_DIM = 5
        return self.S_DIM   

    def reset(self):
        self.sim_x = []
        self.sim_y = []
        self.sim_yaw = []
        self.sim_v = []
        self.sim_t = []
        self.sim_a = []
        self.sim_d = []
        self.sim_reward = []
        self.sim_ai = []
        self.sim_di = []
        self.sim_target_dis = []

        self.time = 0
        self.x = self.cx[0]
        self.y = self.cy[0]
        self.yaw = self.cyaw[0]
        self.v = 0.0
        self.reward = 0
        self.j = 0
        self.accumulated_reward = 0
        self.ave_reward = 0
        self.ai = 0 
        self.di = 0 
        self.target_dis = 0 
        self.target_angle = 0 


        self.target_ind, target_mind = self.calc_nearest_index(self.cx, self.cy, self.cyaw, 0)
        self.xref, self.target_ind, dref = self.calc_ref_trajectory(self.cx, self.cy, self.cyaw, self.ck, self.sp, self.dl, self.target_ind)
        x_target = self.xref[0,0]
        y_target = self.xref[1,0]
        target_dis, target_angle = self.target_dir(x_target, y_target)
        s = np.array([self.ai, self.di, self.v, target_dis, target_angle])
        return s

    def target_dir(self, x_target, y_target): # target direction
        x_diff = x_target - self.x
        y_diff = y_target - self.y

        # Restrict alpha and beta (angle differences) to the range
        # [-pi, pi] to prevent unstable behavior e.g. difference going
        # from 0 rad to 2*pi rad with slight turn

        target_dis = np.sqrt(x_diff**2 + y_diff**2) # target distance: distance between current position to next target point
        target_angle = (np.arctan2(y_diff, x_diff)- self.yaw + np.pi) % (2 * np.pi) - np.pi

        return target_dis, target_angle

    def step(self, action):

        self.ai = action[0] # acceleration
        self.di = action[1] # delta_steer

        self.sim_x.append(self.x)
        self.sim_y.append(self.y)
        self.sim_yaw.append(self.yaw)
        self.sim_v.append(self.v)
        self.sim_t.append(self.time)
        self.sim_a.append(self.ai)
        self.sim_d.append(self.di)
        self.sim_reward.append(self.reward)
        self.sim_ai.append(self.ai)
        self.sim_di.append(self.di)
        self.sim_target_dis.append(self.target_dis)

        # self.show_animation()

        self.update_state(self.ai, self.di)

        self.target_ind, target_mind = self.calc_nearest_index(self.cx, self.cy, self.cyaw, self.target_ind)
        self.xref, self.target_ind, dref = self.calc_ref_trajectory(self.cx, self.cy, self.cyaw, self.ck, self.sp, self.dl, self.target_ind)
        x_target = self.xref[0,3]
        y_target = self.xref[1,3]
        self.target_dis, self.target_angle = self.target_dir(x_target, y_target)

        total_ref_dis = 0
        ref_dis_list = []
        ref_angle_list = []
        for i in range(5):
            ref_dis, ref_angle = self.target_dir(self.xref[0,i], self.xref[1,i])
            ref_dis_list.append(ref_dis)
            ref_angle_list.append(ref_angle)
            total_ref_dis = total_ref_dis + ref_dis

        s_ = np.array([self.ai, self.di, self.v, self.target_dis, self.target_angle])

        self.time = round(self.time + DT,2)
        self.j += 1

        self.reward = self.get_reward(self.target_dis,total_ref_dis)
        self.accumulated_reward = self.accumulated_reward+self.reward
        if self.if_done():
            self.ave_reward = self.accumulated_reward/self.j
            self.episode += 1
            self.sim_ave_reward.append(self.ave_reward)
            self.sim_j.append(self.j)
            self.sim_episode.append(self.episode)

        return s_, self.reward

    def get_reward(self,target_dis,total_ref_dis):
        reward = 50*0.999**(target_dis)+50*0.8**(10*abs(TARGET_SPEED-self.v))
        # print(target_dis,self.v,reward)

        return reward

    def update_state(self, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        self.x = self.x + self.v * math.cos(self.yaw) * DT
        self.y = self.y + self.v * math.sin(self.yaw) * DT
        self.yaw = self.yaw + self.v / WB * math.tan(delta) * DT
        self.v = self.v + a * DT

        if self.v > MAX_SPEED:
            self.v = MAX_SPEED
        elif self.v < MIN_SPEED:
            self.v = MIN_SPEED


    def show_animation(self):  # pragma: no cover
        plt.figure(1)
        plt.cla()
        # if ox is not None:
        #     plt.plot(ox, oy, "xr", label="MPC")
        plt.plot(self.cx, self.cy, "-r", label="course")
        plt.plot(self.sim_x, self.sim_y, "ob", label="trajectory")
        plt.plot(self.xref[0, :], self.xref[1, :], "xk", label="xref")
        plt.plot(self.cx[self.target_ind], self.cy[self.target_ind], "xg", label="target")
        self.plot_car(self.x, self.y, self.yaw, steer=self.di)
        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(self.time, 2))
                  + ", speed[km/h]:" + str(round(self.v * 3.6, 2)))
        plt.pause(0.0001)

    def episode_plot(self):
        plt.figure(2)
        plt.cla()
        # plt.subplots()
        ax = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1)
        plt.plot(self.cx, self.cy, "-r", label="spline")
        plt.plot(self.sim_x, self.sim_y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        # plt.legend()

        # plt.subplots()

        ax = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
        plt.plot(self.sim_episode, self.sim_ave_reward, "-b", label="speed")
        plt.grid(True)
        plt.xlabel("episode")
        plt.ylabel("Average Reward")

        ax = plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=1)
        plt.plot(self.sim_t, self.sim_reward, "-g", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Instant Reward")

        ax = plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=1)
        plt.plot(self.sim_t, self.sim_ai, "-g", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")

        ax = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1)
        plt.plot(self.sim_t, self.sim_di, "-g", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Steering Angle")

        ax = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=1)
        plt.plot(self.sim_t, self.sim_v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.pause(0.0001)

    def pi_2_pi(self, angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi

        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi

        return angle

    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

        outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                            [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

        fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                             [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                         [-math.sin(steer), math.cos(steer)]])

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += WB
        fl_wheel[0, :] += WB

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
                 np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
                 np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
                 np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
                 np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(x, y, "*")

    def calc_nearest_index(self, cx, cy, cyaw, pind):

        dx = [self.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy = [self.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind)

        dxl = cx[ind] - self.x
        dyl = cy[ind] - self.y

        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def calc_ref_trajectory(self, cx, cy, cyaw, ck, sp, dl, pind):
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(T + 1):
            travel += abs(self.v) * DT
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref


    def check_goal(self, goal, tind, nind):

        # check goal
        dx = self.x - goal[0]
        dy = self.y - goal[1]
        d = math.sqrt(dx ** 2 + dy ** 2)

        isgoal = (d <= GOAL_DIS)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(self.v) <= STOP_SPEED)

        if isgoal and isstop:
            return True

        return False

    def if_done(self):
        ifdone = False
        # print(self.time,MAX_TIME)
        if self.time == MAX_TIME:
            ifdone = True
            print('Time is up...')
        if self.check_goal(self.goal, self.target_ind, len(self.cx)):
            ifdone = True
            print('Goal!')
        return ifdone

    def calc_speed_profile(self, cx, cy, cyaw, target_speed):

        speed_profile = [target_speed] * len(cx)
        direction = 1.0  # forward

        # Set stop point
        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            move_direction = math.atan2(dy, dx)

            if dx != 0.0 and dy != 0.0:
                dangle = abs(self.pi_2_pi(move_direction - cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

        speed_profile[-1] = 0.0

        return speed_profile


    def smooth_yaw(self, yaw):

        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw


    def get_straight_course(self, dl):
        ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)

        return cx, cy, cyaw, ck


    def get_straight_course2(self, dl):
        ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
        ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)

        return cx, cy, cyaw, ck


    def get_straight_course3(self, dl):
        ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
        ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)

        cyaw = [i - math.pi for i in cyaw]

        return cx, cy, cyaw, ck


    def get_forward_course(self, dl):
        ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
        ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)

        return cx, cy, cyaw, ck


    def get_switch_back_course(self, dl):
        ax = [0.0, 30.0, 6.0, 20.0, 35.0]
        ay = [0.0, 0.0, 20.0, 35.0, 20.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)
        ax = [35.0, 10.0, 0.0, 0.0]
        ay = [20.0, 30.0, 5.0, 0.0]
        cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)
        cyaw2 = [i - math.pi for i in cyaw2]
        cx.extend(cx2)
        cy.extend(cy2)
        cyaw.extend(cyaw2)
        ck.extend(ck2)

        return cx, cy, cyaw, ck
