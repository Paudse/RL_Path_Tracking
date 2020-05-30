"""

Path tracking simulation with pure pursuit steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

"""
import numpy as np
import math
import matplotlib.pyplot as plt


k = 1/2 # look forward gain 0.1
Lfc = 2  # look-ahead distance 2, 20
Kp = 2  # speed proportional gain 1.0
Ki = 1
dt = 0.1  # [s]
L = 2.9  # [m] wheel base of vehicle


old_nearest_point_index = None


def a_PIDControl(target, current):
    if current<50:
        accumulated_error = 0
    else:
        pass
        # accumulated_error += error

    error = target - current
    # accumulated_error += error

    # a = Kp * error + Ki * accumulated_error
    a = Kp * error

    return a

def d_PIDControl(target, current):
    # if current<50:
    #     accumulated_error = 0
    # else:
    #     accumulated_error += error

    error = target - current
    # accumulated_error += error

    # a = Kp * error + Ki * accumulated_error
    a = 0.7 * error
    # print(a)

    return a

def pure_pursuit_control(state, cx, cy, pind, ind_reset):

    ind = calc_target_index(state, cx, cy, ind_reset)

    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    Lf = k * state.v + Lfc

    # delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)
    delta = math.atan2(2 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind

def calc_distance(state, point_x, point_y):

    dx = state.rear_x - point_x
    dy = state.rear_y - point_y

    return math.sqrt(dx ** 2 + dy ** 2)

def nearest_point_distance(state, cx, cy):
    dx = [state.rear_x - icx for icx in cx]
    dy = [state.rear_y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    return min(d)

def calc_target_index(state, cx, cy, ind_reset):

    global old_nearest_point_index

    if old_nearest_point_index is None:
        # search nearest point index
        dx = [state.rear_x - icx for icx in cx]
        dy = [state.rear_y - icy for icy in cy]
        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
        ind = d.index(min(d))
        nearest_distance = min(d)
        old_nearest_point_index = ind
    else:
        ind = old_nearest_point_index
        distance_this_index = calc_distance(state, cx[ind], cy[ind])
        while True:
            ind = ind + 1 if (ind + 1) < len(cx) else ind
            distance_next_index = calc_distance(state, cx[ind], cy[ind])
            if distance_this_index < distance_next_index:
                break
            distance_this_index = distance_next_index
        old_nearest_point_index = ind

    L = 0.0

    Lf = k * (state.vx*3.6-10) + Lfc
    # Lf = Lfc

    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind] - state.rear_x
        dy = cy[ind] - state.rear_y
        L = math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    if ind_reset == 1:
        ind = 0
        old_nearest_point_index = None

    return ind
