import math

dt = 0.1  # [s]
L = 2.9  # [m] wheel base of vehicle 5
Lr = L / 2.0  # [m]
Lf = L - Lr

Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg
# m = 15000.0  # kg

def update_cv(state, v, delta):

    # state.v = v

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    # state.v = state.v + a * dt
    state.v = v
    
    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))
    # print(state.v)

    return state

def dynamic_update_cv(state, v, delta):
    state.x = state.x + state.vx * math.cos(state.yaw) * dt - state.vy * math.sin(state.yaw) * dt
    state.y = state.y + state.vx * math.sin(state.yaw) * dt + state.vy * math.cos(state.yaw) * dt
    state.yaw = state.yaw + state.omega * dt

    Ffy = -Cf * math.atan2(((state.vy + Lf * state.omega) / state.vx - delta), 1.0)
    Fry = -Cr * math.atan2((state.vy - Lr * state.omega) / state.vx, 1.0)

    # state.vx = state.vx + (a - Ffy * math.sin(delta) / m + state.vy * state.omega) * dt
    state.vx = v
    state.vy = state.vy + (Fry / m + Ffy * math.cos(delta) / m - state.vx * state.omega) * dt
    # print('state.vx,state.vy,state.yaw,Ffy,Fry',state.vx,state.vy,state.yaw,Ffy,Fry)

    state.v = math.sqrt(state.vx ** 2 + state.vy ** 2)
    state.omega = state.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))

    return state

def update(state, a, delta):

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt
    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))
    # print(state.v)

    return state


def kinematic_update(state, a, delta):

    state.beta = math.atan2(Lr / L * math.tan(delta), 1.0)

    state.x = state.x + state.v * math.cos(state.yaw + state.beta) * dt
    state.y = state.y + state.v * math.sin(state.yaw + state.beta) * dt
    state.yaw = state.yaw + state.v / Lr * math.sin(state.beta) * dt
    state.v = state.v + a * dt
    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))

    return state


def dynamic_update(state, a, delta):
    state.x = state.x + state.vx * math.cos(state.yaw) * dt - state.vy * math.sin(state.yaw) * dt
    state.y = state.y + state.vx * math.sin(state.yaw) * dt + state.vy * math.cos(state.yaw) * dt
    state.yaw = state.yaw + state.omega * dt

    Ffy = -Cf * math.atan2(((state.vy + Lf * state.omega) / state.vx - delta), 1.0)
    Fry = -Cr * math.atan2((state.vy - Lr * state.omega) / state.vx, 1.0)

    state.vx = state.vx + (a - Ffy * math.sin(delta) / m + state.vy * state.omega) * dt
    state.vy = state.vy + (Fry / m + Ffy * math.cos(delta) / m - state.vx * state.omega) * dt
    # print('state.vx,state.vy,state.yaw,Ffy,Fry',state.vx,state.vy,state.yaw,Ffy,Fry)

    state.v = math.sqrt(state.vx ** 2 + state.vy ** 2)
    state.omega = state.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))

    return state

def dynamic_update_beta(state, a, delta):
    state.x = state.x + state.vx * math.cos(state.yaw) * dt - state.vy * math.sin(state.yaw) * dt
    state.y = state.y + state.vx * math.sin(state.yaw) * dt + state.vy * math.cos(state.yaw) * dt
    state.yaw = state.yaw + state.omega * dt

    beta = state.vy/state.vx
    alpha_f = -math.atan2(beta + (Lf * state.omega) / state.vx - delta, 1.0)
    alpha_r = -math.atan2( beta - (Lr * state.omega) / state.vx, 1.0)
    Ffy = Cf * alpha_f
    Fry = Cr * alpha_r

    state.vx = state.vx + (a - Ffy * math.sin(delta) / m + state.vy * state.omega) * dt
    state.vy = state.vy + (Fry / m + Ffy * math.cos(delta) / m - state.vx * state.omega) * dt

    state.v = math.sqrt(state.vx ** 2 + state.vy ** 2)
    state.omega = state.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))

    return state

def vehicle_tire_dynamics_update(state, a, delta):
    # T_w = driving torque on each wheel
    # delta = steering angle of front wheel in rad

    state.x = state.x + state.vx * math.cos(state.yaw) * dt - state.vy * math.sin(state.yaw) * dt
    state.y = state.y + state.vx * math.sin(state.yaw) * dt + state.vy * math.cos(state.yaw) * dt
    state.yaw = state.yaw + state.omega * dt

    i_rl, i_rr = tire_slip(state, w_rl, w_rr) # tire silp ratio, rear left, rear right
    trc_rl, trc_rr = tire_traction(i_rl, i_rr)

    w_rl, w_rr = wheel_dynamics(T_w, trc_rl, trc_rr)
    # resistance = # longitudinal driving resistance
    # a = # vehicle longitudinal acceleration

    Ffy = -Cf * math.atan2(((state.vy + Lf * state.omega) / state.vx - delta), 1.0)
    Fry = -Cr * math.atan2((state.vy - Lr * state.omega) / state.vx, 1.0)

    state.vx = state.vx + (a - Ffy * math.sin(delta) / m + state.vy * state.omega) * dt
    state.vy = state.vy + (Fry / m + Ffy * math.cos(delta) / m - state.vx * state.omega) * dt

    state.v = math.sqrt(state.vx ** 2 + state.vy ** 2)
    state.omega = state.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))

    return state


def wheel_dynamics(T_w, trc_rl, trc_rr):
    alpha_rl = (T_w- trc_rl)/I_w
    alpha_rr = (T_w- trc_rr)/I_w
    w_rl = w_rl + lpha_rl*dt
    w_rr = w_rr + lpha_rr*dt
    return w_rl, w_rr

def tire_slip(state, w_rl, w_rr):

    return i_rl, i_rr


def tire_traction():
    fn_rl, fn_rr = tire_nornal_force()
    trc_rl, trc_rr = magic_formula()
    return trc_rl, trc_rr

def tire_nornal_force():

    return fn_rl, fn_rr

def magic_formula():
    return

