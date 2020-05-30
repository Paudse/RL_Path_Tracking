import matplotlib.pyplot as plt
import scipy.io 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io

# Load PID result
data = scipy.io.loadmat('PP_PPO_v30r20_0_reward-55.02')
cx = data['cx'][0]
cy = data['cy'][0]
cv = data['cv'][0]
sim_t = data['sim_t'][0]
sim_x = data['sim_x'][0]
sim_y = data['sim_y'][0]
sim_yaw = data['sim_yaw'][0]
sim_v = data['sim_v'][0]
sim_a_PID = data['sim_a_PID'][0]
sim_d_pure_pursuit = data['sim_d_pure_pursuit'][0]
sim_a_RL = data['sim_a_RL'][0]
sim_d_RL = data['sim_d_RL'][0]
sim_reward = data['sim_reward'][0]
sim_episode = data['sim_episode'][0]
sim_average_reward = data['sim_average_reward'][0]
# sim_a_RL_or_not = data['sim_a_RL_or_not'][0]
# sim_d_RL_or_not = data['sim_d_RL_or_not'][0]
# sim_a_RL_percentage = data['sim_a_RL_percentage'][0]
# sim_d_RL_percentage = data['sim_d_RL_percentage'][0]

data = scipy.io.loadmat('PP_v30_r20_0_reward-42.51')
cx_1 = data['cx'][0]
cy_1 = data['cy'][0]
cv_1 = data['cv'][0]
sim_t_1 = data['sim_t'][0]
sim_x_1 = data['sim_x'][0]
sim_y_1 = data['sim_y'][0]
sim_yaw_1 = data['sim_yaw'][0]
sim_v_1 = data['sim_v'][0]
sim_a_PID_1 = data['sim_a_PID'][0]
sim_d_pure_pursuit_1 = data['sim_d_pure_pursuit'][0]
sim_a_RL_1 = data['sim_a_RL'][0]
sim_d_RL_1 = data['sim_d_RL'][0]
sim_reward_1 = data['sim_reward'][0]
sim_episode_1 = data['sim_episode'][0]
sim_average_reward_1 = data['sim_average_reward'][0]

#### Fig1 ################################################################################################################################
fig1 = plt.figure(figsize=(7,3), dpi=300)
# plt.subplots_adjust(wspace =0.5, hspace =0.5)

##########################################
# ax = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1)
# plt.plot(cx, cy, "-r", label="spline")
# plt.plot(sim_x, sim_y, "-g", label="tracking")
# plt.grid(True)
# plt.axis("equal")
# plt.xlabel("x[m]")
# plt.ylabel("y[m]")
# plt.legend(loc='upper left')

# ax = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
plt.plot(cx[0:2500], cy[0:2500], "-r", label="reference",linewidth=2)
# plt.plot(sim_x[0:380], sim_y[0:380], "-g", label="Pure-Pursuit+PPO")
plt.plot(sim_x_1[0:380], sim_y_1[0:380], "--k", label="Pure-Pursuit",linewidth=2)
plt.plot(sim_x[0:380], sim_y[0:380], "-b", label="Pure-Pursuit+PPO",linewidth=2)
plt.grid(True)
plt.axis("equal")
plt.ylim(-60,40)
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend(loc='upper left')
# ax = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_x, "-g", label="reward")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Instant Reward")

# ax = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_reward, "-g", label="reward")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Instant Reward")

# # ax = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
# # plt.plot(sim_episode, sim_average_reward, "-", color='C1', label="ave_reward")
# # plt.plot(sim_episode, sim_average_reward, ".", color='C1', label="ave_reward")
# # plt.grid(True)
# # plt.xlabel("Episode")
# # plt.ylabel("Average Reward")

# ###

# # ax = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
# # plt.plot(sim_t, sim_a_PID, "--b", label="PID")
# # plt.plot(sim_t, sim_a_RL, "-g", label="RL")
# # plt.grid(True)
# # plt.xlabel("Time [s]")
# # plt.ylabel("Acceleration")
# # plt.legend(loc='upper left')

# ax = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_d_pure_pursuit, "--b", label="pure_pursuit")
# plt.plot(sim_t, sim_d_RL, "-g", label="RL")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Steering")
# plt.legend(loc='upper left')

# ax = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
# # plt.plot(sim_t, sim_a_RL_or_not, ".b", label="a_RL", markersize=10)
# # plt.plot(sim_t, sim_d_RL_or_not, ".g", label="d_RL", markersize=5)
# plt.grid(True)
# plt.ylim([0, 2])
# plt.xlabel("Time [s]")
# plt.ylabel("RL command")
# plt.legend(loc='upper left')

###

# ax = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
# plt.plot(sim_t, [i * 3.6 for i in sim_v], "-g", label="speed")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Speed [kmh]")

# ax = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_yaw, "-g", label="yaw")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Yaw [rad]")

# ax = plt.subplot2grid((3, 2), (2, 2), rowspan=1, colspan=1)
# plt.plot(sim_episode, sim_a_RL_percentage, "-b", alpha = 0.5, label="a_RL")
# plt.plot(sim_episode, sim_d_RL_percentage, "-g", alpha = 0.5, label="d_RL")
# plt.plot(sim_episode, sim_a_RL_percentage, ".b", label="a_RL")
# plt.plot(sim_episode, sim_d_RL_percentage, ".g", label="d_RL")
# plt.grid(True)
# plt.xlabel("Episode")
# plt.ylabel("RL Control Percentage [%]")
# plt.legend(loc='upper left')
####### transfer png to tiff
png1 = io.BytesIO()
fig1.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# # Save as TIFF
png2.save("result_plot.tiff")
png1.close()



# ##########################################
# plt.legend(loc='upper left')
# ax = plt.subplot(3,1,2)
# for i in range(1,6):
# 	plt.plot([0, max(sim_t_dqn)], [i, i],'lightgrey', linewidth=1)
# plt.plot(sim_t_dqn,sim_gear_dqn,color='C1',linewidth=3 )
# plt.plot(sim_t_dqn,sim_gear_pid,c = 'blue', linewidth=2,alpha=0.5)
# plt.title('(b) Gear Shift', loc='left')
# plt.yticks(np.arange(1, 6, 1))
# plt.ylabel('gear #')
# plt.xlim(0, max(sim_t_dqn))
# # plt.xlim(0, 1000)
# plt.grid(color='gray', linestyle='-', linewidth=0.2)
# ax.spines['top'].set_linewidth(0)
# ax.spines['right'].set_linewidth(0)

# ##########################################
# ax = plt.subplot(3,1,3)
# plt.plot(sim_t_dqn,sim_P_ice_dqn,color='C1',linewidth=3 )
# plt.plot(sim_t_dqn,sim_P_ice_pid,color='blue',linewidth=2,alpha=0.5)
# plt.title('(c) ICE Power', loc='left')
# plt.ylabel('kW')
# plt.xlim(0, max(sim_t_dqn))
# plt.xlabel('time (s)')
# # plt.xlim(0, 1000)
# plt.grid(color='gray', linestyle='-', linewidth=0.2)

# ax.spines['top'].set_linewidth(0)
# ax.spines['right'].set_linewidth(0)

# # Save the image in memory in PNG format
# png1 = io.BytesIO()
# fig1.savefig(png1, format="png")

# # Load this image into PIL
# png2 = Image.open(png1)

# # Save as TIFF
# png2.save("save1.tiff")
# png1.close()

