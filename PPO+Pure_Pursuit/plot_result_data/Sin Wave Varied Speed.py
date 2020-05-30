import matplotlib.pyplot as plt
import scipy.io 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io

# print(np.sin(30/180*np.pi))

# Load PID result
data = scipy.io.loadmat('4_reward-92.43')
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
sim_a_final = data['sim_a_final'][0]
sim_d_final = data['sim_d_final'][0]
sim_reward = data['sim_reward'][0]
sim_episode = data['sim_episode'][0]
sim_average_reward = data['sim_average_reward'][0]
# sim_a_RL_or_not = data['sim_a_RL_or_not'][0]
# sim_d_RL_or_not = data['sim_d_RL_or_not'][0]
# sim_a_RL_percentage = data['sim_a_RL_percentage'][0]
# sim_d_RL_percentage = data['sim_d_RL_percentage'][0]
sim_a_RL_correction = data['sim_a_RL_correction'][0]
sim_d_RL_correction = data['sim_d_RL_correction'][0]
sim_target_dis = data['sim_target_dis'][0]
sim_target_angle = data['sim_target_angle'][0]

data = scipy.io.loadmat('0_reward-77.11')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]
sim_d_final_pp = data['sim_d_final'][0]





7_reward-88.07
data = scipy.io.loadmat('7_reward-88.07')
cx2 = data['cx'][0]
cy2 = data['cy'][0]
cv2 = data['cv'][0]
sim_t2 = data['sim_t'][0]
sim_x2 = data['sim_x'][0]
sim_y2 = data['sim_y'][0]


data = scipy.io.loadmat('0_reward-77.11')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]
sim_d_final_pp = data['sim_d_final'][0]


data = scipy.io.loadmat('0_reward-79.28')
sim_t_pp2 = data['sim_t'][0]
sim_x_pp2 = data['sim_x'][0]
sim_y_pp2 = data['sim_y'][0]
sim_d_final_pp2 = data['sim_d_final'][0]

#### Fig1 ################################################################################################################################
# fig1 = plt.figure(figsize=(13,9), dpi=300)
fig1 = plt.figure(figsize=(6,5.5), dpi=300)
plt.subplots_adjust(wspace =1, hspace =0.5)
# plt.subplots_adjust(wspace =0.5, hspace =0.5)

##########################################

###

ax = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
plt.plot(cx, cy, "-r", label="Ref")
plt.plot(sim_x, sim_y, "-g", label="PP+PPO")
plt.plot(sim_x_pp, sim_y_pp, "--b", label="PP")
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
# plt.legend(loc='upper left')
plt.title('(a) Trajectory', loc='left')

# ax = plt.subplot2grid((3, 1), (0, 1), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_reward, "-g", label="reward")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Instant Reward")

# ax = plt.subplot2grid((3, 1), (0, 2), rowspan=1, colspan=1)
# plt.plot(sim_episode, sim_average_reward, "-", color='C1', label="ave_reward")
# plt.plot(sim_episode, sim_average_reward, ".", color='C1', label="ave_reward")
# plt.grid(True)
# plt.xlabel("Episode")
# plt.ylabel("Average Reward")

###
# ax = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_target_dis, "-g")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("target_dis")


# ax = plt.subplot2grid((3, 1), (1, 1), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_a_PID, "--b", label="PID")
# # plt.plot(sim_t, sim_a_RL, "-g", label="RL")
# plt.plot(sim_t, sim_a_RL_correction, "-g", label="RL")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Acceleration")
# # plt.legend(loc='upper left')

# ax = plt.subplot2grid((3, 1), (1, 2), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_a_final, "-g", label="a_final")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Acceleration")
# # plt.legend(loc='upper left')

# ax = plt.subplot2grid((3, 1), (1, 3), rowspan=1, colspan=1)
# plt.plot(sim_t, [i * 3.6 for i in sim_v], "-g", label="speed")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Speed [kmh]")

# ax = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_target_angle, "-g")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("target_angle")

# ax = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
# plt.title('(b) Steering Command', loc='left')
# # plt.plot(sim_t, sim_d_PID, "--b", label="PID")
# plt.plot(sim_t, sim_d_pure_pursuit/np.pi*180, "-k", label="PP")
# # plt.plot(sim_t, sim_d_RL, "-g", label="RL")
# plt.plot(sim_t, sim_d_RL_correction/np.pi*180, "--g", label="PPO")
# plt.grid(True)
# plt.xlabel("Time (s))")
# plt.ylabel("($^\circ$)")
# plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
plt.title('(b) Steering Angle', loc='left')
plt.plot(sim_t, sim_d_final/np.pi*180, "-g", label="PP+PPO")
plt.plot(sim_t_pp, sim_d_final_pp/np.pi*180, "--b", label="PP")
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("($^\circ$)")
plt.legend(loc='upper left')

ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
plt.title('(c) Vehicle Speed', loc='left')
# plt.plot(sim_t, sim_d_PID, "--b", label="PID")
plt.plot(sim_t, sim_v*3.6 , "-b", label="PP")
# plt.plot(sim_t, sim_d_RL, "-g", label="RL")
# plt.plot(sim_t, sim_d_RL_correction/np.pi*180, "--g", label="PPO")
plt.grid(True)
plt.xlabel("Time (s))")
plt.ylabel("(km/h)")
# plt.legend(loc='upper right')
# ax = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_a_RL_or_not, ".b", label="a_RL", markersize=10)
# plt.plot(sim_t, sim_d_RL_or_not, ".g", label="d_RL", markersize=5)
# plt.grid(True)
plt.ylim([0, 70])
plt.yticks(np.arange(0,70, 20))
# plt.xlabel("Time [s]")
# plt.ylabel("RL command")
# plt.legend(loc='upper left')

###


# ax = plt.subplot2grid((3, 4), (2, 3), rowspan=1, colspan=1)
# plt.plot(sim_t, sim_yaw, "-g", label="yaw")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Yaw [rad]")
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

