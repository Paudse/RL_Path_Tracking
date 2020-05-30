import matplotlib.pyplot as plt
import scipy.io 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io



data = scipy.io.loadmat('2_io-82.8')
input_ = data['input_'][0]
output = data['output'][0]

#### Fig1 ################################################################################################################################
# fig1 = plt.figure(figsize=(13,9), dpi=300)
fig1 = plt.figure(figsize=(6,5), dpi=300)
plt.subplots_adjust(wspace =0, hspace =0)
# plt.subplots_adjust(top = 1, bottom = -1, right = 1, left = 0.1, hspace = 1, wspace = 0)
# plt.margins(0,0)
# plt.subplots_adjust(wspace =0.5, hspace =0.5)

##########################################

###

# ax = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)

# plt.plot(input_/np.pi*180, output/np.pi*180, "b", label="PP+PPO")
plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
plt.grid(True)
plt.plot([-180, 180], [0, 0],color='black')
plt.plot([0, 0], [-30, 30],color='black')
plt.plot(input_/np.pi*180, output/np.pi*180, "b", label="PP+PPO")

plt.plot([i*180/np.pi for i in input_], [i*180/np.pi for i in output])
# plt.axis("equal")
plt.ylabel("PPO Steering Cmd ($^\circ$)")
plt.xlabel("Target Angle ($^\circ$)")
plt.xlim(-180, 180)
plt.ylim(-25, 25)
# plt.legend(loc='upper right')
# plt.title('(a) Trajectory', loc='left')


####### transfer png to tiff
png1 = io.BytesIO()
fig1.savefig(png1, format="png",pad_inches=0.0)

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

