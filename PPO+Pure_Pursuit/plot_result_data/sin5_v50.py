import matplotlib.pyplot as plt
import scipy.io 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io


fig1 = plt.figure(figsize=(6,5), dpi=300)
plt.subplots_adjust(wspace =0.5, hspace =0.3)

################################################################################################################################
data = scipy.io.loadmat('PPO_d5_v20_0_reward-82.93')
cx = data['cx'][0]
cy = data['cy'][0]
cv = data['cv'][0]
sim_t_ppo = data['sim_t'][0]
sim_x_ppo = data['sim_x'][0]
sim_y_ppo = data['sim_y'][0]

data = scipy.io.loadmat('PP_d5_v20_0_reward-43.7')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]

ax = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
plt.plot(cx, cy, "-r", label="Ref")
plt.plot(sim_x_ppo, sim_y_ppo, "-g", label="PP+PPO")
plt.plot(sim_x_pp, sim_y_pp, "--b", label="PP")
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
# plt.legend(loc='upper right')
plt.title('(d) v=20, rc=5', loc='left')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
############################

################################################################################################################################
data = scipy.io.loadmat('PPO_d5_v50_0_reward-77.99')
cx = data['cx'][0]
cy = data['cy'][0]
cv = data['cv'][0]
sim_t_ppo = data['sim_t'][0]
sim_x_ppo = data['sim_x'][0]
sim_y_ppo = data['sim_y'][0]

data = scipy.io.loadmat('PP_d5_v50_0_reward-31.85')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]

ax = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
plt.plot(cx, cy, "-r", label="Ref")
plt.plot(sim_x_ppo, sim_y_ppo, "-g", label="PP+PPO")
plt.plot(sim_x_pp, sim_y_pp, "--b", label="PP")
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
# plt.legend(loc='upper right')
plt.title('(b) v=50, rc=5', loc='left')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height*0.8])


############################

data = scipy.io.loadmat('PPO_d20_v20_0_reward-95.04')
cx = data['cx'][0]
cy = data['cy'][0]
cv = data['cv'][0]
sim_t_ppo = data['sim_t'][0]
sim_x_ppo = data['sim_x'][0]
sim_y_ppo = data['sim_y'][0]

data = scipy.io.loadmat('PP_d20_v20_0_reward-70.71')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]

ax = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
plt.plot(cx, cy, "-r", label="Ref")
plt.plot(sim_x_ppo, sim_y_ppo, "-g", label="PP+PPO")
plt.plot(sim_x_pp, sim_y_pp, "--b", label="PP")
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
# plt.legend(loc='upper right', ncol=3)
plt.title('(a) v=20, rc=20', loc='left')

# ax1 = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
ax.legend(loc='center', bbox_to_anchor=(1.75, 1.4),ncol=3)


############################

data = scipy.io.loadmat('PPO_d20_v50_0_reward-91.92')
cx = data['cx'][0]
cy = data['cy'][0]
cv = data['cv'][0]
sim_t_ppo = data['sim_t'][0]
sim_x_ppo = data['sim_x'][0]
sim_y_ppo = data['sim_y'][0]

data = scipy.io.loadmat('PP_d20_v50_0_reward-49.23')
sim_t_pp = data['sim_t'][0]
sim_x_pp = data['sim_x'][0]
sim_y_pp = data['sim_y'][0]

ax = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
plt.plot(cx, cy, "-r", label="Ref")
plt.plot(sim_x_ppo, sim_y_ppo, "-g", label="PP+PPO")
plt.plot(sim_x_pp, sim_y_pp, "--b", label="PP")
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
# plt.legend(loc='upper right', ncol=3)
plt.title('(c) v=50, rc=20', loc='left') # radius coefficient
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height*0.8])


############################

############################ transfer png to tiff ############################
png1 = io.BytesIO()
fig1.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# # Save as TIFF
png2.save("result_plot.tiff")
png1.close()

