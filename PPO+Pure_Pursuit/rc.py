import matplotlib.pyplot as plt
import numpy as np
import math

def cal_r(rc):	
	x = np.arange(0, 1000, 0.1)
	y = [-25+math.cos(ix / rc) * 25 for ix in x]
	y_dot =  [-25/rc*math.sin(ix / rc) for ix in x]
	y_dot_dot =  [-25/(rc**2)*math.cos(ix / rc) for ix in x]
	r = [(1+idy**2)**(3/2)/abs(iddy) for (idy, iddy) in zip(y_dot, y_dot_dot)]
	k = [1/ir for ir in r]
	return x, r

def plot_r(rc):
	fig = plt.figure(1)
	# fig.canvas.manager.window.wm_geometry('+2200+200')
	plt.cla()

	x, r = cal_r(rc)
	plt.plot(x, r)
	# plt.axhline(y=min(r), color='r', linestyle='-')
	plt.xlim(0, 500)
	plt.ylim(0, min(r)+300)
	plt.xlabel('x')
	plt.ylabel('Turning Radius (m)')
	plt.grid()
	print(min(r))
	plt.show()

if __name__ == '__main__':
	# r_min_list = []
	# rc_list = []
	# for rc in range(5,100,5):
	# 	x, r = cal_r(rc)
	# 	rc_list.append(rc)
	# 	r_min_list.append(min(r))
	# print(rc_list)
	# print(r_min_list)

	# plt.plot(rc_list,r_min_list)
	# plt.xlabel('rc')
	# plt.ylabel('Minimum Turning Radius (m)')
	# # plt.ylim(0, 500)
	# plt.grid()
	# plt.show()

	plot_r(20)

