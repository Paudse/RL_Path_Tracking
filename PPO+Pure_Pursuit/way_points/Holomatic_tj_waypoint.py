import csv
import numpy as np

lv_x = []
lv_y = []
av_z = []
dt = 0.1
Yaw = 0
X = 0
Y = 0
X_array = []
Y_array = []
Yaw_array = []

with open('linear_velocity.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    lv = [e for e in reader]
    csvfile.close()
    flip_k = 0
    for i in range(1,len(lv)):
        lv_x.append(float(lv[i][1]))
        lv_y.append(float(lv[i][2]))

with open('angular_velocity.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    av = [e for e in reader]
    csvfile.close()
    flip_k = 0
    for i in range(1,len(av)):
        av_z.append(float(av[i][3]))
for i in range(1,len(av)-1000):
	Yaw = Yaw + av_z[i]*dt
	dX = lv_x[10*i]*dt*np.cos(np.deg2rad(Yaw))-lv_y[10*i]*dt*np.sin(np.deg2rad(Yaw))
	dY = lv_x[10*i]*dt*np.sin(np.deg2rad(Yaw))+lv_y[10*i]*dt*np.cos(np.deg2rad(Yaw))
	X = X+dX
	Y = Y+dY
	X_array.append(round(X,2))
	Y_array.append(round(Y,2))
	Yaw_array.append(round(Yaw,2))


l = [X_array, Y_array, Yaw_array]

writer = csv.writer(open("Holomatic_tj_.csv", 'w'))
for i in range(0,len(Yaw_array)):
    writer.writerow(np.transpose(l)[i])


