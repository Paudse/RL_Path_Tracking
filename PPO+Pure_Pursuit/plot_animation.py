import math
import matplotlib.pyplot as plt
import scipy.io
import numpy as np



def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def show_step_animation(cx, cy, x, y, yaw, state_x, state_y, state_yaw, current_v, target_ind):  # pragma: no cover
    plt.figure(1)
    plt.cla()
    plot_arrow(state_x, state_y, state_yaw)
    plt.plot(cx, cy, "-r", label="course")
    plt.plot(x, y, "-b", label="trajectory")
    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Speed[km/h]:" + str(current_v * 3.6)[:4])
    plt.pause(0.001)

def show_episode_plot(cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, sim_reward,\
    sim_episode, sim_average_reward, sim_a_RL_or_not, sim_d_RL_or_not, sim_a_RL_percentage, sim_d_RL_percentage, sim_a_RL_correction, sim_d_RL_correction, sim_target_dis,\
    sim_target_angle):
    fig = plt.figure(1)
    fig.canvas.manager.window.wm_geometry('+2200+200')
    plt.cla()

    ###

    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=1, colspan=1)
    plt.plot(cx, cy, "-r", label="spline")
    plt.plot(sim_x, sim_y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")

    ax = plt.subplot2grid((3, 4), (0, 1), rowspan=1, colspan=1)
    plt.plot(sim_t, sim_reward, "-g", label="reward")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Instant Reward")

    ax = plt.subplot2grid((3, 4), (0, 2), rowspan=1, colspan=1)
    plt.plot(sim_episode, sim_average_reward, "-", color='C1', label="ave_reward")
    # plt.text(sim_episode[-1], sim_average_reward[-1]+1, '%.2f' % sim_average_reward[-1], ha='center')
    plt.plot(sim_episode, sim_average_reward, ".", color='C1', label="ave_reward")
    yrange = max(sim_average_reward)- min(sim_average_reward)
    plt.text(sim_episode[-1], sim_average_reward[-1]+0.1*yrange, '%.2f' % sim_average_reward[-1], ha='right', va= 'bottom',fontsize=9)
    # print(sim_episode[-1], sim_average_reward[-1])
    # print(plt.text(sim_episode[-1], sim_average_reward[-1]+1, '%.2f' % sim_average_reward[-1], ha='center'))
    plt.ylim(min(sim_average_reward)-0.2*yrange,max(sim_average_reward)+0.5*yrange)
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")

    ###
    ax = plt.subplot2grid((3, 4), (1, 0), rowspan=1, colspan=1)
    plt.plot(sim_t, sim_target_dis, "-g")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("target_dis")


    ax = plt.subplot2grid((3, 4), (1, 1), rowspan=1, colspan=1)
    plt.plot(sim_t, sim_a_PID, "--b", label="PID")
    # plt.plot(sim_t, sim_a_RL, "-g", label="RL")
    plt.plot(sim_t, sim_a_RL_correction, "-g", label="RL")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration")
    # plt.legend(loc='upper left')

    ax = plt.subplot2grid((3, 4), (1, 2), rowspan=1, colspan=1)
    plt.plot(sim_t, sim_a_final, "-g", label="a_final")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration")
    # plt.legend(loc='upper left')

    ax = plt.subplot2grid((3, 4), (1, 3), rowspan=1, colspan=1)
    plt.plot(sim_t, [i * 3.6 for i in sim_v], "-g", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [kmh]")

    ax = plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=1)
    plt.plot(sim_t, [i*180/np.pi for i in sim_target_angle], "-g")
    # print(sim_target_angle)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("target_angle(degree)")

    ax = plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=1)
    # plt.plot(sim_t, sim_d_PID, "--b", label="PID")
    plt.plot(sim_t, [i*180/np.pi for i in sim_d_pure_pursuit], "--k", label="pure_pursuit")
    # plt.plot(sim_t, sim_d_RL, "-g", label="RL")
    plt.plot(sim_t, [i*180/np.pi for i in sim_d_RL_correction], "-g", label="RL")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("steering_cmd(degree)")
    plt.legend(loc='upper left')

    ax = plt.subplot2grid((3, 4), (2, 2), rowspan=1, colspan=1)
    plt.plot(sim_t, [i*180/np.pi for i in sim_d_final], "-g", label="d_final")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("final_steering_angle(degree)")
    # plt.legend(loc='upper left')

    # ax = plt.subplot2grid((3, 3), (1, 2), rowspan=1, colspan=1)
    # plt.plot(sim_t, sim_a_RL_or_not, ".b", label="a_RL", markersize=10)
    # plt.plot(sim_t, sim_d_RL_or_not, ".g", label="d_RL", markersize=5)
    # plt.grid(True)
    # plt.ylim([0, 2])
    # plt.xlabel("Time [s]")
    # plt.ylabel("RL command")
    # plt.legend(loc='upper left')

    ###


    ax = plt.subplot2grid((3, 4), (2, 3), rowspan=1, colspan=1)
    plt.plot(sim_t, sim_yaw, "-g", label="yaw")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")

    # ax = plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1)
    # plt.plot(sim_episode, sim_a_RL_percentage, "-b", alpha = 0.5, label="a_RL")
    # plt.plot(sim_episode, sim_d_RL_percentage, "-g", alpha = 0.5, label="d_RL")
    # # plt.plot(sim_episode, sim_a_RL_percentage, ".b", label="a_RL")
    # # plt.plot(sim_episode, sim_d_RL_percentage, ".g", label="d_RL")
    # plt.plot(sim_episode, sim_a_RL_percentage, ".b")
    # plt.plot(sim_episode, sim_d_RL_percentage, ".g")
    # plt.grid(True)
    # plt.xlabel("Episode")
    # plt.ylabel("RL Control Percentage [%]")
    # plt.legend(loc='upper left')

    plt.pause(0.0001)

def save_plot(episode,average_reward):

    plt.savefig('result_figure/episode_'+str(episode)+'_reward-'+str(round(average_reward,2))+'.png', format="png", dpi=100)

def save_sim(i,average_reward,cx, cy, cv, sim_t, sim_x, sim_y, sim_yaw, sim_v, \
                sim_a_PID, sim_d_PID, sim_d_pure_pursuit, sim_a_RL, sim_d_RL, sim_a_final, sim_d_final, \
                sim_reward, sim_episode, sim_average_reward, \
                sim_a_RL_correction, sim_d_RL_correction, sim_target_dis, sim_target_angle):
    scipy.io.savemat('result_data/'+str(i)+'_reward-'+str(round(average_reward,2))+'.mat', 
    {'cx':cx,
    'cy':cy,
    'cv':cv,
    'sim_t':sim_t,
    'sim_x':sim_x,
    'sim_y':sim_y,
    'sim_yaw':sim_yaw,
    'sim_v':sim_v,
    'sim_a_PID':sim_a_PID,
    'sim_d_pure_pursuit':sim_d_pure_pursuit,
    'sim_a_RL':sim_a_RL,
    'sim_d_RL':sim_d_RL,
    'sim_a_final':sim_a_final,
    'sim_d_final':sim_d_final,
    'sim_reward':sim_reward,
    'sim_episode':sim_episode,
    'sim_average_reward':sim_average_reward,
    'sim_a_RL_correction':sim_a_RL_correction,
    'sim_d_RL_correction':sim_d_RL_correction,
    'sim_target_dis':sim_target_dis, 
    'sim_target_angle':sim_target_angle
    })

def strategy_3d_map(model):
    # target_angle_list = np.arange(-3,3.3,0.3) 
    target_angle_list = np.arange(-2,2.1,0.1) 
    v_list = np.arange(10/3.6,55/3.6,1) 
    output = np.zeros((len(v_list),len(target_angle_list)))
    # print(output)
    # print(len(v_list),len(rc_list))
    target_angle_mesh, v_mesh = np.meshgrid(target_angle_list, v_list)
    # print(target_angle_mesh)
    # print(v_mesh[3,8],rc_mesh[3,8])
    # print(v_mesh)

    for i in range (0,(len(v_list))):
        for j in range (0,(len(target_angle_list))): 
            # print(i,j)
            output[i][j] = model.choose_action(np.array([target_angle_mesh[i,j], v_mesh[i,j]]))
            # print(i,j,output[i][j])
    # print(output)

    from matplotlib import cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    # plt.ion()
    norm = plt.Normalize(output.min(), output.max())
    colors = cm.viridis(norm(output))
    rcount, ccount, _ = colors.shape

    fig3 = plt.figure(3)
    fig3.canvas.manager.window.wm_geometry('+3000+500')
    plt.cla()
    ax = fig3.gca(projection='3d')
    # surf = ax.plot_surface(v_mesh, target_angle_mesh*180/np.pi, output*180/np.pi, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf = ax.plot_surface(target_angle_mesh*180/np.pi, v_mesh*3.6, output*180/np.pi, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    # ax.view_init(30, -90) 
    # ax.view_init(30, -60) 
    # ax.view_init(30, -70)
    ax.view_init(30, -110) 
    zplate = ax.plot_surface(target_angle_mesh*180/np.pi, v_mesh*3.6, np.zeros((len(v_list),len(target_angle_list))),alpha=0.5)
    ax.plot(np.zeros_like(v_list), v_list*3.6, 0, color='k',linewidth=2)
    ax.set_xlabel("target_angle(degree)")
    ax.set_ylabel("vehicle_speed(km/h)")
    ax.set_zlabel("steering_angle_compensation(degree)")
    # zplate = ax.plot_surface(v_mesh,target_angle_mesh*180/np.pi,  np.zeros((len(v_list),len(target_angle_list))),alpha=0.5)
    # ax.plot(v_list,np.zeros_like(v_list),  0, color='k')
    # ax.plot(target_angle_list, np.zeros_like(target_angle_list), 0, color='k')
    # xplate = ax.plot_surface(np.zeros((len(v_list),len(target_angle_list))), v_mesh, output*180/np.pi)
    # ax.contourf(target_angle_mesh*180/np.pi, v_mesh, output*180/np.pi, zdir='x', offset=-2, cmap=plt.get_cmap('rainbow'))
    # ax.contourf(target_angle_mesh*180/np.pi, v_mesh, output*180/np.pi, zdir='y', offset=-2, cmap=plt.get_cmap('rainbow'))
    # ax.contourf(target_angle_mesh*180/np.pi, v_mesh, output*180/np.pi, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
    # plt.show()

    # plt.pause(0.4)

def plot_action(model, save_io, episode, average_reward, target_speed):
    target_angle_list = np.arange(-2,2.1,0.05) 
    v_list = np.arange(10/3.6,55/3.6,5/3.6) 
    output = np.zeros((len(v_list),len(target_angle_list)))
    target_angle_mesh, v_mesh = np.meshgrid(target_angle_list, v_list)

    for i in range (0,(len(v_list))):
        for j in range (0,(len(target_angle_list))): 
            output[i][j] = model.choose_action(np.array([target_angle_mesh[i,j], v_mesh[i,j]]))

    # print('target_angle_list',len(target_angle_list))
    # print('v_list',len(v_list))
    # print('output[1]',len(output))
    fig4 = plt.figure(4)
    fig4.canvas.manager.window.wm_geometry('+3000+0')
    plt.cla()
    for k in range(1,len(output),3):
        plt.plot([i*180/np.pi for i in target_angle_list], [i*180/np.pi for i in output[k]], label= round(v_list[k]*3.6,0))
    # plt.show()
    plt.xticks(np.arange(-60,70,10))
    plt.plot([-50, 50], [0, 0],color='black')
    plt.plot([0, 0], [-1.5*10, 1.5*10],color='black')
    plt.xlabel("target_angle(degree)")
    plt.ylabel("steering_angle_compensation(degree)")
    # plt.plot(x, y, "-b", label="trajectory")
    # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    # plt.axis("equal")
    plt.grid(True)
    plt.title("Target Speed[km/h]:" + str(round(target_speed*3.6,2)))
    plt.legend(loc='upper left')

def plot_action_old(model, save_io, episode, average_reward, target_speed):
    output = []
    input_ = [] # target_angle
    for i in range(-300,300,1):
        # np.array([i])
        k = i/100
        action_RL = model.choose_action(np.array([k, target_speed]))
        d_RL = action_RL[0]
        input_.append(k)
        output.append(d_RL)

    import scipy.io
    import matplotlib.pyplot as plt


    # mngr = plt.get_current_fig_manager()  # 获取当前figure manager
    # mngr.window.wm_geometry("+380+310")

    fig=plt.figure(2)
    fig.canvas.manager.window.wm_geometry('+3000+0')
    plt.cla()
    plt.plot([i*180/np.pi for i in input_], [i*180/np.pi for i in output])
    plt.plot([-5, 5], [0, 0],color='black')
    plt.plot([0, 0], [-1.5, 1.5],color='black')
    plt.xlabel("target_angle(degree)")
    plt.ylabel("steering_angle_compensation(degree)")
    # plt.plot(x, y, "-b", label="trajectory")
    # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    # plt.axis("equal")
    # plt.grid(True)
    plt.title("Target Speed[km/h]:" + str(round(target_speed*3.6,2)))
    # plt.pause(0.001)
    # if save_io:
    #   scipy.io.savemat('result_data/'+str(episode)+'_io-'+str(round(average_reward,2))+'.mat', {'input_':input_,'output':output,})
        # plt.savefig('result_figure/episode_'+str(episode)+'_io-'+str(round(average_reward,2))+'.png', format="png", dpi=100)