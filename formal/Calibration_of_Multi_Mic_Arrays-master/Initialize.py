# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ Time   ：2022-09-16
# @ File   : Initialize.py
# The function of this file is to generate motion trajectories and measurement information
# and to produce suitable initial values using the methods proposed in
# Graph SLAM-Based Joint Calibration of Multiple Asynchronous Microphone Arrays and Sound Source Localization

import random
import numpy as np
from  numpy.linalg import norm
from scipy.optimize import leastsq,least_squares
import scipy.io
from  utils import *
from  parameter import  true_value

def ICP(mic_num,time_steps,src_pos,measure_info,b):
    """
    :param mic_num
    :param time_steps
    :param src_pos
    :param measure_info:
    :param b: distance between the sound source position and the i-th mic
    :return: mic position, mic Euler angles
    """
    p_list = np.zeros((0, 3))
    p_prime_list = np.zeros((0, 3))
    for i in range(time_steps):
        p_i = src_pos[i]
        p_list = np.vstack((p_list, p_i))
        p_prime_i = measure_info[2 * i][mic_num][1:] * b[mic_num-1][i]
        p_prime_list = np.vstack((p_prime_list, p_prime_i))

    #  center of mass
    inertial_p = np.array([np.sum(p_list[:, 0]), np.sum(p_list[:, 1]), np.sum(p_list[:, 2])]) / len(p_list)
    inertial_p_prime = np.array(
        [np.sum(p_prime_list[:, 0]), np.sum(p_prime_list[:, 1]), np.sum(p_prime_list[:, 2])]) / len(p_prime_list)

    q_list = p_list - inertial_p
    q_prime_list = p_prime_list - inertial_p_prime

    W = np.zeros((3, 3))
    q_list_T = q_list.T
    for i in range(len(q_list)):
        W = W + q_list_T[:, i].reshape((3, 1)) @ q_prime_list[i].reshape((1, 3))
    rank = np.linalg.matrix_rank(W)
    if rank!=3:
        raise ValueError("W IS NOT FULL RANK")
    u, s, v = np.linalg.svd(W)
    R = u @ v
    xarr = inertial_p.reshape((3, 1)) - R@inertial_p_prime.reshape((3, 1))
    angle = rotationMatrixToEulerAngles(R)
    angle = angle/np.pi*180
    return xarr,angle

def TDOA(dis_i_k,sound_speed,initial_mic_asyn,time_step,interval,i):
    '''
    :param dis_i_k: distance between the sound source position and the i-th mic
    :param sound_speed
    :param initial_mic_asyn: x_tau & x_delta
    :param time_step
    :param interval
    :return: TDOA measurement
    '''
    result = float(dis_i_k[i]-dis_i_k[0])/sound_speed+ initial_mic_asyn[i][0]+time_step*interval*initial_mic_asyn[i][1]
    return result

def DOA(theta,S,X):
    '''
    :param R: rotation matrix
    :param S: source location
    :param X: array location
    :return: DOA measurement
    '''
    R_T = rotation_matrix(theta)
    X   = np.array(X)
    d = R_T@((S-X).reshape((3,1)))/distant(S,X)
    return d.T

def fitting_func(x, p):
    """
    y = kx+b
    """
    A, k = p
    return k*x+A

def fitting_residuals(p, y, x):
    return 0.5*(y - fitting_func(x, p))**2

def solve_b(X,angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,\
     angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD):
    '''
    :return: the distance of source from the i-th microphone array at the k-th time instance
    '''
    a = X[0]
    b = X[1]
    c = X[2]
    d = X[3]
    return [a ** 2 + b ** 2 - 2 * a * b * np.cos(angle_AB) - distant_AB**2,
            b ** 2 + c ** 2 - 2 * c * b * np.cos(angle_BC) - distant_BC**2,
            c ** 2 + d ** 2 - 2 * c * d * np.cos(angle_CD) - distant_CD**2,
            a ** 2 + d ** 2 - 2 * a * d * np.cos(angle_AD) - distant_AD**2,
            a ** 2 + c ** 2 - 2 * a * c * np.cos(angle_AC) - distant_AC**2,
            b ** 2 + d ** 2 - 2 * b * d * np.cos(angle_BD) - distant_BD**2]

def angle_bottom_edg(ori_A, ori_B, ori_C, ori_D,point_A,point_B,point_C,point_D):
    angle_AB = vex2theta(ori_A, ori_B)
    distant_AB = distant_b(point_A, point_B)
    angle_BC = vex2theta(ori_B, ori_C)
    distant_BC = distant_b(point_B, point_C)
    angle_CD = vex2theta(ori_C, ori_D)
    distant_CD = distant_b(point_C, point_D)
    angle_AD = vex2theta(ori_A, ori_D)
    distant_AD = distant_b(point_A, point_D)
    angle_AC = vex2theta(ori_A, ori_C)
    distant_AC = distant_b(point_A, point_C)
    angle_BD = vex2theta(ori_B, ori_D)
    distant_BD = distant_b(point_B, point_D)
    return [angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD]

def get_value():
    # true values of 8 mic_arrays and other init parameters
    initial_mic_location, initial_mic_angle, initial_mic_asyn, \
    sound_speed, time_steps, interval, \
    fig, TDOA_FIG, \
    mic_pose_std, mic_rot_std, max_offset, max_clock_diff, src_pose_std,\
    TDOA_std, DOA_std, odo_std = true_value()
    mic_num = len(initial_mic_angle)
    s_k_real = np.array([
                    [0.5, 1 ,0]
                    ])
    # Ten random trajectories in paper
    # random_choices = np.load("10_trajectories/random_np4.npy")

    # init the sound source trajectories
    # orient_option = np.array([
    #     [1, 0, 0],
    #     [-1, 0, 0],
    #     [0, 1, 0],
    #     [0, -1, 0],
    #     [0, 0, 1],
    #     [0, 0, -1]]
    # ) / 3
    # init_choices = [0]
    # random_choices = init_choices+random.choices([0,1,2,3,4,5],weights=[1,1,1,1,1,1], k=time_steps-1-len(init_choices))

    # Pentagram track in paper
    random_choices = list(range(119))
    orient_option = np.load("Pentagram_track.npy")

    # TDOA & DOA & Odometry measurement
    measure_info = []  # measurement in paper
    ID = []            # nodes of graph SLAM
    for i in range(time_steps):
        measure_info.append([])
        dis_i_k = [distant(s_k_real[-1], mic) for mic in initial_mic_location]
        T_i_k   = np.array([])
        d_i_k   = np.zeros((0,3))
        for j in range(mic_num):                                  # P-L constraint
            T_i_k= np.append(T_i_k, TDOA(dis_i_k, sound_speed, initial_mic_asyn, i + 1, interval,j))
            d_i_k= np.vstack((d_i_k,
                                DOA(initial_mic_angle[j], s_k_real[-1],initial_mic_location[j])
                              ))
            TDOA_error = np.random.normal(0, TDOA_std)
            DOA_error_x = np.random.normal(0, DOA_std)
            DOA_error_y = np.random.normal(0, DOA_std)
            DOA_error_z = np.random.normal(0, DOA_std)
            noisy = np.array([TDOA_error, DOA_error_x, DOA_error_y, DOA_error_z])
            measure = np.insert(d_i_k[j], 0, T_i_k[j])
            measure = measure+noisy
            measure_info[-1].append(measure)
        ID.append([1, mic_num * 8 + 1 + 3 * i])
        if i + 1 < time_steps:                                     # P-P constraint
            choice    = random_choices[i]
            noisy_odo = np.random.normal(0, odo_std, (3, 1)).reshape(-1)
            odo_info  = np.array(orient_option[choice])
            odo_info  = odo_info + noisy_odo
            measure_info.append([odo_info])
            next_point = s_k_real[-1] + orient_option[choice]
            s_k_real = np.append(s_k_real, np.array([next_point]), axis=0)
            ID.append([mic_num * 8 + 1 + 3 * i, mic_num * 8 + 4 + 3 * i])

    # True values of all unkown parameters in paper
    x_gt = np.zeros((8*mic_num+3*time_steps,1))
    for i in range(mic_num):
        x_gt[i*8  :i*8+3] = initial_mic_location[i].reshape((3,1))
        x_gt[i*8+3:i*8+6] = initial_mic_angle[i].reshape((3,1))
        x_gt[i*8+6:i*8+8] = initial_mic_asyn[i].reshape((2,1))
    for i in range(time_steps):
        x_gt[i*3+8*mic_num:i*3+8*mic_num+3] = s_k_real[i].reshape((3,1))

    # Estimated values of all unkown parameters in paper
    x = np.zeros((8*mic_num+3*time_steps,1))
    for i in range(1,mic_num):
        # For Experiment 1: x = x_gt + Gaussion Noise
        # x[i*8:i*8+3]   = x_gt[i*8:i*8+3]+np.random.normal(0,mic_pose_std*3,(3,1))
        # x[i*8+3:i*8+6] = x_gt[i*8+3:i*8+6]+np.random.normal(0,mic_rot_std*3,(3,1))#(0,15*np.pi/180,(3,1))
        # x[i*8+6]       =x_gt[i*8+6]+np.random.normal(0,max_offset/10*3)
        # x[i*8+7]       =x_gt[i*8+7]+np.random.normal(0,max_clock_diff/10*3)
        x[i*8+6]       =np.random.uniform(0,max_offset)
        x[i*8+7]       =np.random.uniform(0,max_clock_diff)

    # For Experiment 1: x = x_gt + Gaussion Noise
    # for i in range(time_steps):
    #     x[i*3+8*mic_num:i*3+8*mic_num+3] = x_gt[i*3+8*mic_num:i*3+8*mic_num+3]+ np.random.normal(0,src_pose_std*3,(3,1))

    # 1. Estimation of sound source position
    L = measure_info[1][0][0]                                          # the fist distance of moving
    measure_DOA_data = np.array(measure_info[::2])[:, 0][:, 1:]        # the DOA measurements of the 1st mic
    #  triangulation
    src_rad1 = measure_DOA_data[0]@ measure_DOA_data[1].T/(norm(measure_DOA_data[0])*norm(measure_DOA_data[1]))
    src_rad2 = measure_DOA_data[1]@ np.array([1,0,0]).T/(norm(measure_DOA_data[1]))
    d_11 = L*np.sin(np.arccos(src_rad2))/np.sin(np.arccos(src_rad1))

    src_pos = np.zeros((0,3))
    src_pos = np.vstack((src_pos,d_11*measure_DOA_data[0]))
    for i in range(len(random_choices)):
        src_pos = np.vstack((src_pos,src_pos[-1]+measure_info[2*i+1]))

    #  fuse odometry and DOA info
    dis_frame_1 = np.array([norm(src_pos[i]) for i in range(len(random_choices)+1)])
    src_pos_est_by_doa = np.array([dis_frame_1[i]* measure_DOA_data[i] for i  in range(len(measure_DOA_data))])
    src_pos = (src_pos+src_pos_est_by_doa)/2
    dis_frame_1 = np.array([norm(src_pos[i]) for i in range(len(random_choices)+1)])
    for i in range(time_steps):
        x[i*3+8*mic_num:i*3+8*mic_num+3] = src_pos[i].reshape(3,1)

    # 2. Estimation of the distance between the sound source and i-th mic
    b = []
    if time_steps%4 !=0:
        raise ValueError("Todo: Adding non-integer multiples of information processing")

    for mic in range(1,mic_num):
        measure_DOA_data = np.array(measure_info[::2])[:, mic][:, 1:]      # the DOA measurements of the i-th mic
        mic_i_b = np.array([])
        for i in range(0,time_steps,4):
            if i+4-time_steps>0:
                ori_A  , ori_B  , ori_C  , ori_D   = measure_DOA_data[time_steps-4:time_steps]
                point_A, point_B, point_C, point_D = src_pos[time_steps-4:time_steps]
            else:
                ori_A  , ori_B  , ori_C  , ori_D   = measure_DOA_data[i:i+4]
                point_A, point_B, point_C, point_D = src_pos[i:i+4]
            param = angle_bottom_edg(ori_A, ori_B, ori_C, ori_D,point_A,point_B,point_C,point_D)
            X0 = [1, 1, 1, 1]
            try:
                h = least_squares(solve_b, X0, args=param,bounds=(0, 10))
                mic_i_b = np.append(mic_i_b, h.x)
            except ValueError:
                point_E,point_F = src_pos[0:2]
                ori_E, ori_F = measure_DOA_data[0:2]
                param = angle_bottom_edg(ori_A, ori_B, ori_E, ori_F, point_A, point_B, point_E, point_F)
                h = least_squares(solve_b, X0, args=param,bounds=(0, 10))
                mic_i_b = np.append(mic_i_b, h.x[0:2])
                param = angle_bottom_edg(ori_C, ori_D, ori_E, ori_F, point_C, point_D, point_E, point_F)
                h = least_squares(solve_b, X0, args=param,bounds=(0, 10))
                mic_i_b = np.append(mic_i_b, h.x[0:2])
        b.append(mic_i_b)

    # 3. Estimation of microphone arrays positions and orientations
    # 4. Estimation of microphone arrays asynchronous parameters
    measure_TDOA_data = np.array(measure_info[::2])
    for i in range(1,mic_num):
        xarr, angle = ICP(i, time_steps, src_pos, measure_info, b)
        x[i * 8:i * 8 + 3] = xarr
        x[i * 8 + 3:i * 8 + 6] = angle.reshape(3, 1)

        # leastsq
        init_time_offset = x[i*8+6]
        init_clock_diff  = x[i*8+7]
        p0 = np.array([init_time_offset,init_clock_diff])
        b_i = b[i-1]
        fx = measure_TDOA_data[:,i][:,0]-(b_i-dis_frame_1)/sound_speed
        fitting_result = leastsq(fitting_residuals, p0, args=(fx, np.linspace(1,len(random_choices)+1,len(random_choices)+1)))
        x[i * 8 + 6] = fitting_result[0][0]
        x[i * 8 + 7] = fitting_result[0][1]

        # Asynchronous parameter fitting effect
        if TDOA_FIG:
            plt.plot(np.linspace(1,time_steps,time_steps),initial_mic_asyn[i][0]+np.linspace(1,time_steps,time_steps)*initial_mic_asyn[i][1] , label="True")
            plt.plot(np.linspace(1,time_steps,time_steps),x[i * 8 + 6]+np.linspace(1,time_steps,time_steps)*x[i * 8 + 7] , label="EST")
            plt.scatter(np.linspace(1,time_steps,time_steps), measure_TDOA_data[:,i][:,0]-(b_i-dis_frame_1)/sound_speed, label="$T_i^k-(\hat{d}_i^k-\hat{d}_1^k)/c$")
            plt.legend(fontsize = 17)
            plt.title("the i-th mic",fontsize = 20)
            plt.yticks(fontproperties='Times New Roman', size=15)
            plt.xticks(fontproperties='Times New Roman', size=18)
            # plt.savefig("the {}-th mic.svg".format(str(i+1)), dpi=750, format="svg")
            plt.show()
            plt.plot(np.linspace(1, time_steps, time_steps),b_i, label="Init", linewidth=5.0)
            plt.plot(np.linspace(1, time_steps, time_steps),[distant_b(src_pos[index_],initial_mic_location[i]) for index_ in range(time_steps)], label="True", linewidth=5.0)
            plt.legend()
            plt.title("the {}-th mic b".format(str(i + 1)))
            plt.show()

    if fig:
        plot_result(x,x_gt,mic_num,title="Initial Results")
    return x,measure_info,ID,mic_num,x_gt

if __name__=="__main__":
    get_value()








