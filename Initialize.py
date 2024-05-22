# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ File   : Initialize.py
# The function of this file is to produce suitable initial values using the methods proposed in
# SLAM-Based Joint Calibration of Multiple Asynchronous Microphone Arrays and Sound Source Localization

import random
from collections import defaultdict

import numpy as np
from  numpy.linalg import norm
from scipy.optimize import leastsq,least_squares
import scipy.io
from  utils import *
from  parameter import  *
import math
from itertools import combinations
from scipy.optimize import minimize
from scipy.stats import zscore
def ICP(mic_num,time_steps,src_pos,measure_info,b):
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
        print("W IS NOT FULL RANK")
    u, s, v = np.linalg.svd(W)
    R = u @ v
    xarr = inertial_p.reshape((3, 1)) - R@inertial_p_prime.reshape((3, 1))
    angle = rotationMatrixToEulerAngles(R)
    angle = angle/np.pi*180
    return xarr,angle

def TDOA(T_i_k, mic, s_k_real, sound_speed, time_step):
    for i in range(1,len(mic)):
        T_i_k[i] = (distant(mic[i], s_k_real[time_step]) - distant(mic[0], s_k_real[time_step])) / sound_speed
    return T_i_k

def DOA(theta,S,X):
    R_T = rotation_matrix(theta)
    X   = np.array(X)
    d = R_T@((S-X).reshape((3,1)))/distant(S,X)
    return d.T

def DOA_ERROR(DOAS,DOA_STD):
    r = np.sqrt(DOAS[:, 0] ** 2 + DOAS[:, 1] ** 2)
    azi = np.arctan2(DOAS[:, 1], DOAS[:, 0]) / np.pi * 180
    ele = np.arctan2(DOAS[:, 2], r) / np.pi * 180

    azi = azi+np.random.normal(0,DOA_STD,len(DOAS))
    ele = ele+np.random.normal(0,DOA_STD,len(DOAS))

    r = np.cos(ele / 180 * np.pi)
    x = r * np.cos(azi / 180 * np.pi)
    y = r * np.sin(azi / 180 * np.pi)
    z = np.sin(ele / 180 * np.pi)

    doas = np.vstack((x, y, z))
    return doas.T

def fitting_func(x, p):
    A, k = p
    return k*x+A

def fitting_residuals(p, y, x):
    return 0.5*(y - fitting_func(x, p))**2

def solve_b(X,angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,\
     angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD):
    a = X[0]
    b = X[1]
    c = X[2]
    d = X[3]
    result = [a ** 2 + b ** 2 - 2 * a * b * np.cos(angle_AB) - distant_AB**2,
            b ** 2 + c ** 2 - 2 * c * b * np.cos(angle_BC) - distant_BC**2,
            c ** 2 + d ** 2 - 2 * c * d * np.cos(angle_CD) - distant_CD**2,
            a ** 2 + d ** 2 - 2 * a * d * np.cos(angle_AD) - distant_AD**2,
            a ** 2 + c ** 2 - 2 * a * c * np.cos(angle_AC) - distant_AC**2,
            b ** 2 + d ** 2 - 2 * b * d * np.cos(angle_BD) - distant_BD**2]
    return result

def jac_h(X,angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,\
     angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD):
    a = X[0]
    b = X[1]
    c = X[2]
    d = X[3]
    jacobian = np.array([
        [2*a-2*b*np.cos(angle_AB),2*b-2*a*np.cos(angle_AB),0,0],
        [0,2*b-2*c*np.cos(angle_BC),2*c-2*b*np.cos(angle_BC),0],
        [0, 0, 2*c-2*d*np.cos(angle_CD), 2*d-2*c*np.cos(angle_CD)],
        [2*a-2*d*np.cos(angle_AD),0,0,2*d-2*a*np.cos(angle_AD)],
        [2*a-2*c*np.cos(angle_AC),0,2*c-2*a*np.cos(angle_AC),0],
        [0,2*b-2*d*np.cos(angle_BD),0,2*d-2*b*np.cos(angle_BD)]
    ])
    return  jacobian

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
def filter_points_in_same_line(pointA,pointB,pointC,pointD):
    distance1 = point_to_line(pointA,pointB,pointC)
    distance2 = point_to_line(pointA,pointB,pointD)
    distance3 = point_to_line(pointA, pointC, pointD)
    distance4 = point_to_line(pointB, pointC, pointD)
    return min(distance1,distance2,distance3,distance4)
def point_to_line(pointA,pointB,pointP):
    direction_vector = pointB - pointA
    length_AB = np.linalg.norm(direction_vector)
    cross_product = np.cross(pointB - pointA, pointA - pointP)
    distance = np.linalg.norm(cross_product) / length_AB
    return distance


def get_initial_value(dataset,category):
    # true values of  mic_arrays and src
    mic_location_gt, mic_angle_gt, mic_asyn_gt, s_k_real= true_value(dataset,type=category)
    record_time = time_duration(dataset, category)

    if category =='Simulation':
        sound_speed, interval, \
        fig, TDOA_FIG, \
        mic_pose_std, mic_rot_std, max_offset, max_clock_diff, src_pose_std, \
        TDOA_std, DOA_std, odo_std = initial_config(category)
    elif category == 'Real_world':
        sound_speed, interval, \
        fig, TDOA_FIG, \
        mic_pose_std, mic_rot_std, max_offset, max_clock_diff, src_pose_std, \
        _, _, _ = initial_config(category)

    mic_num = len(mic_location_gt)
    time_steps = len(s_k_real)

    # TDOA & DOA & Odometry measurement
    measure_info = []
    ID = []

    if dataset != 0:
        tdoa_mea, doa_mea, odo_mea = load_measurement(dataset,type = category)
        if category == 'Real_world':
            mic_1_angle = np.array([90, 0, 90])
            for i in range(len(odo_mea)):
                odo_mea[i] = rotation_matrix(mic_1_angle) @ (odo_mea[i])

    # Simulation preset trajactory
    else:
        tdoa_mea = np.zeros((time_steps,mic_num))
        doa_mea  = np.zeros((time_steps,mic_num,3))
        odo_mea = np.zeros((time_steps,3))
        for i in range(time_steps-1):
            odo_mea[i] = s_k_real[i+1]-s_k_real[i]  + np.random.normal(0,odo_std,(1,3))

    src_pos_without_startpoint = np.zeros((time_steps,3))
    for i in range(1,time_steps):
        src_pos_without_startpoint[i] = src_pos_without_startpoint[i-1]+odo_mea[i-1]
    for i in range(time_steps):
        measure_info.append([])
        T_i_k   = tdoa_mea[i]
        d_i_k = doa_mea[i]
        if dataset == 0:
            # pre-set simulation case
            T_i_k    = TDOA(T_i_k, mic_location_gt,s_k_real,sound_speed,i)
            T_i_k   = async_param(T_i_k,mic_asyn_gt,record_time[i])
            for mic in range(mic_num):
                d_i_k[mic] = DOA(mic_angle_gt[mic],s_k_real[i],mic_location_gt[mic]).reshape(-1)
            T_i_k[1:] = T_i_k[1:]+np.random.normal(0, TDOA_std, (len(T_i_k)-1))
            d_i_k = DOA_ERROR(d_i_k,DOA_std)
            tdoa_mea[i] = T_i_k
            doa_mea[i] = d_i_k
        for j in range(mic_num):                                  # P-L constraint
            measure = np.insert(d_i_k[j], 0, T_i_k[j])
            measure_info[-1].append(measure)
        ID.append([1, mic_num * 8 + 1 + 3 * i])
        if i + 1 < time_steps:                                    # P-P constraint
            measure_info.append([odo_mea[i]])
            ID.append([mic_num * 8 + 1 + 3 * i, mic_num * 8 + 4 + 3 * i])

    # True values of all parameters
    x_gt = np.zeros((8*mic_num+3*time_steps,1))
    for i in range(mic_num):
        x_gt[i*8  :i*8+3] = mic_location_gt[i].reshape((3,1))
        x_gt[i*8+3:i*8+6] = mic_angle_gt[i].reshape((3,1))
        x_gt[i*8+6:i*8+8] = mic_asyn_gt[i].reshape((2,1))
    for i in range(time_steps):
        x_gt[i*3+8*mic_num:i*3+8*mic_num+3] = s_k_real[i].reshape((3,1))

    # Estimated values of all parameters
    x = np.zeros((8*mic_num+3*time_steps,1))

    # 1. Estimation of sound source position
    #  triangulation
    measure_DOA_data = np.array(measure_info[::2])[:, 0][:, 1:]         # the DOA measurements of the 1st mic
    src_rad1 = measure_DOA_data[0] @ measure_DOA_data[1].T / \
               (norm(measure_DOA_data[0]) * norm(measure_DOA_data[1]))
    src_rad2 = measure_DOA_data[1] @ np.array(measure_info[1]).reshape(3,1) / \
               (norm(measure_DOA_data[1])*norm(np.array(measure_info[1])))
    L = norm(np.array(measure_info[1]))
    d_11 = L*np.sin(np.arccos(src_rad2))/np.sin(np.arccos(src_rad1))

    d_11 = []
    for step in range(1,time_steps):
        src_rad1 = measure_DOA_data[0] @ measure_DOA_data[step].T / \
                   (norm(measure_DOA_data[0]) * norm(measure_DOA_data[step]))
        point_to_1 = src_pos_without_startpoint[step]
        src_rad2 = measure_DOA_data[step] @ point_to_1.reshape(3, 1) / \
                   (norm(measure_DOA_data[step]) * norm(point_to_1))
        src_rad1 = max(min(src_rad1, 1), -1)
        src_rad2 = max(min(src_rad2, 1), -1)
        d_11_sub = norm(point_to_1)*np.sin(np.arccos(src_rad2))/(np.sin(np.arccos(src_rad1)))
        d_11.append(d_11_sub)
    d_11 = remove_outliers(d_11)
    d_11 = np.mean(np.array(d_11))

    src_pos = np.zeros((0,3))
    src_pos = np.vstack((src_pos,d_11*measure_DOA_data[0]))
    for i in range(time_steps-1):
        src_pos = np.vstack((src_pos,src_pos[-1]+measure_info[2*i+1]))

    dis_frame_1 = np.array([norm(src_pos[i]) for i in range(time_steps)])
    src_pos_est_by_doa = np.array([dis_frame_1[i] * measure_DOA_data[i] for i in range(time_steps)])
    src_pos = (src_pos + src_pos_est_by_doa) / 2

    dis_frame_1 = np.array([norm(src_pos[i]) for i in range(time_steps)])
    for i in range(time_steps):
        x[i*3+8*mic_num:i*3+8*mic_num+3] = src_pos[i].reshape(3,1)

    # 2. Estimation of the distance between the sound source and i-th mic
    b = []
    if category =="Simulation" :
        combi_list = [(i,i+1,i+2,i+3) for i in range(time_steps-3)]
    else:
        combi_list = list(combinations([i for i in range(time_steps)], 4))
    for mic in range(1,mic_num):
        measure_DOA_data = np.array(measure_info[::2])[:, mic][:, 1:]      # the DOA measurements of the i-th mic
        mic_i_b = np.array([])
        lenth_src_mic = defaultdict(list)
        for pair in combi_list:
            index_a, index_b, index_c, index_d = pair
            ori_A, ori_B, ori_C, ori_D = measure_DOA_data[index_a], measure_DOA_data[index_b], measure_DOA_data[
                index_c], measure_DOA_data[index_d]
            point_A, point_B, point_C, point_D = src_pos[index_a], src_pos[index_b], src_pos[index_c], src_pos[index_d]
            param = angle_bottom_edg(ori_A, ori_B, ori_C, ori_D,point_A,point_B,point_C,point_D)
            X0 = [2.0]*4

            h = least_squares(solve_b, X0, args=param,jac=jac_h,bounds=(0,10))
            lenth_src_mic[str(index_a)].append(h.x[0])
            lenth_src_mic[str(index_b)].append(h.x[1])
            lenth_src_mic[str(index_c)].append(h.x[2])
            lenth_src_mic[str(index_d)].append(h.x[3])

        index = sorted(lenth_src_mic.keys())
        int_index = list(map(int, index))
        int_index = sorted(int_index)

        # do the outlier removing
        for i in index:
            lenth_src_mic[i] = remove_outliers(lenth_src_mic[i])
        mic_i_b = np.array([np.mean(lenth_src_mic[str(i)]) for i in int_index])
        b.append(mic_i_b)

    # 3. Estimation of microphone arrays positions and orientations
    # 4. Estimation of microphone arrays asynchronous parameters
    measure_TDOA_data = np.array(measure_info[::2])

    for i in range(1,mic_num):
        xarr, angle = ICP(i, time_steps, src_pos, measure_info, b)
        x[i * 8:i * 8 + 3] = xarr.reshape(3, 1)
        x[i * 8 + 3:i * 8 + 6] = angle.reshape(3, 1)

        # leastsq
        init_time_offset = x[i*8+6]
        init_clock_diff  = x[i*8+7]
        p0 = np.array([init_time_offset,init_clock_diff])
        b_i = b[i-1]

        # LS Method
        fx = measure_TDOA_data[:,i][:,0]-(b_i-dis_frame_1)/sound_speed
        fitting_result = leastsq(fitting_residuals, p0, args=(fx, record_time))

        # fit
        regression_predict = fitting_result[0][0]+record_time*fitting_result[0][1]
        residuals = fx - regression_predict
        z_scores = np.abs(zscore(residuals))
        threshold = 2
        outliers = np.where(z_scores > threshold)[0]
        x_cleaned = np.delete(record_time, outliers)
        y_cleaned = np.delete(regression_predict, outliers)
        fitting_result = leastsq(fitting_residuals, fitting_result[0], args=(y_cleaned, x_cleaned))

        x[i * 8 + 6] = fitting_result[0][0]
        x[i * 8 + 7] = fitting_result[0][1]


        # Asynchronous parameter fitting effect
        if TDOA_FIG:
            plt.plot(np.linspace(1,time_steps,time_steps),mic_asyn_gt[i][0]+record_time*mic_asyn_gt[i][1] , label="True")
            plt.plot(np.linspace(1,time_steps,time_steps),x[i * 8 + 6]+record_time*x[i * 8 + 7] , label="EST")
            plt.scatter(np.linspace(1,time_steps,time_steps), measure_TDOA_data[:,i][:,0]-(b_i-dis_frame_1)/sound_speed, label="$T_i^k-(\hat{d}_i^k-\hat{d}_1^k)/c$")
            plt.legend(fontsize = 17)
            plt.title("the i-th mic",fontsize = 20)
            plt.yticks(fontproperties='DejaVu Sans', size=15)
            plt.xticks(fontproperties='DejaVu Sans', size=18)
            # plt.savefig("the {}-th mic.svg".format(str(i+1)), dpi=750, format="svg")
            plt.show()
            plt.scatter(np.linspace(1, time_steps, time_steps),b_i, label="Init", linewidth=5.0)
            plt.scatter(np.linspace(1, time_steps, time_steps),[distant_b(s_k_real[index_],mic_location_gt[i]) for index_ in range(time_steps)], label="True", linewidth=5.0)
            plt.legend()
            plt.title("the {}-th mic b".format(str(i + 1)))
            plt.show()

    if fig:
        plot_result(x,x_gt,mic_num,title="Initial Results")

    return x,measure_info,ID,mic_num,x_gt,interval,record_time