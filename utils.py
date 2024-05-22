# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ File   : utils.py

import numpy as np
from numpy import sin as s
from numpy import cos as c
import math
from  numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.io import loadmat

def vertical_merge(matrix_block):
    result = matrix_block[0]
    for matrix in matrix_block[1:]:
        result = np.vstack((result, matrix))
    return result

def horizon_merge(matrix_block):
    result = matrix_block[0]
    for matrix in matrix_block[1:]:
        result = np.concatenate((result, matrix), axis=1)
    return result

def distant(a,b):
    c = (b-a)
    return (c[0]**2+c[1]**2+c[2]**2)**0.5

def rotation_matrix(theta , type = "Trans"):
    theta_x = theta[0]*np.pi/180
    theta_y = theta[1]*np.pi/180
    theta_z = theta[2]*np.pi/180
    R_x = np.array([
        [1.0,0,0],
        [0,c(theta_x),-s(theta_x)],
        [0,s(theta_x),c(theta_x)]
    ],dtype=np.float32)
    R_y = np.array([
        [c(theta_y) , 0.0 , s(theta_y)],
        [0,1.0,0],
        [-s(theta_y), 0 , c(theta_y)]
    ],dtype=np.float32)
    R_z = np.array([
        [c(theta_z), -s(theta_z),0.0],
        [s(theta_z), c(theta_z), 0.0],
        [0.0,0.0,1.0]
    ],dtype=np.float32)
    if type =="Trans":
        R = R_x.T@R_y.T@R_z.T
    elif type == "Not Trans":
        R = R_z@ R_y @ R_x
    return R

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def plot_axis(ax,origin_x,ax_vec_x,origin_y,ax_vec_y,origin_z,ax_vec_z,theta,trans,type):
    rot_mat = np.array(rotation_matrix(theta,type="Not Trans"))
    T_sb = np.zeros((4, 4))
    T_sb[:3, :3] = rot_mat
    T_sb[3, 3] = 1
    T_sb[:3, 3] = np.array(trans)
    x4 = np.array([(T_sb @ np.append(np.array(origin_x), 1))[0:3],
                   (T_sb @ np.append(np.array(ax_vec_x), 1))[0:3]
                   ])
    y4 = np.array([(T_sb @ np.append(np.array(origin_y), 1))[0:3],
                   (T_sb @ np.append(np.array(ax_vec_y), 1))[0:3]
                   ])
    z4 = np.array([(T_sb @ np.append(np.array(origin_z), 1))[0:3],
                   (T_sb @ np.append(np.array(ax_vec_z), 1))[0:3]
                   ])

    if type == "real":
        fig = ax.plot(x4[:, 0], x4[:, 1], x4[:, 2], c='r')
        fig = ax.plot(y4[:, 0], y4[:, 1], y4[:, 2], c='g')
        fig = ax.plot(z4[:, 0], z4[:, 1], z4[:, 2], c='b')

    elif type == "estimate":
        fig = ax.plot(x4[:, 0], x4[:, 1], x4[:, 2], c=(1.00,0.61,0.61))
        fig = ax.plot(y4[:, 0], y4[:, 1], y4[:, 2], c=(0.61,0.90,0.61))
        fig = ax.plot(z4[:, 0], z4[:, 1], z4[:, 2], c=(0.59,0.75,0.95))
    return ax,fig,x4[0]

def plot_result(x,x_gt,mic_num,title,other_data = None):
    fig = plt.figure(figsize=(8,6),dpi=100) #(figsize=plt.figaspect(1))
    ax = plt.axes(projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_title(title)
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")

    # Base coordinate system
    # x axis
    x1 = np.linspace(0, 0.2, num=2)
    y1 = np.zeros_like(x1)
    z1 = np.zeros_like(x1)
    fig1 = ax.plot(x1, y1, z1, c='r')
    frame_1 = zip(x1, y1, z1)
    [origin_x, ax_vec_x] = list(frame_1)
    # y axis
    y2 = np.linspace(0, 0.2, num=2)
    x2 = np.zeros_like(y2)
    z2 = np.zeros_like(x2)
    fig2 = ax.plot(x2, y2, z2, c='g')
    frame_2 = zip(x2, y2, z2)
    [origin_y, ax_vec_y] = list(frame_2)
    # z axis
    z3 = np.linspace(0, 0.2, num=2)
    x3 = np.zeros_like(z3)
    y3 = np.zeros_like(z3)
    fig3 = ax.plot(x3, y3, z3, c='b')
    frame_3 = zip(x3, y3, z3)
    [origin_z, ax_vec_z] = list(frame_3)
    ax.scatter(0, 0, 0, c='r', marker='o', label="Mic. pos. g.t.",s=10)

    # Plotting the true and initial values of the microphone array
    for i in range(1, mic_num):
        # True values
        pos = x_gt[8 * i:8 * i + 3].reshape(3)
        theta = x_gt[8 * i + 3:8 * i + 6].reshape(3)
        ax, fig,origin = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="real")
        ax.scatter(origin[0], origin[1], origin[2], c='r', marker='o',s=10)

        # initial values
        pos = x[8 * i:8 * i + 3].reshape(3)
        theta = x[8 * i + 3:8 * i + 6].reshape(3)
        ax, fig,origin = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="estimate")
        if i == 1:
            ax.scatter(origin[0], origin[1], origin[2], c='g', marker='s', label="Mic. pos. est.",s=10)
        else:
            ax.scatter(origin[0], origin[1], origin[2], c='g', marker='s',s=10)

    # True values
    ax.plot3D(x_gt[8 * mic_num::3].reshape(-1),
             x_gt[8 * mic_num + 1::3].reshape(-1),
             x_gt[8 * mic_num + 2::3].reshape(-1),
             color = 'blue',marker="x",linewidth=0.5,label="sound source position",markersize=4
             )
    # initial values
    ax.scatter(x[8 * mic_num::3],
               x[8 * mic_num + 1::3],
               x[8 * mic_num + 2::3],
               marker="s", color=(0.00,1.00,1.00),label='estimated src. pos.',s=10
               )
    if other_data is not None:
        ax.scatter(other_data[:,0],other_data[:,1],other_data[:,2])
    ax.legend()
    plt.show()

    # plot time offset
    # # True values
    plt.scatter(range(1, len(x_gt[8 + 6:mic_num * 8:8]) + 1), x_gt[8 + 6:mic_num * 8:8], marker='o', c='blue',label='True')
    # initial values
    plt.scatter(range(1, len(x[8 + 6:mic_num * 8:8]) + 1), x[8 + 6:mic_num * 8:8], marker='x', c='red',label = 'estimate')
    plt.legend()
    plt.title("Time offset")
    plt.show()

    # plot clock diff
    # # True values
    plt.scatter(range(1, len(x_gt[8 + 7:mic_num * 8:8]) + 1), x_gt[8 + 7:mic_num * 8:8], marker='o', c='blue',label='True')
    # initial values
    plt.scatter(range(1, len(x[8 + 7:mic_num * 8:8]) + 1), x[8 + 7:mic_num * 8:8], marker='x', c='red',label = 'estimate')
    plt.legend()
    plt.title("Clock difference")
    plt.show()

def distant_b(a,b):
    return norm((b-a))

def vex2theta(a,b):
    value = np.dot(a,b)/(norm(a)*norm(b))
    value = min(max(value,-1),1)
    return np.arccos(value)

def async_param(TDOA_measure,mic_asyn_param,record_time):
    TDOA_measure = TDOA_measure+ mic_asyn_param[:,0]+record_time*mic_asyn_param[:,1]
    return TDOA_measure

def load_measurement(pattern,type = 'Simulation'):
    if type == "Simulation":
        tdoa_mea  = np.load("Simulation_dataset/1_TDOA/pattern_{}_measurement.npy".format(pattern))
        tdoa_ture = np.load("Simulation_dataset/1_TDOA/pattern_{}_true.npy".format(pattern))
        doa_mea = np.load("Simulation_dataset/2_DOA/pattern_{}_measurement.npy".format(pattern))
        doa_ture = np.load("Simulation_dataset/2_DOA/pattern_{}_ture.npy".format(pattern))
        odo_mea = np.load("Simulation_dataset/3_ODO/pattern_{}_measurement.npy".format(pattern))
        odo_ture = np.load("Simulation_dataset/3_ODO/pattern_{}_ture.npy".format(pattern))

    elif type == "Real_world":
        tdoa_mea = (loadmat(f"Real_world_dataset/exp2/1_TDOA/TDOA_exp_{pattern}mea.mat")["delay_mean"]) / 16000
        doa_mea  = np.load(f"Real_world_dataset/exp2/2_DOA/pattern_{pattern}.npy")
        odo_mea  = np.load(f"Real_world_dataset/exp2/3_ODO/pattern_{pattern}.npy")

    return  tdoa_mea,doa_mea,odo_mea

def time_duration(dataset,type = 'Simulation'):
    if type == "Simulation":
        if dataset == 0:
            time_step = np.array(list(range(1,25)))
        else:
            time_step = np.array(list(range(1,81)))
    elif type == "Real_world":
        sound_event = loadmat(r"Real_world_dataset/exp2/the pattern " + str(dataset) + " sound seq.mat")["seq_time"]
        time_step = np.zeros(len(sound_event)-1)
        for i in range(len(sound_event)-1):
            current_time = sound_event[i+1,1] + sound_event[i+1, 2] / 1e9
            start_time = sound_event[0,1] + sound_event[0,2] / 1e9
            time_step[i] = current_time -start_time
    return  time_step

def transform_to_mic1_frame(mic_angle_gt,mic_location_gt,s_k_real):
    R_s1 = rotation_matrix(mic_angle_gt[0], type="Not Trans")
    mic_num = len(mic_angle_gt)

    for i in range(len(s_k_real)):
        s_k_real[i] = (R_s1.T @ s_k_real[i].reshape((3, 1))).reshape(3) - (
                    R_s1.T @ mic_location_gt[0].reshape((3, 1))).reshape(3)

    for i in range(1, mic_num):
        R_i1 = R_s1.T @ rotation_matrix(mic_angle_gt[i], type="Not Trans")
        x_arri = R_s1.T @ mic_location_gt[i].reshape((3, 1)) - R_s1.T @ mic_location_gt[0].reshape((3, 1))
        mic_angle_gt[i] = rotationMatrixToEulerAngles(R_i1) / np.pi * 180
        mic_location_gt[i] = x_arri.reshape(3)

    mic_location_gt[0] = np.zeros((1, 3))
    mic_angle_gt[0] = np.zeros((1, 3))
    return  mic_angle_gt,mic_location_gt,s_k_real

def remove_outliers(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    Q1 = np.percentile(data, 30)
    Q3 = np.percentile(data, 70)
    IQR = Q3 - Q1

    k = 1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    cleaned_data = [x for x in data if x >= lower_bound and x <= upper_bound]
    return cleaned_data

def transform_to_global_frame(x_gt,x_ICP,x,mic_num):

    R_s1 = rotation_matrix(np.array([90, 0, 90]))
    global_location = np.array([0,-0.109,0])

    for i in range(mic_num*8,len(x),3):
        x_gt[i:i+3] = R_s1.T @ x_gt[i:i+3] - R_s1.T @ global_location.reshape((3, 1))
        x_ICP[i:i + 3] = R_s1.T @ x_ICP[i:i + 3] -R_s1.T @ global_location.reshape((3, 1))
        x[i:i + 3] = R_s1.T @ x[i:i + 3] -R_s1.T @ global_location.reshape((3, 1))
    for i in range(mic_num):
        R_i1 = R_s1.T @ rotation_matrix(x_gt[i*8+3:i*8+6], type="Not Trans")
        x_arri = R_s1.T @ x_gt[i*8:i*8+3] - R_s1.T @ global_location.reshape((3, 1))
        x_gt[i*8+3:i*8+6] = (rotationMatrixToEulerAngles(R_i1) / np.pi * 180).reshape((3, 1))
        x_gt[i*8:i*8+3] = x_arri

        R_i1 = R_s1.T @ rotation_matrix(x[i*8+3:i*8+6], type="Not Trans")
        x_arri = R_s1.T @ x[i*8:i*8+3] - R_s1.T @ global_location.reshape((3, 1))
        x[i*8+3:i*8+6] = (rotationMatrixToEulerAngles(R_i1) / np.pi * 180).reshape((3, 1))
        x[i*8:i*8+3] = x_arri

        R_i1 = R_s1.T @ rotation_matrix(x_ICP[i*8+3:i*8+6], type="Not Trans")
        x_arri = R_s1.T @ x_ICP[i*8:i*8+3] - R_s1.T @ global_location.reshape((3, 1))
        x_ICP[i*8+3:i*8+6] = (rotationMatrixToEulerAngles(R_i1) / np.pi * 180).reshape((3, 1))
        x_ICP[i*8:i*8+3] = x_arri

    return  x_gt,x_ICP,x




