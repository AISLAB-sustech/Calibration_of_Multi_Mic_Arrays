# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ Time   ：2022-09-16
# @ File   : utils.py

import numpy as np
from numpy import sin as s
from numpy import cos as c
import math
from  numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl

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
    '''
    :param a: position vector
    :param b: position vector
    :return: distant
    '''
    c = (b-a)
    return (c[0]**2+c[1]**2+c[2]**2)**0.5

def rotation_matrix(theta , type = "Trans"):
    """
    :param a: rotation of x axis
    :param b: rotation of y axis
    :param c: rotation of z axis
    :return: rotation matrix
    """
    theta_x = theta[0]*np.pi/180
    theta_y = theta[1]*np.pi/180
    theta_z = theta[2]*np.pi/180
    R_x = np.array([
        [1,0,0],
        [0,c(theta_x),-s(theta_x)],
        [0,s(theta_x),c(theta_x)]
    ])
    R_y = np.array([
        [c(theta_y) , 0 , s(theta_y)],
        [0,1,0],
        [-s(theta_y), 0 , c(theta_y)]
    ])
    R_z = np.array([
        [c(theta_z), -s(theta_z),0],
        [s(theta_z), c(theta_z), 0],
        [0,0,1]
    ])
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
    return ax,fig

def plot_result(x,x_gt,mic_num):
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = plt.axes(projection='3d')
    ax.set_title("3D")
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1, 2)

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

    # Plotting the true and initial values of the microphone array
    for i in range(1, mic_num):
        # True values
        pos = x_gt[8 * i:8 * i + 3].reshape(3)
        theta = x_gt[8 * i + 3:8 * i + 6].reshape(3)
        ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="real")

        # initial values
        pos = x[8 * i:8 * i + 3].reshape(3)
        theta = x[8 * i + 3:8 * i + 6].reshape(3)
        ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="estimate")

    # True values
    real = ax.plot3D(x_gt[8 * mic_num::3].reshape(-1),
                     x_gt[8 * mic_num + 1::3].reshape(-1),
                     x_gt[8 * mic_num + 2::3].reshape(-1),
                     linewidth=1.0
                     )

    # initial values
    ax.scatter(x[8 * mic_num::3],
               x[8 * mic_num + 1::3],
               x[8 * mic_num + 2::3],
               marker="x", c="red"
               )
    methods = ('g.t.', 'init')
    color = ['b', 'r']
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
    legend_labels = [methods[y] for y in range(2)]
    ax.legend(legend_lines, legend_labels, numpoints=1)
    plt.show()

    # plot time offset
    # # True values
    plt.scatter(range(1, len(x_gt[8 + 6:mic_num * 8:8]) + 1), x_gt[8 + 6:mic_num * 8:8], marker='o', c='blue')
    # initial values
    plt.scatter(range(1, len(x[8 + 6:mic_num * 8:8]) + 1), x[8 + 6:mic_num * 8:8], marker='x', c='red')
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
    legend_labels = [methods[y] for y in range(2)]
    plt.legend(legend_lines, legend_labels, numpoints=1)
    plt.title("Time offset")
    plt.show()

    # plot clock diff
    # # True values
    plt.scatter(range(1, len(x_gt[8 + 7:mic_num * 8:8]) + 1), x_gt[8 + 7:mic_num * 8:8], marker='o', c='blue')
    # initial values
    plt.scatter(range(1, len(x[8 + 7:mic_num * 8:8]) + 1), x[8 + 7:mic_num * 8:8], marker='x', c='red')
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
    legend_labels = [methods[y] for y in range(2)]
    plt.legend(legend_lines, legend_labels, numpoints=1)
    plt.title("clock diff")
    plt.show()

def distant_b(a,b):
    return norm((b-a))

def vex2theta(a,b):
    return np.arccos(np.dot(a,b)/(norm(a)*norm(b)))