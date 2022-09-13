import numpy as np
# import jax.numpy as np
# from create_general_route import *
from scipy import sparse
from math import sqrt
import random
from scipy.sparse.linalg import inv,spsolve
from  numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from ICP import *
import scipy.io

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

def linearize_pose_pose_constraint(x1,x2,measurement):
    measurement = np.array(measurement)
    row = measurement.shape[1]
    measurement = measurement.reshape((row,1))
    e = x2-x1-measurement
    A = -np.eye(3)
    B = np.eye(3)
    return e,A,B

def linearize_pose_landmark_constraint(x,l,measurement,toidx,mic_num):
    '''
    :param x: robot node info                                        (3x1)
    :param l: mic node info (positions tau delta)*8                  (40x1)
    :param measurement:   edge[1]    difference between ith and 1st  (7x1)
    :param toidx:         edge[-1]   robot node about time           (scalar)
    :param est_delay_on:
    :param est_drift_on:
    :param g:
    :return:
    '''
    # 定义误差矩阵
    # print("measure:",measurement)
    e = np.zeros((4*(mic_num),1))
    measurement = np.array(measurement).reshape((mic_num)*4,1)
    # 对每个麦克风求误差
    e[0] = 0
    e[1:4]= x/norm(x)-measurement[1:4]
    # e = TDOA_n-1 - Measure
    for n in range(1,mic_num):
        del_x = float(x[0] - l[8 * n])
        del_y = float(x[1] - l[8 * n + 1])
        del_z = float(x[2] - l[8 * n + 2])
        distance = sqrt( del_x**2+ del_y**2+ del_z**2)
        # TDOA error
        e[4*n]= (distance- sqrt(x[0]**2+x[1]**2+x[2]**2))/340.0+ l[8*n+6]+(toidx-8*mic_num+2)/3.0 * l[8*n+7] \
                    - measurement[4*n]
        # DOA error
        R_T = rotation_matrix([float(l[8*n+3]),float(l[8*n+4]),float(l[8*n+5])])
        R_T = np.array(R_T)
        e[4*(n)+1:4*(n)+4] = R_T@(x-l[8*n:8*n+3])/distance\
                                 -measurement[4*(n)+1:4*(n)+4]

    # L matrix
    A = np.zeros((4*(mic_num-1),8))      # computation of A,  de/dl    the partial of error function for landmark
    B = np.zeros((0,3))          # computation of B,  de/dx    the partial of error function for src

    for n in range(1,mic_num):
        # 计算每一个麦克风的 A B
        del_x = float(x[0] - l[8 * n])
        del_y = float(x[1] - l[8 * n + 1])
        del_z = float(x[2] - l[8 * n + 2] )
        distance = sqrt(del_x ** 2 + del_y ** 2 + del_z ** 2)
        theta_x = float(l[8 * n + 3])
        theta_y = float(l[8 * n + 4])
        theta_z = float(l[8 * n + 5])
        # Jacobian of TDOA
        h = np.array([
            (-del_x / distance) / 340.0,
            (-del_y / distance) / 340.0,
            (-del_z / distance) / 340.0,
            0,
            0,
            0,
            1,
            (toidx-8*mic_num+2)/3.0
        ])

        # Jacobian of DOA
        R_T = rotation_matrix([theta_x , theta_y , theta_z])
        theta_x = theta_x *np.pi/180
        theta_y = theta_y * np.pi / 180
        theta_z = theta_z * np.pi / 180
        U_A =  np.array([
            [del_y ** 2 + del_z ** 2, -del_x * del_y, -del_x * del_z],
            [-del_x * del_y, del_x ** 2 + del_z ** 2, -del_y * del_z],
            [-del_x * del_z, -del_y * del_z, del_x ** 2 + del_y ** 2]])/(distance**3)
        U = np.dot(-1*np.array(R_T),U_A)
        V =  np.array([
                    [0
                         ,
                      -s(theta_y) * c(theta_z) * del_x -
                      s(theta_y) * s(theta_z) * del_y -
                      c(theta_y) * del_z
                         ,
                      (-c(theta_y) * s(theta_z)) * del_x +
                      (c(theta_y) * c(theta_z)) * del_y
                      ]
            ,
            [
                (c(theta_x) * s(theta_y) * c(theta_z) + s(theta_x) * s(theta_z)) * del_x +
                (c(theta_x) * s(theta_y) * s(theta_z) - s(theta_x) * c(theta_z)) * del_y +
                (c(theta_x) * c(theta_y)) * del_z
                ,
                s(theta_x) * c(theta_y) * c(theta_z) * del_x +
                s(theta_x) * c(theta_y) * s(theta_z) * del_y -
                s(theta_x) * s(theta_y) * del_z
                ,
                (-c(theta_x) * c(theta_z) - s(theta_x) * s(theta_y) * s(theta_z)) * del_x +
                (-c(theta_x) * s(theta_z) + s(theta_x) * s(theta_y) * c(theta_z)) * del_y
            ]
            ,
            [
                (-s(theta_x) * s(theta_y) * c(theta_z) + c(theta_x) * s(theta_z)) * del_x +
                (-s(theta_x) * s(theta_y) * s(theta_z) - c(theta_x) * c(theta_z)) * del_y +
                (-s(theta_x) * c(theta_y)) * del_z
                ,
                c(theta_x) * c(theta_y) * c(theta_z) * del_x +
                c(theta_x) * c(theta_y) * s(theta_z) * del_y -
                c(theta_x) * s(theta_y) * del_z
                ,
                (s(theta_x) * c(theta_z) - c(theta_x) * s(theta_y) * s(theta_z)) * del_x +
                (s(theta_x) * s(theta_z) + c(theta_x) * s(theta_y) * c(theta_z)) * del_y
            ]
        ])/distance
        U_V = horizon_merge([U,V,np.zeros((3,2))])

        # 前面空了一个7x5 全0阵
        # L^k

        A = horizon_merge([ A , vertical_merge([
                                            np.zeros((4*(n-1),8)),
                                            h,
                                            U_V,
                                            np.zeros((4*(mic_num-1-n),8))
                                                ])
                            ])
        # A = vertical_merge([np.zeros((4,A.shape[1])),A])
        # T
        B1 = np.array([
            ((del_x/distance)  - float(x[0]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0,
            ((del_y/distance)  - float(x[1]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0,
            ((del_z/distance)  - float(x[2]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0
            ])
        B  = vertical_merge([B, B1, -U])

    # print(h)
    # print("A.shape",A.shape)
    A = vertical_merge([np.zeros((4, A.shape[1])), A])

    del_x = float(x[0] )
    del_y = float(x[1] )
    del_z = float(x[2] )
    # print("B.shape",B.shape)
    U = np.array([
            [0,0,0],
            [del_y ** 2 + del_z ** 2, -del_x * del_y, -del_x * del_z],
            [-del_x * del_y, del_x ** 2 + del_z ** 2, -del_y * del_z],
            [-del_x * del_z, -del_y * del_z, del_x ** 2 + del_y ** 2]])/(norm((del_x,del_y,del_z))**3)
    B = vertical_merge([U, B])
    # print(B[1:])
    # print(A.shape,B.shape)
    return e, A , B
    # A 7X40
    # B 7X3


def info_matrix(mic_num):
    TDOA_var = 2.3e8
    DOA_var = 1111#TDOA_var#150/(np.pi/180)**3
    # result = np.zeros((4*mic_num,4*mic_num))
    # result[:4,:4] = np.diag([1,1,1,1])
    # print(result)
    result = np.diag([TDOA_var, DOA_var,DOA_var,DOA_var]*(mic_num))
    return result

def linearize_and_solve_with_H(x,measures,ID,mic_num):
    H = np.zeros((len(x),len(x)))        # 省略二阶偏导的 对原函数求二次导数    280x280
    # H = sparse.lil_matrix(H)
    b = np.zeros((len(x),1))               # 原函数的导数                        280x1
    needToAddPrior = True
    omiga_L = info_matrix(mic_num)
    # omiga_P = np.diag([9,9,9])          # without odometry
    # omiga_P =np.diag([1e4,1e4,1e4])     # with odometry
    omiga_P = np.diag([1111, 1111, 1111])
    # True --> P    False --> L
    P_L = False

    for eid in range(len(measures)):
        # edge = g.edges[eid]
        # edge[0]  "constraint type"
        # edge[1]  "the measurement difference between ith and 1st"
        # edge[2]  "information" standard deviation of Gaussion distribution
        # ID[0]
        # edge[3]  "fromIDX"     Here ,mic is the first node, and  the other is src node
        # ID[1]
        # edge[4]  "toIDX"

        # pose-pose constraint
        if P_L:
            # x1 is the position at t_k-1    (3x1)
            # x2 is the position at t_k      (3x1)
            x1 = x[ID[eid][-2]-1:ID[eid][-2]+2]
            x2 = x[ID[eid][-1]-1:ID[eid][-1]+2]

            # e = x2-x1-measurement
            # computation of A,  de/ds_k-1    the partial of error function for s_k-1
            # computation of B,  de/ds_k    the partial of error function for s_k
            [e,A,B] = linearize_pose_pose_constraint(x1,x2,measures[eid])

            # the first order  partial of error function for unknown param
            # b_i^T = \sum e_ij^T@OMIGA@A_ij
            # b_j^T = \sum e_ij^T@OMIGA@B_ij

            b[ID[eid][-2]-1:ID[eid][-2]+2] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2]+2])+np.transpose(e)@omiga_P@A).T
            b[ID[eid][-1]-1:ID[eid][-1]+2] = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1]+2])+np.transpose(e)@omiga_P@B).T

            # H_ii = \sum A_ij^T @ OMIGA @ A_ij
            # H_ij = \sum A_ij^T @ OMIGA @ B_ij
            # H_ji = \sum B_ij^T @ OMIGA @ A_ij
            # H_jj = \sum B_ij^T @ OMIGA @ B_ij
            # OMIGA IS INFORMATION MATRIX
            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ A.T@omiga_P@A
            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ A.T@omiga_P@B
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ B.T@omiga_P@A
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ B.T@omiga_P@B
            P_L = False

        # pose-landmark constraint(L)
        else:
            x1 = x[ID[eid][-1]- 1:ID[eid][-1] + 2]           # robot node info
            x2 = x[ID[eid][-2]- 1:ID[eid][-2] + 8*mic_num-1]     # mic node info  (positions tau delta)*8
            # e = TDOA_n-1 - Measure
            # computation of A,  de/dl    the partial of error function for landmark     (7X40)
            # computation of B,  de/dx    the partial of error function for src          (7X3)
            [e,A,B] = linearize_pose_landmark_constraint(x1,x2,measures[eid],ID[eid][-1],mic_num)

            # the first order  partial of error function for unknown param
            # b_i^T = \sum e_ij^T @OMIGA @A_ij
            # b_j^T = \sum e_ij^T @OMIGA @B_ij
            # print(A.shape)
            b[ID[eid][-1] - 1:ID[eid][-1] + 2]         = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1] + 2]) + np.transpose(e) @ omiga_L @ B).T
            b[ID[eid][-2] - 1:ID[eid][-2]+8*mic_num-1] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2] +8*mic_num-1]) + np.transpose(e) @ omiga_L @ A).T


            # the second order  partial of error function for unknown param but omit the second order
            # H_ii = \sum A_ij^T @ OMIGA @ A_ij
            # H_ij = \sum A_ij^T @ OMIGA @ B_ij
            # H_ji = \sum B_ij^T @ OMIGA @ A_ij
            # H_jj = \sum B_ij^T @ OMIGA @ B_ij
            # OMIGA IS INFORMATION MATRIX
            H[ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1 , ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1] = H[ID[eid][-2] - 1:ID[eid][-2] + 8*mic_num-1 ,  ID[eid][-2]-1:ID[eid][-2] + 8*mic_num-1] + np.transpose(A) @ omiga_L @ A
            H[ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1 , ID[eid][-1]-1:ID[eid][-1]+2]           = H[ID[eid][-2] - 1:ID[eid][-2] + 8*mic_num-1 ,  ID[eid][-1]-1:ID[eid][-1] + 2]       + np.transpose(A) @ omiga_L @ B
            H[ID[eid][-1]-1:ID[eid][-1]+2           , ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1] = H[ID[eid][-1] - 1:ID[eid][-1] + 2           ,  ID[eid][-2]-1:ID[eid][-2] + 8*mic_num-1] + np.transpose(B) @ omiga_L @ A
            H[ID[eid][-1]-1:ID[eid][-1]+2           , ID[eid][-1]-1:ID[eid][-1]+2]           = H[ID[eid][-1] - 1:ID[eid][-1] + 2           ,  ID[eid][-1]-1:ID[eid][-1] + 2]       + np.transpose(B) @ omiga_L @ B
            P_L = True

    if needToAddPrior:
        # 限制x y z 轴
        # first mic
        H[0:8,0:8] = H[0:8,0:8] + np.eye(8)
    # start = time.time()
    H = sparse.lil_matrix(H)
    dx = inv(H)@(-b)
    # end = time.time()
    # print(end-start)
    return dx,H

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
    fig = ax.plot(x4[:, 0], x4[:, 1], x4[:, 2], c='r')
    if type == "real":
        fig = ax.plot(y4[:, 0], y4[:, 1], y4[:, 2], c='g')
        fig = ax.plot(z4[:, 0], z4[:, 1], z4[:, 2], c='b')
    elif type == "estimate":
        fig = ax.plot(x4[:, 0], x4[:, 1], x4[:, 2], c=(1.00,0.61,0.61))
        fig = ax.plot(y4[:, 0], y4[:, 1], y4[:, 2], c=(0.61,0.90,0.61))
        fig = ax.plot(z4[:, 0], z4[:, 1], z4[:, 2], c=(0.59,0.75,0.95))
    return ax,fig

# def compute_global_error(x,measures,ID,mic_num):
#     Fx = 0
#     P_L = False
#     omiga_L = info_matrix(mic_num)
#     # omiga_P = np.diag([9,9,9])          # without odometry
#     omiga_P = np.diag([1e4, 1e4, 1e4])  # with odometry
#     for eid in range(len(measures)):
#         measurement = np.array(measures[eid])
#         if P_L:
#             x1 = x[ID[eid][-2] - 1:ID[eid][-2] + 2]
#             x2 = x[ID[eid][-1] - 1:ID[eid][-1] + 2]
#             row = measurement.shape[1]
#             measurement = measurement.reshape((row, 1))
#             e_ij = x2 - x1 - measurement
#             e_ls_ij = np.transpose(e_ij)@omiga_P@e_ij
#             Fx = Fx + e_ls_ij
#             P_L = False
#         else:
#             x1 = x[ID[eid][-1] - 1:ID[eid][-1] + 2]  # robot node info
#             l = x[ID[eid][-2] - 1:ID[eid][-2] + 8 * mic_num - 1]  # mic node info  (positions tau delta)*8
#             toidx = ID[eid][-1]
#             e_il =  np.zeros((4*(mic_num),1))
#             measurement = measurement.reshape((mic_num) * 4, 1)
#             # 对每个麦克风求误差
#             e_il[0] = 0
#             e_il[1:4] = x1 / norm(x1) - measurement[1:4]
#             # e = TDOA_n-1 - Measure
#             for n in range(1, mic_num):
#                 del_x = float(x1[0] - l[8 * n])
#                 del_y = float(x1[1] - l[8 * n + 1])
#                 del_z = float(x1[2] - l[8 * n + 2])
#                 distance = sqrt(del_x ** 2 + del_y ** 2 + del_z ** 2)
#                 # TDOA error
#                 e_il[4 * n] = (distance - sqrt(x1[0] ** 2 + x1[1] ** 2 + x1[2] ** 2)) / 340.0 + l[8 * n + 6] + (toidx - 8 * mic_num + 2) / 3.0 * l[8 * n + 7] \
#                            - measurement[4 * n]
#                 # DOA error
#                 R_T = rotation_matrix([float(l[8 * n + 3]), float(l[8 * n + 4]), float(l[8 * n + 5])])
#                 R_T = np.array(R_T)
#                 e_il[4 * (n) + 1:4 * (n) + 4] = R_T @ (x1 - l[8 * n:8 * n + 3]) / distance \
#                                              - measurement[4 * (n) + 1:4 * (n) + 4]
#             e_ls_il = np.transpose(e_il) @ omiga_L @ e_il
#             Fx = Fx+e_ls_il
#             P_L = True
#     return Fx
if  __name__ == "__main__":
    # initial param
    numIterations =50
    epsilon = 1e-5
    data_i =6                # 之前的随机种子数为6 五角星对应的随机种子
    random.seed(data_i)
    np.random.seed(data_i)
    # display_delay_error_on = True
    display_norm_dx_on = True
    fig = False
    mic_num = 8

    ini_delta_mic_pos = np.zeros((0, 1))
    ini_delta_mic_theta = np.zeros((0, 1))
    ini_delta_mic_tau = np.zeros((0, 1))
    ini_delta_mic_delta = np.zeros((0, 1))
    ini_delta_src_pos = np.zeros((0, 1))

    fin_delta_mic_pos = np.zeros((0, 1))
    fin_delta_mic_theta = np.zeros((0, 1))
    fin_delta_mic_tau = np.zeros((0, 1))
    fin_delta_mic_delta = np.zeros((0, 1))
    fin_delta_src_pos = np.zeros((0, 1))

    # x, measures, ID, mic_num, x_gt = get_value()
    # x_ICP = x.copy()
    # norm_list = []

    for i in range(1):
        print("7 the {} iter**********************".format(i))
        x,measures,ID,mic_num,x_gt = get_value()
        x_ICP = x.copy()
        norm_list = []

        # 存储所有初值
        delta = x-x_gt
        for j in range(1,mic_num):
            ini_delta_mic_pos = np.append(ini_delta_mic_pos,delta[j*8:j*8+3])
            ini_delta_mic_theta = np.append(ini_delta_mic_theta,delta[j*8+3:j*8+6])
            ini_delta_mic_tau = np.append(ini_delta_mic_tau,delta[j*8+6])
            ini_delta_mic_delta = np.append(ini_delta_mic_delta, delta[j * 8 + 7])
        ini_delta_src_pos =  np.append(ini_delta_src_pos, delta[8*mic_num:])

        # 非线性优化
        try:
            for Iter in range(numIterations):
                # solve the dx
                [dx,H] = linearize_and_solve_with_H(x,measures,ID,mic_num)
                end = time.time()

                x = x +dx
                for mic in range(1, mic_num):
                    theta = np.append(ini_delta_mic_theta, x[mic_num * 8 + 3:mic_num * 8 + 6])
                    if theta[0] >180:
                        x[mic_num * 8 + 3] = theta[0]-360
                    elif theta[0] <-180:
                        x[mic_num * 8 + 3] = theta[0] + 360
                    if theta[2] >180:
                        x[mic_num * 8 + 5] = theta[0]-360
                    elif theta[2] <-180:
                        x[mic_num * 8 + 5] = theta[0] + 360

                #
                # # compute rotation matrix
                # # 第二个麦克风 ---> ypr 中绕z轴旋转角度 ， 绕y轴旋转角度
                # rot_yaw   = -np.arctan2(g.x[ (g.M_x-1)*5+1], g.x[(g.M_x-1)*5])
                # rot_pitch = np.arctan2( g.x[(g.M_x-1)*5+2], sqrt((g.x[(g.M_x-1)*5])**2+(g.x[(g.M_x-1)*5+1])**2 ))
                # # 将ypr（绕z轴 ， 绕y轴） 转化到 zyx 欧拉角
                # M_half    = transform_matrix_from_trans_ypr(0,0,0,rot_yaw,rot_pitch,0)
                # # 将第三个麦克风坐标归算到理想坐标系下 确定实际平面 xoz
                # M_y_p_hom = M_half@vertical_merge([
                #     g.x[(g.M_y-1)*g.M_x*5 : (g.M_y-1)*g.M_x*5+3],1
                # ])
                # rot_roll  = -np.arctan2(M_y_p_hom[2], M_y_p_hom[1]   )
                #
                # # 4*4 T transform zyx
                # M_transform = transform_matrix_from_trans_ypr(0,0,0,rot_yaw, rot_pitch, rot_roll)
                # # rotate the mic position
                # for n in range(g.M):
                #     n = n+2
                #     # 从第二个麦克风开始
                #     g.x[5*(n-1):5*(n-1)+3] =  (M_transform @ vertical_merge([ g.x[5*(n-1):5*(n-1)+3] , 1 ]))[:3]
                #
                # # rotate the sound source position
                # for n in range(int((len(g.x)-5*g.M)/3)):
                #     # 转换每一个时刻的信息
                #     n = n+1
                #     g.x[5 * g.M+ 3*(n-1) : 5 * g.M+3*(n-1)+3] =  (M_transform @ vertical_merge([ g.x[5*g.M+3*(n-1):5*g.M+3*(n-1)+3] , 1 ]))[:3]
                #
                # if display_delay_error_on:
                #     x_3_error = x[14:mic_num*8-1:8]- x_gt[14:mic_num*8-1:8]
                #     print("estimation error of starting time delay:")
                #     print(x_3_error.T)

                # err = compute_global_error(measures,ID,mic_num)
                norm_dx = norm(dx)
                if norm_dx>100:
                    break
                if display_norm_dx_on:
                    print("total norm(dx) = ", norm_dx)
                    norm_list.append(norm_dx)
                    # print(dx[8:26])
                    # print(dx[:25])

                if  norm_dx<epsilon:
                    break

            if fig:
                fig = plt.figure(figsize=plt.figaspect(1))
                ax= plt.axes(projection='3d')

                # ax = fig.gca()
                ax.set_title("3D")
                ax.set_xlabel("X/m")
                ax.set_ylabel("Y/m")
                ax.set_zlabel("Z/m")
                ax.set_xlim(-1,2)
                ax.set_ylim(-1,2)
                ax.set_zlim(-1,2)

                # theta = np.array([30,45,315])/180*np.pi
                # theta2 = np.array([45,30,100])/180*np.pi
                # trans = [0, 0.2, 0.2]
                # trans2 = [0.2, 0.1, 0.1]

                # 绘制基坐标系
                # array_1
                # x axis
                x1 = np.linspace(0,0.2,num = 2)
                y1 = np.zeros_like(x1)
                z1 = np.zeros_like(x1)
                fig1 = ax.plot(x1, y1, z1, c='r')
                frame_1 = zip(x1,y1,z1)
                [origin_x,ax_vec_x] = list(frame_1)
                # y axis
                y2 = np.linspace(0,0.2,num = 2)
                x2 = np.zeros_like(y2)
                z2 = np.zeros_like(x2)
                fig2 = ax.plot(x2, y2, z2, c='g')
                frame_2 = zip(x2,y2,z2)
                [origin_y,ax_vec_y] = list(frame_2)
                # z axis
                z3 = np.linspace(0,0.2,num = 2)
                x3 = np.zeros_like(z3)
                y3 = np.zeros_like(z3)
                fig3 = ax.plot(x3, y3, z3, c='b')
                frame_3 = zip(x3,y3,z3)
                [origin_z,ax_vec_z] = list(frame_3)

                # 绘制麦克风阵列的真实值和估计值
                for i in range(1,mic_num):
                    # 真实值
                    pos = x_gt[8*i:8*i+3].reshape(3)
                    theta = x_gt[8*i+3:8*i+6].reshape(3)
                    ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos,type = "real")
                    # ax.scatter(x_gt[8*i+6:8*i+3+7].reshape(3))

                    # 估计值
                    pos = x[8 * i:8 * i + 3].reshape(3)
                    theta = x[8 * i + 3:8 * i + 6].reshape(3)
                    # print(theta)
                    ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type = "estimate")

                # 真实值
                # ax.scatter3D(x_gt[8 * mic_num::3],
                #            x_gt[8 * mic_num + 1::3],
                #            x_gt[8 * mic_num + 2::3]
                #            )
                real = ax.plot3D(x_gt[8 * mic_num::3].reshape(-1),
                                 x_gt[8 * mic_num + 1::3].reshape(-1),
                                 x_gt[8 * mic_num + 2::3].reshape(-1),
                                 linewidth=1.0
                                 )

                # 初值
                # ax.scatter(x[8 * mic_num::3],
                #            x[8 * mic_num + 1::3],
                #            x[8 * mic_num + 2::3],
                #            marker="x", c="red"
                #            )
                init = ax.plot3D(x[8 * mic_num::3].reshape(-1),
                                 x[8 * mic_num + 1::3].reshape(-1),
                                 x[8 * mic_num + 2::3].reshape(-1),
                                 marker="x", c="red"
                                 )
                methods = ('g.t.', 'est.')
                color = ['b', 'r']
                legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
                legend_labels = [methods[y] for y in range(2)]
                ax.legend(legend_lines, legend_labels, numpoints=1)
                plt.show()

                plt.plot(range(1,len(norm_list[1:])+1),norm_list[1:])
                plt.show()

                # 绘制time offset
                # 真值
                plt.scatter(range(1, len(x_gt[8 + 6:mic_num * 8:8]) + 1), x_gt[8 + 6:mic_num * 8:8], marker='o', c='blue')
                # 估计值
                plt.scatter(range(1, len(x[8 + 6:mic_num * 8:8]) + 1), x[8 + 6:mic_num * 8:8], marker='x', c='red')
                legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
                legend_labels = [methods[y] for y in range(2)]
                plt.legend(legend_lines, legend_labels, numpoints=1)
                plt.title("Time offset")
                # plt.show()

                # 绘制clock diff
                # 真值
                plt.scatter(range(1, len(x_gt[8 + 7:mic_num * 8:8]) + 1), x_gt[8 + 7:mic_num * 8:8], marker='o', c='blue')
                # 估计值
                plt.scatter(range(1, len(x[8 + 7:mic_num * 8:8]) + 1), x[8 + 7:mic_num * 8:8], marker='x', c='red')
                legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
                legend_labels = [methods[y] for y in range(2)]
                plt.legend(legend_lines, legend_labels, numpoints=1)
                plt.title("clock diff")
                # plt.show()
            # scipy.io.savemat("dataset_10_traj/random_seed{}.mat".format(data_i), {'value': [x_ICP,x,x_gt]})
        except OverflowError:
            plt.show()
            plt.plot(range(1, len(norm_list) + 1), norm_list)
            plt.show()
        if norm(dx) > 100:
            continue
        # 存储所有估计值
        delta = x - x_gt
        for j in range(1, mic_num):
            fin_delta_mic_pos = np.append(fin_delta_mic_pos, delta[j * 8:j * 8 + 3])
            fin_delta_mic_theta = np.append(fin_delta_mic_theta, delta[j * 8 + 3:j * 8 + 6])
            fin_delta_mic_tau = np.append(fin_delta_mic_tau, delta[j * 8 + 6])
            fin_delta_mic_delta = np.append(fin_delta_mic_delta, delta[j * 8 + 7])
        fin_delta_src_pos = np.append(fin_delta_src_pos, delta[8*mic_num:])


    ini_norm_mic_pos = norm(ini_delta_mic_pos) / np.sqrt(len(ini_delta_mic_pos))
    ini_norm_mic_theta = norm(ini_delta_mic_theta) / np.sqrt(len(ini_delta_mic_theta))
    ini_norm_mic_tau = norm(ini_delta_mic_tau) / np.sqrt(len(ini_delta_mic_tau))
    ini_norm_mic_delta = norm(ini_delta_mic_delta) / np.sqrt(len(ini_delta_mic_delta))
    ini_norm_src_pos = norm(ini_delta_src_pos) / np.sqrt(len(ini_delta_src_pos))
    print(ini_norm_mic_pos, ini_norm_mic_theta, ini_norm_mic_tau, ini_norm_mic_delta, ini_norm_src_pos)

    fin_norm_mic_pos = norm(fin_delta_mic_pos) / np.sqrt(len(fin_delta_mic_pos))
    fin_norm_mic_theta = norm(fin_delta_mic_theta) / np.sqrt(len(fin_delta_mic_theta))
    fin_norm_mic_tau = norm(fin_delta_mic_tau) / np.sqrt(len(fin_delta_mic_tau))
    fin_norm_mic_delta = norm(fin_delta_mic_delta) / np.sqrt(len(fin_delta_mic_delta))
    fin_norm_src_pos = norm(fin_delta_src_pos) / np.sqrt(len(fin_delta_src_pos))
    print(fin_norm_mic_pos, fin_norm_mic_theta, fin_norm_mic_tau, fin_norm_mic_delta, fin_norm_src_pos)

