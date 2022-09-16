# -*- coding: utf -8 -*-
# @ Author          : Jiang WANG
# @ Time            ：2022-09-16
# @ File            : Multi_Mic_Arrays_Calibration.py
# @ Acknowledgement : The code partially based on https://github.com/daobilige-su/obs-mic-array-calib provided by Daobilige Su

# solving the nonlinear LS problems adopt Gauss-Newton types of iterations
from scipy import sparse
from math import sqrt
from scipy.sparse.linalg import inv,spsolve
import time
from Initialize  import *
from  utils import *
from  parameter import *

def linearize_pose_pose_constraint(x1,x2,measurement):
    """
    :param x1: robot node info (positions at s_k-1)
    :param x2: robot node info (positions at s_k)
    :param measurement: odometry measurements
    :return: errors, Jacobian matrices
    """
    measurement = np.array(measurement)
    row = measurement.shape[1]
    measurement = measurement.reshape((row,1))
    e = x2-x1-measurement
    A = -np.eye(3)
    B = np.eye(3)
    return e,A,B

def linearize_pose_landmark_constraint(x,l,measurement,toidx,mic_num):
    '''
    :param x: robot node info (positions)
    :param l: mic node info (positions, angles, tau, delta)*8
    :param measurement: TDOA and DOA measurement
    :param toidx: robot node
    :return: errors, Jacobian matrices
    '''
    e = np.zeros((4*(mic_num),1))
    measurement = np.array(measurement).reshape((mic_num)*4,1)
    # measurements error
    e[0] = 0
    e[1:4]= x/norm(x)-measurement[1:4]
    for n in range(1,mic_num):
        del_x = float(x[0] - l[8 * n])
        del_y = float(x[1] - l[8 * n + 1])
        del_z = float(x[2] - l[8 * n + 2])
        distance = sqrt( del_x**2+ del_y**2+ del_z**2)
        # TDOA error
        e[4*n]= (distance- sqrt(x[0]**2+x[1]**2+x[2]**2))/340.0+ l[8*n+6]+(toidx-8*mic_num+2)/3.0 * l[8*n+7]- measurement[4*n]
        # DOA error
        R_T = rotation_matrix([float(l[8*n+3]),float(l[8*n+4]),float(l[8*n+5])])
        R_T = np.array(R_T)
        e[4*(n)+1:4*(n)+4] = R_T@(x-l[8*n:8*n+3])/distance-measurement[4*(n)+1:4*(n)+4]

    A = np.zeros((4*(mic_num-1),8))      # computation of A,  de/dl    the partial of error function for landmark
    B = np.zeros((0,3))                  # computation of B,  de/dx    the partial of error function for src

    for n in range(1,mic_num):
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

        A = horizon_merge([ A , vertical_merge([
                                            np.zeros((4*(n-1),8)),
                                            h,
                                            U_V,
                                            np.zeros((4*(mic_num-1-n),8))
                                                ])
                            ])
        # T
        B1 = np.array([
            ((del_x/distance)  - float(x[0]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0,
            ((del_y/distance)  - float(x[1]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0,
            ((del_z/distance)  - float(x[2]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/340.0
            ])
        B  = vertical_merge([B, B1, -U])
    A = vertical_merge([np.zeros((4, A.shape[1])), A])
    del_x = float(x[0] )
    del_y = float(x[1] )
    del_z = float(x[2] )
    U = np.array([
            [0,0,0],
            [del_y ** 2 + del_z ** 2, -del_x * del_y, -del_x * del_z],
            [-del_x * del_y, del_x ** 2 + del_z ** 2, -del_y * del_z],
            [-del_x * del_z, -del_y * del_z, del_x ** 2 + del_y ** 2]])/(norm((del_x,del_y,del_z))**3)
    B = vertical_merge([U, B])
    return e, A , B


def info_matrix(mic_num):
    TDOA_var = 2.3e8
    DOA_var = 1111
    result = np.diag([TDOA_var, DOA_var,DOA_var,DOA_var]*(mic_num))
    return result

def linearize_and_solve_with_H(x,measures,ID,mic_num):
    H = np.zeros((len(x),len(x)))
    b = np.zeros((len(x),1))
    needToAddPrior = True
    omiga_L = info_matrix(mic_num)
    omiga_P = np.diag([1111, 1111, 1111])
    # True --> P    False --> L
    P_L = False

    for eid in range(len(measures)):
        # ID[0]  "fromIDX"   ID[1]  "toIDX"
        # pose-pose constraint
        if P_L:
            # x1 is the position at t_k
            # x2 is the position at t_k+1
            x1 = x[ID[eid][-2]-1:ID[eid][-2]+2]
            x2 = x[ID[eid][-1]-1:ID[eid][-1]+2]

            # e = x2-x1-measurement
            # computation of A,  de/ds_k-1    the partial of error function for s_k
            # computation of B,  de/ds_k    the partial of error function for s_k+1
            [e,A,B] = linearize_pose_pose_constraint(x1,x2,measures[eid])

            # b_i^T = \sum e_ij^T@OMIGA@A_ij
            # b_j^T = \sum e_ij^T@OMIGA@B_ij
            b[ID[eid][-2]-1:ID[eid][-2]+2] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2]+2])+np.transpose(e)@omiga_P@A).T
            b[ID[eid][-1]-1:ID[eid][-1]+2] = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1]+2])+np.transpose(e)@omiga_P@B).T

            # H_ii = \sum A_ij^T @ OMIGA @ A_ij
            # H_ij = \sum A_ij^T @ OMIGA @ B_ij
            # H_ji = \sum B_ij^T @ OMIGA @ A_ij
            # H_jj = \sum B_ij^T @ OMIGA @ B_ij
            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ A.T@omiga_P@A
            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ A.T@omiga_P@B
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ B.T@omiga_P@A
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ B.T@omiga_P@B
            P_L = False

        # pose-landmark constraint(L)
        else:
            x1 = x[ID[eid][-1]- 1:ID[eid][-1] + 2]           # robot node info
            x2 = x[ID[eid][-2]- 1:ID[eid][-2] + 8*mic_num-1]     # mic node info  (positions tau delta)*8
            # computation of A,  de/dl    the partial of error function for landmark
            # computation of B,  de/dx    the partial of error function for src
            [e,A,B] = linearize_pose_landmark_constraint(x1,x2,measures[eid],ID[eid][-1],mic_num)

            # b_i^T = \sum e_ij^T @OMIGA @A_ij
            # b_j^T = \sum e_ij^T @OMIGA @B_ij
            b[ID[eid][-1] - 1:ID[eid][-1] + 2]         = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1] + 2]) + np.transpose(e) @ omiga_L @ B).T
            b[ID[eid][-2] - 1:ID[eid][-2]+8*mic_num-1] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2] +8*mic_num-1]) + np.transpose(e) @ omiga_L @ A).T

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
        H[0:8,0:8] = H[0:8,0:8] + np.eye(8)     # Fixed the global frame
    H = sparse.lil_matrix(H)
    dx = inv(H)@(-b)
    return dx,H

if  __name__ == "__main__":
    numIterations, epsilon, data_seed, est_fig = config()
    random.seed(data_seed)
    np.random.seed(data_seed)
    display_norm_dx_on = True
    mic_num = 8

    # to get the  MONTE CARLO results
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

    # MONTE CARLO , the Minimum is 1
    for i in range(1):
        print("the {} iter MONTE CARLO **********************".format(i))
        # A. Initial Value Selection
        x,measures,ID,mic_num,x_gt = get_value()
        x_ICP = x.copy()
        norm_list = []

        # Store all initial values
        delta = x-x_gt
        for j in range(1,mic_num):
            ini_delta_mic_pos = np.append(ini_delta_mic_pos,delta[j*8:j*8+3])
            ini_delta_mic_theta = np.append(ini_delta_mic_theta,delta[j*8+3:j*8+6])
            ini_delta_mic_tau = np.append(ini_delta_mic_tau,delta[j*8+6])
            ini_delta_mic_delta = np.append(ini_delta_mic_delta, delta[j * 8 + 7])
        ini_delta_src_pos =  np.append(ini_delta_src_pos, delta[8*mic_num:])

        # B. Error Minimization Procedure
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

                norm_dx = norm(dx)
                if norm_dx>100:
                    break
                if display_norm_dx_on:
                    print("norm(dx) = ", norm_dx)
                    norm_list.append(norm_dx)
                if  norm_dx<epsilon:
                    break

            if est_fig:
                plot_result(x,x_gt,mic_num)
            # scipy.io.savemat("dataset_10_traj/random_seed{}.mat".format(data_i), {'value': [x_ICP,x,x_gt]})
        except OverflowError:
            plt.show()
            plt.plot(range(1, len(norm_list) + 1), norm_list)
            plt.show()
        if norm(dx) > 100:
            continue

        delta = x - x_gt
        for j in range(1, mic_num):
            fin_delta_mic_pos = np.append(fin_delta_mic_pos, delta[j * 8:j * 8 + 3])
            fin_delta_mic_theta = np.append(fin_delta_mic_theta, delta[j * 8 + 3:j * 8 + 6])
            fin_delta_mic_tau = np.append(fin_delta_mic_tau, delta[j * 8 + 6])
            fin_delta_mic_delta = np.append(fin_delta_mic_delta, delta[j * 8 + 7])
        fin_delta_src_pos = np.append(fin_delta_src_pos, delta[8*mic_num:])

    # MONTE CARLO results
    ini_norm_mic_pos = norm(ini_delta_mic_pos) / np.sqrt(len(ini_delta_mic_pos))
    ini_norm_mic_theta = norm(ini_delta_mic_theta) / np.sqrt(len(ini_delta_mic_theta))
    ini_norm_mic_tau = norm(ini_delta_mic_tau) / np.sqrt(len(ini_delta_mic_tau))
    ini_norm_mic_delta = norm(ini_delta_mic_delta) / np.sqrt(len(ini_delta_mic_delta))
    ini_norm_src_pos = norm(ini_delta_src_pos) / np.sqrt(len(ini_delta_src_pos))
    print("RMSE of  Initial Values:",ini_norm_mic_pos, ini_norm_mic_theta, ini_norm_mic_tau, ini_norm_mic_delta, ini_norm_src_pos)

    fin_norm_mic_pos = norm(fin_delta_mic_pos) / np.sqrt(len(fin_delta_mic_pos))
    fin_norm_mic_theta = norm(fin_delta_mic_theta) / np.sqrt(len(fin_delta_mic_theta))
    fin_norm_mic_tau = norm(fin_delta_mic_tau) / np.sqrt(len(fin_delta_mic_tau))
    fin_norm_mic_delta = norm(fin_delta_mic_delta) / np.sqrt(len(fin_delta_mic_delta))
    fin_norm_src_pos = norm(fin_delta_src_pos) / np.sqrt(len(fin_delta_src_pos))
    print("RMSE of Estimated Values:",fin_norm_mic_pos, fin_norm_mic_theta, fin_norm_mic_tau, fin_norm_mic_delta, fin_norm_src_pos)

