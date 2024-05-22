# -*- coding: utf -8 -*-
# @ Author          : Jiang WANG
# @ File            : Multi_Mic_Arrays_Calibration.py
# @ Acknowledgement : The code partially based on https://github.com/daobilige-su/obs-mic-array-calib provided by Daobilige Su

# solving the nonlinear LS problems adopt Gauss-Newton types of iterations
from scipy import sparse
from math import sqrt
from scipy.sparse.linalg import inv,spsolve
from Initialize  import *
from  utils import *
from  parameter import *
import time
import numpy as np

def linearize_pose_pose_constraint(x1,x2,measurement):
    measurement = np.array(measurement)
    row = measurement.shape[1]
    measurement = measurement.reshape((row,1))
    e = x2-x1-measurement
    A = -np.eye(3)
    B = np.eye(3)
    return e,A,B
def linearize_pose_landmark_constraint(x,l,measurement,toidx,mic_num,interval,record_time):
    e = np.zeros((4*(mic_num),1))
    measurement = np.array(measurement).reshape((mic_num)*4,1)
    e[0] = 0
    e[1:4]= x/norm(x)-measurement[1:4]
    for n in range(1,mic_num):
        del_x = float(x[0] - l[8 * n])
        del_y = float(x[1] - l[8 * n + 1])
        del_z = float(x[2] - l[8 * n + 2])
        distance = sqrt( del_x**2+ del_y**2+ del_z**2)
        # TDOA error
        e[4*n]= (distance- sqrt(x[0]**2+x[1]**2+x[2]**2))/346.0+ l[8*n+6]+record_time[int((toidx-8*mic_num+2)/3.0)-1] * l[8*n+7]- measurement[4*n]
        # DOA error
        R_T = rotation_matrix([float(l[8*n+3]),float(l[8*n+4]),float(l[8*n+5])])
        R_T = np.array(R_T)
        e[4*(n)+1:4*(n)+4] = R_T@(x-l[8*n:8*n+3])/distance-measurement[4*(n)+1:4*(n)+4]

    A = np.zeros((4*(mic_num-1),8))
    B = np.zeros((0,3))

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
            (-del_x / distance) / 346.0,
            (-del_y / distance) / 346.0,
            (-del_z / distance) / 346.0,
            0,
            0,
            0,
            1,
            record_time[int((toidx-8*mic_num+2)/3.0)-1]
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
            ((del_x/distance)  - float(x[0]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/346.0,
            ((del_y/distance)  - float(x[1]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/346.0,
            ((del_z/distance)  - float(x[2]/sqrt( x[0]**2+x[1]**2+x[2]**2 )))/346.0
            ])
        B  = vertical_merge([B, B1, -U])
    A = vertical_merge([np.zeros((4, A.shape[1])), A])
    del_x = float(x[0] )
    del_y = float(x[1] )
    del_z = float(x[2] )
    U = np.array([
            [0.0,0.0,0.0],
            [del_y ** 2 + del_z ** 2, -del_x * del_y, -del_x * del_z],
            [-del_x * del_y, del_x ** 2 + del_z ** 2, -del_y * del_z],
            [-del_x * del_z, -del_y * del_z, del_x ** 2 + del_y ** 2]])/(norm((del_x,del_y,del_z))**3)
    B = vertical_merge([U, B])
    return e, A , B
def covariance_matrix(mic_num,category):
    TDOA_var_inv = 2.3e8 if category=="Simulation" else 2e6
    DOA_var_inv = 1111
    result = np.diag([TDOA_var_inv, DOA_var_inv, DOA_var_inv, DOA_var_inv] * (mic_num))
    # result = np.diag([TDOA_var_inv, 100.0,1000.0, 200.0] * (mic_num))              # exp 1
    return result
def linearize_and_solve_with_H(x,measures,ID,mic_num,interval,category,record_time):
    H = np.zeros((len(x),len(x)))
    b = np.zeros((len(x),1))
    omiga_L = covariance_matrix(mic_num,category)
    omiga_P = np.diag([1111, 1111, 1111])
    # True --> P    False --> L
    P_L = False
    for eid in range(len(measures)):
        # ID[0]  "fromIDX"   ID[1]  "toIDX"
        # pose-pose constraint
        if P_L:
            x1 = x[ID[eid][-2]-1:ID[eid][-2]+2]
            x2 = x[ID[eid][-1]-1:ID[eid][-1]+2]

            # e = x2-x1-measurement
            [e,A,B] = linearize_pose_pose_constraint(x1,x2,measures[eid])
            b[ID[eid][-2]-1:ID[eid][-2]+2] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2]+2])+np.transpose(e)@omiga_P@A).T
            b[ID[eid][-1]-1:ID[eid][-1]+2] = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1]+2])+np.transpose(e)@omiga_P@B).T

            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ A.T@omiga_P@A
            H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-2]-1:ID[eid][-2]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ A.T@omiga_P@B
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-2]-1:ID[eid][-2]+2]+ B.T@omiga_P@A
            H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2] = H[ID[eid][-1]-1:ID[eid][-1]+2,ID[eid][-1]-1:ID[eid][-1]+2]+ B.T@omiga_P@B
            P_L = False

        # pose-landmark constraint(L)
        else:
            x1 = x[ID[eid][-1]- 1:ID[eid][-1] + 2]           # robot node info
            x2 = x[ID[eid][-2]- 1:ID[eid][-2] + 8*mic_num-1]     # mic node info  (positions tau delta)*8
            [e,A,B] = linearize_pose_landmark_constraint(x1,x2,measures[eid],ID[eid][-1],mic_num,interval,record_time)

            b[ID[eid][-1] - 1:ID[eid][-1] + 2]         = np.array(np.transpose(b[ID[eid][-1]-1:ID[eid][-1] + 2]) + np.transpose(e) @ omiga_L @ B).T
            b[ID[eid][-2] - 1:ID[eid][-2]+8*mic_num-1] = np.array(np.transpose(b[ID[eid][-2]-1:ID[eid][-2] +8*mic_num-1]) + np.transpose(e) @ omiga_L @ A).T

            H[ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1 , ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1] = H[ID[eid][-2] - 1:ID[eid][-2] + 8*mic_num-1 ,  ID[eid][-2]-1:ID[eid][-2] + 8*mic_num-1] + np.transpose(A) @ omiga_L @ A
            H[ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1 , ID[eid][-1]-1:ID[eid][-1]+2]           = H[ID[eid][-2] - 1:ID[eid][-2] + 8*mic_num-1 ,  ID[eid][-1]-1:ID[eid][-1] + 2]       + np.transpose(A) @ omiga_L @ B
            H[ID[eid][-1]-1:ID[eid][-1]+2           , ID[eid][-2]-1:ID[eid][-2]+8*mic_num-1] = H[ID[eid][-1] - 1:ID[eid][-1] + 2           ,  ID[eid][-2]-1:ID[eid][-2] + 8*mic_num-1] + np.transpose(B) @ omiga_L @ A
            H[ID[eid][-1]-1:ID[eid][-1]+2           , ID[eid][-1]-1:ID[eid][-1]+2]           = H[ID[eid][-1] - 1:ID[eid][-1] + 2           ,  ID[eid][-1]-1:ID[eid][-1] + 2]       + np.transpose(B) @ omiga_L @ B
            P_L = True

    H[0:8,0:8] = H[0:8,0:8] + np.eye(8)     # Fixed the global frame
    H = sparse.lil_matrix(H)
    dx = inv(H)@(-b)
    return dx,H

if  __name__ == "__main__":
    category,numIterations, epsilon, est_fig,display_norm_dx_on = optimal_config()
    #final  results
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

    #--------------------------------------------------------
    # simulation dataset: preset: 0
    #                     random: 1~10
    # exp 1: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # exp 2: [1,2,3,4,5,6,7,8,9]
    dataset =[0]
    fail_count = 0
    init_cost = []
    calib_cost = []
    for index, i in enumerate(dataset):
        print(f"perform dataset {i}")
        # A. Initial Value Selection
        init_start = time.time()
        x, measures, ID, mic_num, x_gt, interval,record_time  = get_initial_value(i,category)
        init_end = time.time()
        init_cost.append(init_end-init_start)
        x_ICP = x.copy()
        _, x_ICP_global, _ = transform_to_global_frame(x_gt.copy(), x_ICP.copy(), x.copy(), mic_num)

        norm_list = []
        # Store all initial values
        delta = x-x_gt
        constant_vec = np.ones((3, 1))/norm(np.ones((3, 1)))
        for j in range(1,mic_num):
            ini_delta_mic_pos = np.append(ini_delta_mic_pos,delta[j*8:j*8+3])
            theta_est = np.array(x[j * 8 + 3:j * 8 + 6].reshape(-1))
            theta_gt = np.array(x_gt[j * 8 + 3:j * 8 + 6].reshape(-1))
            result = (rotation_matrix(theta_est, type="Not Trans") @ constant_vec).T @ \
                     (rotation_matrix(theta_gt, type="Not Trans") @ constant_vec) / (norm(constant_vec) ** 2)
            result = min(max(result, -1), 1)
            delta_theta = np.arccos(result) / np.pi * 180
            ini_delta_mic_theta = np.append(ini_delta_mic_theta, delta_theta)
            ini_delta_mic_tau = np.append(ini_delta_mic_tau,delta[j*8+6])
            ini_delta_mic_delta = np.append(ini_delta_mic_delta, delta[j * 8 + 7])
        ini_delta_src_pos =  np.append(ini_delta_src_pos, delta[8*mic_num:])
        # B. Error Minimization Procedure
        try:
            for Iter in range(numIterations):
                # solve the dx
                [dx,H] = linearize_and_solve_with_H(x,measures,ID,mic_num,interval,category,record_time)
                x = x +dx
                for mic in range(1, mic_num):
                    theta = x[mic * 8 + 3:mic * 8 + 6].reshape(3)
                    if theta[0] >180:
                        x[mic * 8 + 3] = theta[0] - 360
                    elif theta[0] <-180:
                        x[mic * 8 + 3] = theta[0] + 360
                    if theta[2] >180:
                        x[mic * 8 + 5] = theta[2] - 360
                    elif theta[2] <-180:
                        x[mic * 8 + 5] = theta[2] + 360
                    if theta[1] > 90 or theta[1] < -90:
                        R = rotation_matrix(x[mic * 8 + 3:mic * 8 + 6],type="Not Trans")
                        angle = rotationMatrixToEulerAngles(R)/ np.pi * 180
                        x[mic * 8 + 3:mic * 8 + 6] = angle.reshape((3,1))
                norm_dx = norm(dx)
                if display_norm_dx_on:
                    print("norm(dx) = ", norm_dx)
                    norm_list.append(norm_dx)
                if norm_dx>1e8:
                    fail_count+=1
                    print("THIS ITER FALSE!")
                    break
                _, _, x_global = transform_to_global_frame(x_gt.copy(), x_ICP.copy(), x.copy(), mic_num)
            x_gt, x_ICP, x = transform_to_global_frame(x_gt, x_ICP, x, mic_num)
            if est_fig:
                plot_result(x,x_gt,mic_num,title="Optimization Results")

        except OverflowError:
            fail_count+=1
            print("THIS ITER FALSE!")
        if norm(dx) > 1e8:
            continue
        delta = x - x_gt
        calib_cost.append(time.time()-init_end)

        for j in range(1, mic_num):
            fin_delta_mic_pos = np.append(fin_delta_mic_pos, delta[j * 8:j * 8 + 3])
            theta_est = np.array(x[j * 8 + 3:j * 8 + 6].reshape(-1))
            theta_gt = np.array(x_gt[j * 8 + 3:j * 8 + 6].reshape(-1))
            result = (rotation_matrix(theta_est, type="Not Trans") @ constant_vec).T @ \
                     (rotation_matrix(theta_gt, type="Not Trans") @ constant_vec) / (norm(constant_vec) ** 2)
            delta_theta = np.arccos(result) / np.pi * 180
            fin_delta_mic_theta = np.append(fin_delta_mic_theta, delta_theta)
            fin_delta_mic_tau = np.append(fin_delta_mic_tau, delta[j * 8 + 6])
            fin_delta_mic_delta = np.append(fin_delta_mic_delta, delta[j * 8 + 7])
        fin_delta_src_pos = np.append(fin_delta_src_pos, delta[8*mic_num:])

    print("AVE COST: ", np.mean(np.array(init_cost)),np.mean(np.array(calib_cost)))
    # results
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
    print("fail rate: {}%".format(fail_count/len(dataset)*100))