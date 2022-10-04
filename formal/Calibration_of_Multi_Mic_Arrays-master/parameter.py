# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ Time   ：2022-09-16
# @ File   : parameter.py
# This file contains the microphone parameters and some other configuration

import  numpy  as np
def true_value():
    # true values of 8 mic_arrays
    initial_mic_location = np.array([
        [0, 0, 0],
        [0.5,0,0.1],
        [0.5,0.5,0.1],
        [0,0.5,0.1],
        [0,0,0.3],
        [0.5, 0, 0.3],
        [0.5, 0.5, 0.3],
        [0, 0.5, 0.3],
    ])*3
    initial_mic_angle = np.array([
        [0,0,0],
        [45,30,60],
        [30,45,150],
        [45,45,45],
        [15,30,45],
        [120,75,60],
        [45, -30, 65],
        [60, 60, 60],
    ])
    initial_mic_asyn = np.array([
        [0,0],
    [2.36048090e-02,1.03166034e-05],
    [3.96058243e-02,1.54972271e-05],
    [6.65150957e-03,4.01591014e-05],
    [9.17955043e-02,8.00452351e-05],
    [7.65162603e-02,2.21928176e-05],
    [5.36680008e-02,2.76682643e-05],
    [1.72664529e-02,1.06183292e-05]
    ])
    sound_speed = 340
    time_steps = 120
    interval = 1
    init_fig = True
    TDOA_FIG = False

    # init error
    mic_pose_std = 0.3    # unit (m)
    mic_rot_std = 10      # unit (degree)
    max_offset = 0.1      # unit (s)
    max_clock_diff = 1e-4 # unit (s)
    src_pose_std = 0.3    # unit (m)
    # measure error
    TDOA_std =6.66e-5     # unit (s)
    DOA_std  =0.03        # VECTOR
    odo_std  =0.03
    return initial_mic_location,initial_mic_angle,initial_mic_asyn,\
           sound_speed,time_steps,interval,\
           init_fig,TDOA_FIG, \
           mic_pose_std,mic_rot_std,max_offset,max_clock_diff,src_pose_std,\
           TDOA_std,DOA_std,odo_std

def config():
    numIterations = 50
    epsilon = 1e-4
    data_seed = 1         # most of data_seed is 6 in paper
    est_fig = True
    return numIterations,epsilon,data_seed, est_fig
