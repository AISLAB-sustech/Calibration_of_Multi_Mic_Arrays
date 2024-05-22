# -*- coding: utf -8 -*-
# @ Author : Jiang WANG
# @ File   : parameter.py
# This file contains the microphone parameters and some other configuration

import  numpy  as np
from utils import  transform_to_mic1_frame,distant
from scipy.io import loadmat

def true_value(dataset, type='Simulation'):
    if type == 'Simulation':
        MIC = np.load("Simulation_dataset/0_SETTING/MIC_CONFIG.npy")
        mic_location_gt = MIC[[0,1,4,6,7], :3]
        mic_angle_gt = MIC[[0,1,4,6,7],3:6]/np.pi*180
        mic_asyn_gt = MIC[[0,1,4,6,7],6:8]
        s_k_real = np.load("Simulation_dataset/0_SETTING/pattern_{}_src.npy".format(dataset))

    elif type == "Real_world":
        MA_data = np.load(f"Real_world_dataset/exp2/0_SETTING/MA_{dataset}.npy")
        mic_location_gt = np.array(MA_data[:, :3])
        mic_angle_gt = np.array(MA_data[:, 3:6])
        mic_asyn_gt = np.zeros((4,2))
        s_k_real = np.load(f"Real_world_dataset/exp2/0_SETTING/SRC_{dataset}.npy")
        mic_angle_gt, mic_location_gt,s_k_real = transform_to_mic1_frame(mic_angle_gt,mic_location_gt,s_k_real)
        tdoa_mea = (loadmat(f"Real_world_dataset/exp2/1_TDOA/TDOA_exp_{dataset}mea.mat")["delay_mean"])[0] / 16000
        for i in range(1, len(mic_location_gt)):
            mic_asyn_gt[i][0] = tdoa_mea[i]-(distant(mic_location_gt[i], s_k_real[0]) - distant(mic_location_gt[0], s_k_real[0])) / 346.0
        mic_asyn_gt[1][1] = 919 / (16000*60*60*8)
        mic_asyn_gt[2][1] = -24 / (16000*60*60*8)
        mic_asyn_gt[3][1] = 581 / (16000*60*60*8)
    return mic_location_gt,mic_angle_gt,mic_asyn_gt, s_k_real

def initial_config(category):
    sound_speed = 346
    interval = 10
    init_fig = False
    TDOA_FIG = False

    # the folowing parameters are only for simulation cases
    # init error
    init_mic_pose_std = 0.2  # unit (m)
    init_mic_rot_std = 10  # unit (degree)
    init_src_pose_std = 0.2  # unit (m)
    max_offset = 0.1  # unit (s)
    max_clock_diff = 1e-4  # unit (s)

    # measure error
    TDOA_std = 6.66e-5  # unit (s)
    DOA_std = 5
    odo_std = 0.03
    return sound_speed,interval,\
           init_fig,TDOA_FIG, \
           init_mic_pose_std,init_mic_rot_std,max_offset,max_clock_diff,init_src_pose_std,\
           TDOA_std,DOA_std,odo_std

def optimal_config():
    category = 'Simulation' #'Simulation' OR "Real_world"
    numIterations = 50 if category=="Simulation" else 100
    epsilon = 1e-6
    est_fig = True
    display_norm_dx_on =  True
    return category,numIterations,epsilon, est_fig,display_norm_dx_on
