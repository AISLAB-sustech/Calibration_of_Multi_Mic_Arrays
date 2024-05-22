# Acknowledgement
# https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.multi_mic.html
# https://github.com/BrownsugarZeer/Multi_SSL

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.wavfile import read,write
from  scipy import  signal
from  scipy.linalg import norm
from bss import Duet
from ssl import doa_detection
from scipy.signal import  correlate,correlation_lags
from scipy.stats import pearsonr
import os
import time
from  scipy.io import loadmat,savemat
from numpy import sin as s
from numpy import cos as c

fig_index = 0
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
def doa_calculate(wav_data,channels=6,fs=16000):
    mask = np.ones(channels, bool)
    data_doa = np.transpose(wav_data)[mask, :]
    duet = Duet(data_doa, n_sources=1, sample_rate=fs, output_all_channels=True)
    estimates = duet()
    estimates = estimates.astype(np.float32)
    doas_angle,doas = doa_detection(torch.from_numpy(estimates))
    for doa in doas_angle:
        print(f"real_azi: {doa[0]: 6.1f}, ele: {doa[1]: 6.1f}")
    return  doas_angle.numpy(),doas.numpy()
def azi_ele_calculate(R,s,p):
    doa_theo = (R.T @ (s - p) / norm(s - p)).reshape(-1)
    r = np.sqrt(doa_theo[0] ** 2 + doa_theo[1] ** 2)
    azi_theo = np.arctan2(doa_theo[1], doa_theo[0]) / np.pi * 180
    ele_theo = np.arctan2(doa_theo[2], r) / np.pi * 180
    # print(f"theo_azi: {azi_theo: 6.1f}, ele: {ele_theo: 6.1f}")
    return  azi_theo,ele_theo,doa_theo
def angle_error_calculate(azi_theo,ele_theo,doa_angle):
    diff = abs(azi_theo - doa_angle[0])
    if diff > 200:
        diff = abs(diff - 360)
    # print(f"azi_err: {diff: 6.1f}, ele: {abs(doa_angle[0,1] - ele_theo): 6.1f}")
    return  diff,abs(doa_angle[1] - ele_theo)
def get_duration(current_time,start_time):
    current_time= current_time[1]+current_time[2]/1e9
    start_time = start_time[1]+start_time[2]/1e9
    return current_time - start_time
def find_start_end(data):
    fs = 16000
    window_length = int(fs * 0.01)
    threshold = 3e12

    f, t, Sxx = signal.spectrogram(data, fs, nperseg=window_length)
    for frame_index, frame in enumerate(Sxx.T):
        if np.max(frame) > threshold:
            start_time = t[frame_index]
            break
    end_time = None
    for frame_index, frame in enumerate(Sxx.T[::-1]):
        if np.max(frame) > threshold:
            end_time = t[-frame_index]
            break
    start_index = int(start_time * fs)
    end_index = int(end_time * fs)

    if start_index is not None and end_index is not None:
        pass
        # print("start index:", start_index)
        # print("end index", end_index)
    else:
        print("fail to index")
    return start_index,end_index
def distant(a,b):
    return norm((b-a))
def load_ori_wav(batch,exp):
    master_wav_add = f"audio/exp_{str(batch)}/master/ubuntu_{str(exp)}.wav"
    server1_wav_add = f"audio/exp_{str(batch)}/server1/server1_{str(exp)}.wav"
    server2_wav_add = f"audio/exp_{str(batch)}/server2/server2_{str(exp)}.wav"
    server3_wav_add = f"audio/exp_{str(batch)}/server3/server3_{str(exp)}.wav"
    if Path(master_wav_add).is_file() and Path(server1_wav_add).is_file() \
            and Path(server2_wav_add).is_file() and Path(server3_wav_add).is_file():
        fs, master_data = read(master_wav_add)
        master_data = master_data[:, [0, 5, 4, 3, 2, 1]]
        fs, server1_data = read(server1_wav_add)
        server1_data = server1_data[:, [0, 5, 4, 3, 2, 1]]
        fs, server2_data = read(server2_wav_add)
        server2_data = server2_data[:, [0, 5, 4, 3, 2, 1]]
        fs, server3_data = read(server3_wav_add)
        server3_data = server3_data[:, [0, 5, 4, 3, 2, 1]]
        return master_data, server1_data, server2_data, server3_data,fs
    else:
        raise FileExistsError("the file path is not correct")

def main():
    batch = 2
    # exp 1: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # exp 2: [1,2,3,4,5,6,7,8,9]
    exp_time = [1]
    p = np.array([0, 0, 0.05]).reshape((-1, 1))
    R0 = rotation_matrix([90, 0,90], type="Not Trans")
    R1 = rotation_matrix([-80, 0, 80], type="Not Trans")
    R2 = rotation_matrix([80, 0,-70], type="Not Trans")
    R3 = rotation_matrix([80, 0, 60], type="Not Trans")
    p0 = np.array([0,0,0.109]).reshape((-1, 1))

    base_address =  f"all_measurements/exp{str(batch)}"
    if not (Path(base_address).is_dir()):
        os.mkdir(base_address)
        os.mkdir(base_address+"/0_SETTING")
        os.mkdir(base_address+"/2_DOA")
        os.mkdir(base_address+"/3_ODO")
        os.mkdir(base_address+"/1_TDOA")

    final_doa_error = np.zeros((0,3))
    final_odo_error = np.zeros((0,3))
    final_angle_error = np.zeros((0,3))
    for exp in (exp_time):
        sound_event = np.load(f"../1_visual_processing/sound_marker/exp_{batch}/sound_marker_{exp}.npy")
        pose_mea = np.load(f"../1_visual_processing/pos_mea/exp_{batch}/pose_{exp}.npy")
        master_data, server1_data, server2_data, server3_data, fs = load_ori_wav(batch,exp)
        channels      = master_data.shape[1]
        peak_count    = len(sound_event) - 1
        start_sample_list = []
        Doa_mea = np.zeros((peak_count, 4, 3))

        for i in range(peak_count):
            sound_record = get_duration(sound_event[i+1,:],sound_event[0,:])
            next_point = int(sound_record*fs)
            next_zone = [int(next_point - 0.4 * fs), int(next_point + 0.4 * fs)]

            max_sig_master = np.argmax(master_data[next_zone[0]:next_zone[1], 0])
            max_sig_server1 = np.argmax(server1_data[next_zone[0]:next_zone[1], 0])
            max_sig_server2 = np.argmax(server2_data[next_zone[0]:next_zone[1], 0])
            max_sig_server3 = np.argmax(server3_data[next_zone[0]:next_zone[1], 0])

            start_sample_list.append([next_zone[0]+max_sig_master,next_zone[0]+max_sig_server1,
                                      next_zone[0]+max_sig_server2,next_zone[0]+max_sig_server3])

        if batch == 1:
            p1 = R1 @ p + np.array([2.2, 0, 0.49]).reshape((-1, 1))
            p2 = R2 @ p + np.array([2.0, 1.8, 0.51]).reshape((-1, 1))
            p3 = R3 @ p + np.array([0, 2.0, 0.73]).reshape((-1, 1))
            if exp in [1,2,3]:
                src_pos = np.array([
                    [0.4, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [1.2, 0.4, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.2, 1.6, 0.2],
                    [0.8, 1.6, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.4, 0.8, 0.2],
                    [0.8, 0.8, 0.2],
                    [0.8, 1.2, 0.2],
                    [1.2, 1.2, 0.2],
                    [1.2, 0.8, 0.2],
                ])
                src_pos = src_pos[::-1, :]
            elif exp in [4,5,6]:
                src_pos = np.array([
                    [1.2, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [0.4, 0.8, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.8, 1.6, 0.2],
                    [1.2, 1.6, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.2, 0.8, 0.2],
                    [1.2, 1.2, 0.2],
                    [0.8, 1.2, 0.2],
                    [0.8, 0.8, 0.2],
                ])
            elif exp in [7,8,9]:
                src_pos = np.array([
                    [1.2, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [0.4, 0.8, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.8, 1.6, 0.2],
                    [1.2, 1.6, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.2, 0.8, 0.2],
                    [1.2, 1.2, 0.2],
                    [0.8, 1.2, 0.2],
                    [0.8, 0.8, 0.2],
                    ])
                src_pos = src_pos[::-1, :]
            elif exp in [10,11,12]:
                src_pos = np.array([
                    [0.4, 0.8, 0.2],
                    [0.8, 0.8, 0.2],
                    [1.2, 0.8, 0.2],
                    [1.2, 1.2, 0.2],
                    [0.8, 1.2, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.4, 1.6, 0.2],
                    [0.8, 1.6, 0.2],
                    [1.2, 1.6, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.2, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [0.4, 0.4, 0.2],
                    ])
                src_pos = src_pos[::-1, :]
            elif exp in [13,14,15]:
                src_pos = np.array([
                    [0.4, 0.8, 0.2],
                    [0.8, 0.8, 0.2],
                    [1.2, 0.8, 0.2],
                    [1.2, 1.2, 0.2],
                    [0.8, 1.2, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.4, 1.6, 0.2],
                    [0.8, 1.6, 0.2],
                    [1.2, 1.6, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.2, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [0.4, 0.4, 0.2],
                    ])
        elif batch == 2:
            if exp in [1,2,3]:
                src_pos =np.array([
                    [0.2, 0.2, 0.2],
                    [0.4, 0.2, 0.2],
                    [0.6, 0.2, 0.2],
                    [0.8, 0.4, 0.2],
                    [0.8, 0.6, 0.2],
                    [0.6, 0.8, 0.2],
                    [0.4, 0.8, 0.2],
                    [0.2, 0.6, 0.2],
                    [0.2, 0.4, 0.2],
                    [0.4, 0.4, 0.2],
                    [0.4, 0.6, 0.2],
                    [0.6, 0.6, 0.2],
                    [0.6, 0.4, 0.2],
                ])
                p1 = R1 @ p + np.array([1.1, 0, 0.51]).reshape((-1, 1))
                p2 = R2 @ p + np.array([1.0,0.9, 0.53]).reshape((-1, 1))
                p3 = R3 @ p + np.array([0,1.0, 0.75]).reshape((-1, 1))
            elif exp in [4,5,6]:
                src_pos =np.array([
                    [0.4, 0.4, 0.2],
                    [0.8, 0.4, 0.2],
                    [1.2, 0.4, 0.2],
                    [1.6, 0.8, 0.2],
                    [1.6, 1.2, 0.2],
                    [1.2, 1.6, 0.2],
                    [0.8, 1.6, 0.2],
                    [0.4, 1.2, 0.2],
                    [0.4, 0.8, 0.2],
                    [0.8, 0.8, 0.2],
                    [0.8, 1.2, 0.2],
                    [1.2, 1.2, 0.2],
                    [1.2, 0.8, 0.2],
                ])
                p1 = R1 @ p + np.array([ 2.2, 0, 0.49]).reshape((-1, 1))
                p2 = R2 @ p + np.array([2.0,1.8, 0.51]).reshape((-1, 1))
                p3 = R3 @ p + np.array([0,2.0, 0.73]).reshape((-1, 1))
            elif exp in [7,8,9]:
                src_pos =  np.array([[0.6,0.6, 0.2],
                         [1.2, 0.6, 0.2],
                         [1.8, 0.6 ,0.2],
                         [2.4, 1.2 ,0.2],
                         [2.4 ,1.8 ,0.2],
                         [1.8 ,2.4 ,0.2],
                         [1.2 ,2.4 ,0.2],
                         [0.6 ,1.8 ,0.2],
                         [0.6 ,1.2 ,0.2],
                         [1.2 ,1.2 ,0.2],
                         [1.2 ,1.8 ,0.2],
                         [1.8 ,1.8 ,0.2],
                         [1.8 ,1.2 ,0.2]])
                p1 = R1 @ p + np.array([ 3.3, 0, 0.49]).reshape((-1, 1))
                p2 = R2 @ p + np.array([3.0,2.7, 0.51]).reshape((-1, 1))
                p3 = R3 @ p + np.array([0,3.0, 0.73]).reshape((-1, 1))
        mic_array = np.zeros((4, 6))
        mic_array[0, :3] = p0.reshape(-1)
        mic_array[0, 3:] = np.array([90, 0, 90])
        mic_array[1, :3] = p1.reshape(-1)
        mic_array[1, 3:] = np.array([-80, 0, 80])
        mic_array[2, :3] = p2.reshape(-1)
        mic_array[2, 3:] = np.array([80, 0, -70])
        mic_array[3, :3] = p3.reshape(-1)
        mic_array[3, 3:] = np.array([80, 0, 60])

        DOA_error = np.zeros_like(Doa_mea)
        angle_error = np.zeros_like(DOA_error)

        for i in range(peak_count):
            print("")
            start_sample = start_sample_list[i]
            print(f"{i} signal----------------------------------")

            zone_master  = [int(start_sample[0]  - 0.3 * fs), int(start_sample[0] + 0.5 * fs)]
            if zone_master[0] <0:
                zone_master[0] = 0
            if zone_master[1] > len(master_data):
                zone_master[1] = len(master_data)-1

            zone_server1 = [int(start_sample[1] - 0.3 * fs), int(start_sample[1] + 0.5 * fs)]
            if zone_server1[0] <0:
                zone_server1[0] = 0
            if zone_server1[1] > len(server1_data):
                zone_server1[1] = len(server1_data)-1

            zone_server2 = [int(start_sample[2] - 0.3* fs), int(start_sample[2] + 0.5 * fs)]
            if zone_server2[0] <0:
                zone_server2[0] = 0
            if zone_server2[1] > len(server2_data):
                zone_server2[1] = len(server2_data)-1

            zone_server3 = [int(start_sample[3] - 0.3* fs), int(start_sample[3] + 0.5 * fs)]
            if zone_server3[0] <0:
                zone_server3[0] = 0
            if zone_server3[1] > len(server3_data):
                zone_server3[1] = len(server3_data)-1

            start_index,end_index = find_start_end(master_data[zone_master[0] :zone_master[1],0])
            master_sound_clip = master_data[zone_master[0]+start_index:zone_master[0]+end_index, :]

            start_index, end_index = find_start_end(server1_data[zone_server1[0]:zone_server1[1],0])
            server1_sound_clip = server1_data[zone_server1[0]+start_index:zone_server1[0]+end_index,:]

            start_index, end_index = find_start_end(server2_data[zone_server2[0]:zone_server2[1], 0])
            server2_sound_clip = server2_data[zone_server2[0]+start_index:zone_server2[0]+end_index,:]

            start_index, end_index = find_start_end(server3_data[zone_server3[0]:zone_server3[1], 0])
            server3_sound_clip = server3_data[zone_server3[0]+start_index:zone_server3[0]+end_index, :]

            # estimate DOA
            S1 = src_pos[i].reshape((-1, 1))
            master_doa_angle,master_doa  = doa_calculate(master_sound_clip)
            azi_theo_1, ele_theo_1, doa_theo_1 = azi_ele_calculate(R0, S1, p0)

            master_doa = master_doa[0]
            master_doa_angle = master_doa_angle[0]

            doa_error_1 = doa_theo_1 - master_doa
            error_1 = angle_error_calculate(azi_theo_1, ele_theo_1, master_doa_angle)
            print("azi_theo_1",azi_theo_1,"ele_theo_1",ele_theo_1)
            print("doa_error_1:",error_1)

            # 2号麦克风阵列
            server1_doa_angle,server1_doa = doa_calculate(server1_sound_clip)
            azi_theo_2, ele_theo_2, doa_theo_2 = azi_ele_calculate(R1, S1, p1)

            server1_doa = server1_doa[0]
            server1_doa_angle = server1_doa_angle[0]

            error_2 = angle_error_calculate(azi_theo_2, ele_theo_2, server1_doa_angle)
            doa_error_2 = doa_theo_2-server1_doa
            print("azi_theo_2",azi_theo_2,"ele_theo_2",ele_theo_2)
            print("doa_error_2:", error_2)

            # MIC 3
            server2_doa_angle,server2_doa = doa_calculate(server2_sound_clip)
            azi_theo_3, ele_theo_3, doa_theo_3 = azi_ele_calculate(R2, S1, p2)

            server2_doa = server2_doa[0]
            server2_doa_angle = server2_doa_angle[0]

            error_3 = angle_error_calculate(azi_theo_3, ele_theo_3, server2_doa_angle)
            doa_error_3 = doa_theo_3 - server2_doa
            print("azi_theo_3", azi_theo_3, "ele_theo_3", ele_theo_3)
            print("doa_error_3:",error_3)

            # MIC 4
            server3_doa_angle,server3_doa = doa_calculate(server3_sound_clip)
            azi_theo_4, ele_theo_4, doa_theo_4 = azi_ele_calculate(R3, S1, p3)

            server3_doa = server3_doa[0]
            server3_doa_angle = server3_doa_angle[0]

            error_4 = angle_error_calculate(azi_theo_4, ele_theo_4, server3_doa_angle)
            doa_error_4 = doa_theo_4 - server3_doa
            print("azi_theo_4", azi_theo_4, "ele_theo_4", ele_theo_4)
            print("doa_error_4:",  error_4)

            Doa_mea[i, 0, :] = master_doa
            Doa_mea[i, 1, :] = server1_doa
            Doa_mea[i, 2, :] = server2_doa
            Doa_mea[i, 3, :] = server3_doa

            DOA_error[i,0,:] = doa_error_1
            DOA_error[i,1,:] = doa_error_2
            DOA_error[i,2,:] = doa_error_3
            DOA_error[i, 3, :] = doa_error_4

            angle_error[i, 0, :2] = error_1
            angle_error[i, 1, :2] = error_2
            angle_error[i, 2, :2] = error_3
            angle_error[i, 3, :2] = error_4

        ODO_mea = np.array([np.array(pose_mea[j + 1] - pose_mea[j]) for j in range(len(pose_mea) - 1)])
        ODO_gt = np.zeros_like(ODO_mea)
        for i in range(len(ODO_gt)):
            ODO_gt[i] = src_pos[i+1]-src_pos[i]

        # np.save(base_address + f"/0_SETTING/MA_{str(exp)}.npy", mic_array)
        # np.save(base_address+f"/0_SETTING/SRC_{str(exp)}.npy", src_pos)
        # np.save(base_address+f"/2_DOA/pattern_{str(exp)}.npy",Doa_mea)
        # np.save(base_address+f"/3_ODO/pattern_{str(exp)}.npy",ODO_mea)
        # savemat(base_address+f"/the pattern {str(exp)} sound seq.mat", {'seq_time': sound_event})
        # DOA_error = DOA_error.reshape((-1,3))
        # final_doa_error = np.vstack((final_doa_error,DOA_error))
        # final_odo_error = np.vstack((final_odo_error,ODO_mea-ODO_gt))
        # final_angle_error = np.vstack((final_angle_error,angle_error.reshape(-1,3)))
    #     print("DOA ERROR")
    #     print(f"mean_error {np.mean(abs(DOA_error),axis=0)}, var_error {1/np.var(abs(DOA_error),axis=0)}")
    #     print("VIO ERROR")
    #     print(f"mean_error {np.mean(abs(ODO_mea-ODO_gt),axis=0)}, var_error {1/np.var(abs(ODO_mea-ODO_gt),axis=0)}")
    #     print("ANGLE ERROR")
    #     print(f"mean_error {np.mean(abs(angle_error),axis=0)}")
    #
    # print("Final DOA ERROR")
    # print(f"mean_error {np.mean(abs(final_doa_error),axis=0)}, var_error {np.var((final_doa_error),axis=0)}")
    # print("Final VIO ERROR")
    # print(f"mean_error {np.mean(abs(final_odo_error),axis=0)}, var_error {np.var((final_odo_error),axis=0)}")
    # print("Final ANGLE ERROR")
    # print(f"mean_error {np.mean(abs(final_angle_error.reshape(-1,3)),axis=0)}, var_error {np.var(final_angle_error.reshape(-1,3),axis=0)}")

if __name__ == '__main__':
    main()
