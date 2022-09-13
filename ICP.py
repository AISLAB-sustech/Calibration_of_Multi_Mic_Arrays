# create general route

import random
# import jax.numpy as np
import numpy as np
from  numpy.linalg import norm
from numpy import sin as s
from numpy import cos as c
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
import math
from scipy.optimize import leastsq,least_squares
import scipy.io
import turtle

random.seed(1)
np.random.seed(1)
# 每个麦克风b 应该大于0, 如果存在某个麦克风b 小于0 , 则该麦克风 初始值太大
def cal_d_ik(mic_num,init_choices,measure_info,sound_speed,a,x,interval):
    b = []
    for i in range(1,mic_num):
        while True:
            mic_i_b =[]
            for j in range(len(init_choices)+1):
                mic_i_b.append(float(measure_info[2*j][i][0] + a[j]/sound_speed -x[i*8+6]-(j+1)*interval*x[i*8+7] )*sound_speed)
            if np.average(mic_i_b)<=0:
                x_lower = np.random.uniform(0, x[i * 8 + 6])
                x[i * 8 + 6] = np.random.uniform(x_lower, x[i * 8 + 6])
            else:
                break
        # print(x[i*8 + 6])
        b.append(mic_i_b)     # 非参考麦克风在不同时刻距离的集合
    return b,x

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

def ICP(mic_num,random_choices,src_pos,measure_info,a,b):
    p_list = np.zeros((0, 3))
    p_prime_list = np.zeros((0, 3))

    for i in range(len(random_choices) + 1):
        p_i = src_pos[i]
        p_list = np.vstack((p_list, p_i))
        p_prime_i = measure_info[2 * i][mic_num][1:] * b[mic_num-1][i]
        p_prime_list = np.vstack((p_prime_list, p_prime_i))

    # 去质心坐标
    inertial_p = np.array([np.sum(p_list[:, 0]), np.sum(p_list[:, 1]), np.sum(p_list[:, 2])]) / len(p_list)
    inertial_p_prime = np.array(
        [np.sum(p_prime_list[:, 0]), np.sum(p_prime_list[:, 1]), np.sum(p_prime_list[:, 2])]) / len(p_prime_list)

    q_list = p_list - inertial_p
    q_prime_list = p_prime_list - inertial_p_prime

    W = np.zeros((3, 3))
    q_list_T = q_list.T
    for i in range(len(q_list)):
        # print(q_list_T[:,i].reshape((3,1)))
        # print(q_prime_list[i].reshape((1,3)))
        W = W + q_list_T[:, i].reshape((3, 1)) @ q_prime_list[i].reshape((1, 3))

    rank = np.linalg.matrix_rank(W)
    if rank!=3:
        print("W IS NOT FULL RANK")
        raise (ValueError)
    u, s, v = np.linalg.svd(W)
    R = u @ v
    xarr = inertial_p.reshape((3, 1)) - R@inertial_p_prime.reshape((3, 1))
    # print(xarr)
    # print(initial_mic_location)
    angle = rotationMatrixToEulerAngles(R)
    angle = angle/np.pi*180
    # print(initial_mic_angle)
    return xarr,angle

def TDOA(dis_i_k,sound_speed,initial_mic_asyn,time_step,interval):
    result = [float(dis_i_k[i]-dis_i_k[0])/sound_speed+ initial_mic_asyn[i][0]+time_step*interval*initial_mic_asyn[i][1]
               for i in range(0,len(dis_i_k))]
    return result

def DOA(R_T,S,X):
    '''
    :param R: rotation matrix
    :param S: source location
    :param X: array location
    :return: DOA
    '''
    d = R_T@((S-X).reshape((3,1)))/distant(S,X)
    return d

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

def fitting_func(x, p):
    """
    数据拟合所用的函数: y = kx+b
    """
    A, k = p  # 给A,k赋值,其中x作为输入,
    return k*x+A

def fitting_residuals(p, y, x):
    """
    实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
    """
    return 0.5*(y - fitting_func(x, p))**2

def distant_b(a,b):
    return norm((b-a))

def vex2theta(a,b):
    return np.arccos(np.dot(a,b)/(norm(a)*norm(b)))

def solve_b(X,angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,\
     angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD):
    # [angle_AB,distant_AB,angle_BC,distant_BC,angle_CD,distant_CD,\
    #  angle_AD,distant_AD,angle_AC,distant_AC,angle_BD,distant_BD] = param
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
    max_offset = 0.1  # unit (s)
    max_clock_diff = 1e-4  # unit (s)
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
    ])#*np.pi/180
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
    mic_num = len(initial_mic_angle)
    time_steps = 120       #120
    interval = 1
    add_measure_noise =True
    odo_measure = True
    fig = False
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

    # 方向选择 x , y , z
    orient_option = np.array([
                    [1,0,0],
                    [-1,0,0],
                    [0,1,0],
                    [0,-1,0],
                    [0,0,1],
                    [0,0,-1]]
                             )/ 3
    # 起点
    s_k_real = np.array([
                    [0.5, 1 ,0]
                    ])
    init_choices = [0]#[0,4,2,5,0]
    random_choices = init_choices+random.choices([0,1,2,3,4,5],weights=[1,1,1,1,1,1], k=time_steps-1-len(init_choices))
    # np.save("dataset_10_traj/random_np10",random_choices)
    # random_choices = np.load("dataset_10_traj/random_np4.npy")
    ###########################五角星############################
    random_choices = list(range(119))
    # turtle.setup(500, 300, 0, 0)
    # lenth = 1 / 3.0
    # print(lenth)
    # traj = np.zeros((0, 2))
    # while True:
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.right(72)
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.forward(lenth)
    #     traj = np.vstack((traj, np.array(turtle.pos())))
    #     turtle.left(144)
    #     if abs(turtle.pos()) < 1:
    #         break
    #
    # zero_block = np.zeros((30, 1))
    # traj = np.concatenate((traj, zero_block), axis=1)
    # traj2 = traj
    # traj3 = traj
    # traj4 = traj
    # traj = s_k_real + traj
    # for i in range(2):
    #     down_step = traj[-1] + np.array([0, 0, lenth])
    #     traj = np.vstack((traj, down_step))
    # traj2 = traj[-1] + traj2
    # traj = np.vstack((traj, traj2))
    # for i in range(2):
    #     down_step = traj[-1] + np.array([0, 0, lenth])
    #     traj = np.vstack((traj, down_step))
    # traj3 = traj[-1] + traj3
    # traj = np.vstack((traj, traj3))
    # for i in range(2):
    #     down_step = traj[-1] + np.array([0, 0, lenth])
    #     traj = np.vstack((traj, down_step))
    # traj4 = traj[-1] + traj4
    # for i in range(120 - len(traj)):
    #     traj = np.vstack((traj, traj4[i]))
    #
    # traj2 = np.zeros((0, 3))
    # for i in range(len(traj) - 1):
    #     traj2 = np.vstack((traj2, traj[i + 1] - traj[i]))
    # orient_option = traj2
    # turtle.end_fill()
    # turtle.done()
    # np.save("wujaioxing_np",traj2)
    orient_option = np.load("wujaioxing_np.npy")
    ###################################################################


###########################不可观测场景####################################
    # # T 共线
    # s_k_real = np.array([
    #     [0, 0, 0]
    # ])
    # random_choices = random.choices([0,1], k=time_steps-1)
    # # T共面
    # random_choices = random.choices([0,1,4,5], k=time_steps-1)
    # # L共线
    # s_k_real = np.array([
    #     [0.5,0,0.8],
    # ])
    # random_choices = random.choices([4, 5], k=time_steps - 1)
    # THETA Y =90
    # initial_mic_angle[1][1] = 90
    # initial_mic_angle[2][1] = 90
    # initial_mic_angle[3][1] = 90
    # initial_mic_angle[4][1] = 90
    # initial_mic_angle[5][1]=90
    # initial_mic_angle[6][1]=90
    # initial_mic_angle[7][1] = 90
    # s_k_real = np.array([
    #     [0.5, 1, 0.5]
    # ])
    # random_choices = random.choices([0, 1, 2, 3, 4, 5], weights=[1, 1, 1, 1, 1, 1], k=time_steps - 1)

############################################################################
    # 每个麦克风阵列距离该点的距离
    dis_i_k = [[distant(s_k_real[-1],mic) for mic in initial_mic_location]]


    # TDOA  DOA
    # [[TDOA_K2 TDOA_K3],
    # [TDOA_K2 TDOA_K3]]
    T_i_k = [TDOA(dis_i_k[-1],sound_speed,initial_mic_asyn,1,interval)]
    # [[DOA_K2 DOA_K3 ],
    # [ DOA_K2 DOA_K3]]
    d_i_k = [[ DOA(rotation_matrix(initial_mic_angle[i]), s_k_real[-1] ,np.array(initial_mic_location[i]))\
               for i in range(len(initial_mic_angle))]]

    # 生成measurement
    measure_info = [[]]
    ID =[]
    # 将测量信息按照每时刻 TDOA DOA 重新排列 ; 并添加噪声
    for i in range(mic_num):
        measure = np.insert(d_i_k[-1][i],0,T_i_k[-1][i])
        if add_measure_noise:
            TDOA_error = np.random.normal(0, TDOA_std)
            DOA_error = np.random.normal(0, DOA_std)
            noisy = np.array([TDOA_error, DOA_error, DOA_error, DOA_error])
            measure = measure+noisy
        measure_info[-1].append(measure)
    ID.append([1,8*mic_num+1])

    for i in range(len(random_choices)):
        ID.append([mic_num*8+1+3*i, mic_num*8+4+3*i])      # P-P constraint ID
        choice = random_choices[i]
        if odo_measure:
            # 调整P-P measurement 值之后的情况
            # measure_info.append([np.array(orient_option[choice])])   # 不包含噪声
            noisy_odo = np.random.normal(0, odo_std, (3, 1)).reshape(-1)
            # 里程计信息
            odo_info = np.array(orient_option[choice])
            if add_measure_noise:
                odo_info = odo_info + noisy_odo
            measure_info.append([odo_info])
        else:
            measure_info.append([np.zeros((3))])
        next_point = s_k_real[-1] + orient_option[choice]
        dis_i_k.append([distant(next_point, mic) for mic in initial_mic_location])
        T_i_k.append(TDOA(dis_i_k[-1], sound_speed, initial_mic_asyn, i+2, interval))
        d_i_k.append([
            DOA(rotation_matrix(initial_mic_angle[i]), next_point,np.array(initial_mic_location[i])) \
              for i in range(len(initial_mic_angle))
        ])

        # print(d_i_k)
        measure = [np.insert(d_i_k[-1][j], 0, T_i_k[-1][j]) for j in range (mic_num)]
        if add_measure_noise:
            for j in range((mic_num-1)):
                TDOA_error = np.random.normal(0, TDOA_std)    # 6.66e-5
                DOA_error = np.random.normal(0, DOA_std)
                noisy = np.array([TDOA_error, DOA_error, DOA_error, DOA_error])
                measure[j] = measure[j] + noisy
        measure_info.append(measure)
        ID.append([1, mic_num * 8 + 4 + 3 * i])            # P-L constraint ID
        s_k_real = np.append(s_k_real,np.array([next_point]), axis=0)
    # print(s_k_real)
    x_gt = np.zeros((8*mic_num+3*time_steps,1))
    for i in range(mic_num):
        x_gt[i*8  :i*8+3] = initial_mic_location[i].reshape((3,1))
        x_gt[i*8+3:i*8+6] = initial_mic_angle[i].reshape((3,1))
        x_gt[i*8+6:i*8+8] = initial_mic_asyn[i].reshape((2,1))

    for i in range(time_steps):
        x_gt[i*3+8*mic_num:i*3+8*mic_num+3] = s_k_real[i].reshape((3,1))

    # 迭代初值
    x = np.zeros((8*mic_num+3*time_steps,1))
    for i in range(1,mic_num):
        # 初始化 麦克风信息
        # x[i*8:i*8+3]   = x_gt[i*8:i*8+3]+np.random.normal(0,mic_pose_std*3,(3,1))
        # x[i*8+3:i*8+6] = x_gt[i*8+3:i*8+6]+np.random.normal(0,mic_rot_std*3,(3,1))#(0,15*np.pi/180,(3,1))
        x[i*8+6]       =np.random.uniform(0,max_offset)#x_gt[i*8+6]+np.random.normal(0,max_offset/10*3)
        x[i*8+7]       =np.random.uniform(0,max_clock_diff)#x_gt[i*8+7]+np.random.normal(0,max_clock_diff/10*3)

    # 初值 = 真值 + 噪声
    # for i in range(time_steps):
    #     x[i*3+8*mic_num:i*3+8*mic_num+3] = x_gt[i*3+8*mic_num:i*3+8*mic_num+3]+ np.random.normal(0,src_pose_std*3,(3,1))

    ##1. Estimation of sound source
    # print("measure_info[0]",measure_info[0])  # 2i 为TDOA 和 DOA 信息
    # print(np.array(measure_info[1]))  # 2i+1 为Odo 信息
    L = 1/3.0       # 每次移动步长
    # 获取相对坐标点
    src_pos = np.zeros((1,3))
    for i in range(len(random_choices)):
        src_pos = np.vstack((src_pos,src_pos[-1]+measure_info[2*i+1]))

    # 通过mic_1 计算第一个声源节点
    measure_DOA_data = np.array(measure_info[::2])[:, 0][:, 1:]
    # print("measure_DOA_data",measure_DOA_data.shape)
    src_rad = np.arctan2((measure_DOA_data[0:2,1]), (measure_DOA_data[0:2,0]))             # TODO 此处应改变为更加一般性的情况
    # print("src_rad",src_rad)
    # print(src_rad/np.pi*180)
    a1 = L*np.sin(src_rad[1])/np.sin(src_rad[0]-src_rad[1])
    src_init = np.array([a1*np.cos(src_rad[0]),a1*np.sin(src_rad[0]),0])
    src_pos = src_pos+src_init
    # print(measure_info[2 * i][0][0])
    # print(src_pos)

    for j in range(5):
        a = np.array([norm(src_pos[i]) for i in range(len(random_choices)+1)])           # mic_1 到 初始化声源的距离
        src_pos_est_by_doa = np.array([a[i]* measure_DOA_data[i] for i  in range(len(measure_DOA_data))])
        src_pos = (src_pos+src_pos_est_by_doa)/2

    # 获取初始距离
    a = np.array([norm(src_pos[i]) for i in range(len(random_choices)+1)])
    for i in range(time_steps):
        x[i*3+8*mic_num:i*3+8*mic_num+3] = src_pos[i].reshape(3,1)#x_gt[i*3+8*mic_num:i*3+8*mic_num+3] + np.random.normal(0,src_pose_std,(3,1))

    b = []
    # 四棱柱获取各个麦克风轨迹的距离b
    for mic in range(1,mic_num):
        measure_DOA_data = np.array(measure_info[::2])[:, mic][:, 1:]
        mic_i_b = np.array([])
        for i in range(0,time_steps,4):
            if i+4-time_steps>0:
                ori_A, ori_B, ori_C, ori_D = measure_DOA_data[time_steps-4:time_steps]
                point_A, point_B, point_C, point_D = src_pos[time_steps-4:time_steps]
            else:
                ori_A, ori_B, ori_C, ori_D = measure_DOA_data[i:i+4]
                point_A,point_B,point_C,point_D = src_pos[i:i+4]

            param = angle_bottom_edg(ori_A, ori_B, ori_C, ori_D,point_A,point_B,point_C,point_D)
            X0 = [2, 2, 2, 3]
            try:
                h = least_squares(solve_b, X0, args=param,bounds=(0, 10))
                mic_i_b = np.append(mic_i_b, h.x)
            except ValueError:
                point_E,point_F = src_pos[0:2]
                ori_E, ori_F = measure_DOA_data[0:2]
                param = angle_bottom_edg(ori_A, ori_B, ori_E, ori_F, point_A, point_B, point_E, point_F)
                h = least_squares(solve_b, X0, args=param)
                mic_i_b = np.append(mic_i_b, h.x[0:2])
                param = angle_bottom_edg(ori_C, ori_D, ori_E, ori_F, point_C, point_D, point_E, point_F)
                h = least_squares(solve_b, X0, args=param)
                mic_i_b = np.append(mic_i_b, h.x[0:2])
        b.append(mic_i_b)

    # Todo: 添加非整数倍信息处理

    # for i in range(1,mic_num):
    #     print(distant(src_pos[1],initial_mic_location[i]))
    measure_TDOA_data = np.array(measure_info[::2])
    for i in range(1,mic_num):
        xarr, angle = ICP(i, random_choices, src_pos, measure_info, a, b)
        x[i * 8:i * 8 + 3] = xarr
        x[i * 8 + 3:i * 8 + 6] = angle.reshape(3, 1)
        # leastsq
        init_time_offset = x[i*8+6]
        init_clock_diff  = x[i*8+7]
        p0 = np.array([init_time_offset,init_clock_diff])

        # b, x = cal_d_ik(mic_num, random_choices, measure_info, sound_speed, a, x, interval)
        b_i = b[i-1]#np.array([distant_b(xarr,i.reshape(1,3)) for i in src_pos])
        # if j ==0:
        #     plt.plot(np.linspace(1, len(random_choices) + 1, len(random_choices) + 1),
        #                 measure_TDOA_data[:, i][:, 0] - (b_i - a) / sound_speed, label="No least",linewidth=5.0)
        #     plt.legend()
        fx = measure_TDOA_data[:,i][:,0]-(b_i-a)/sound_speed    # 该麦克风关于全部节点的TDOA 信息
        fitting_result = leastsq(fitting_residuals, p0, args=(fx, np.linspace(1,len(random_choices)+1,len(random_choices)+1)))
        x[i * 8 + 6] = fitting_result[0][0]
        x[i * 8 + 7] = fitting_result[0][1]

###############################
        if TDOA_FIG:
            # TDOA 拟合图
            plt.plot(np.linspace(1,len(random_choices)+1,len(random_choices)+1),initial_mic_asyn[i][0]+np.linspace(1,len(random_choices)+1,len(random_choices)+1)*initial_mic_asyn[i][1] , label="True")
            # # plt.scatter(np.linspace(1,len(random_choices)+1,len(random_choices)+1), fx, label="Noise")
            plt.plot(np.linspace(1,len(random_choices)+1,len(random_choices)+1),x[i * 8 + 6]+np.linspace(1,len(random_choices)+1,len(random_choices)+1)*x[i * 8 + 7] , label="EST")
            plt.scatter(np.linspace(1,len(random_choices)+1,len(random_choices)+1), measure_TDOA_data[:,i][:,0]-(b_i-a)/sound_speed, label="$T_i^k-(l_i^k-l_1^k)/c$")
            plt.legend(fontsize = 17)
            # plt.title("the {}-th mic".format(str(i+1)))
            plt.title("the i-th mic",fontsize = 20)
            plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
            plt.xticks(fontproperties='Times New Roman', size=18)
            plt.savefig("the {}-th mic.svg".format(str(i+1)), dpi=750, format="svg")
            plt.show()
            # # print(fx-x[i * 8 + 6]-np.linspace(1,len(random_choices)+1,len(random_choices)+1)*x[i * 8 + 7])
            # # print(angle)
            # # print("initial_mic_asyn",initial_mic_asyn)
            # # print("fitting_result",fitting_result)
            plt.plot(np.linspace(1, len(random_choices) + 1, len(random_choices) + 1),
                     b_i, label="Init", linewidth=5.0)
            plt.plot(np.linspace(1, len(random_choices) + 1, len(random_choices) + 1),[distant_b(src_pos[index_],initial_mic_location[i]) for index_ in range(time_steps)], label="True", linewidth=5.0)
            plt.legend()
            plt.title("the {}-th mic b".format(str(i + 1)))
            plt.show()
##############################
        # print("fx:",len(fx))
        # print("b_i:",len(b_i))
        # print("a_i:", len(a))
        # break

        # x[i * 8:i * 8 + 3] = xarr
        # x[i * 8 + 3:i * 8 + 6] = angle.reshape(3,1)
        # print(angle)

    # print(measure_TDOA_data)
    # print("asyn", initial_mic_asyn)
    # print("location",initial_mic_location)
    # print("angle",initial_mic_angle)

    if fig:
        # 绘图
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = plt.axes(projection='3d')

        # ax = fig.gca()
        ax.set_title("3D")
        ax.set_xlabel("X/m")
        ax.set_ylabel("Y/m")
        ax.set_zlabel("Z/m")
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1,2)
        ax.set_zlim(-1, 2)

        # 绘制基坐标系
        # array_1
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

        # 绘制麦克风阵列的真实值和估计值
        for i in range(1, mic_num):
            # 真实值
            pos = x_gt[8 * i:8 * i + 3].reshape(3)
            theta = x_gt[8 * i + 3:8 * i + 6].reshape(3)
            ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="real")

            # 初值
            pos   = x[8 * i:8 * i + 3].reshape(3)
            theta = x[8 * i + 3:8 * i + 6].reshape(3)
            ax, fig = plot_axis(ax, origin_x, ax_vec_x, origin_y, ax_vec_y, origin_z, ax_vec_z, theta, pos, type="estimate")

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
        methods = ('g.t.', 'init')
        color = ['b','r' ]
        legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
        legend_labels = [methods[y] for y in range(2)]
        ax.legend(legend_lines, legend_labels, numpoints=1)
        plt.show()

        # 绘制time offset
        # 真值
        plt.scatter(range(1, len(x_gt[8+6:mic_num * 8:8]) + 1), x_gt[8+6:mic_num * 8:8], marker='o', c='blue')
        # 初值
        plt.scatter(range(1, len(x[8+6:mic_num * 8:8]) + 1), x[8+6:mic_num * 8:8],marker='x',c='red')
        legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
        legend_labels = [methods[y] for y in range(2)]
        plt.legend(legend_lines, legend_labels, numpoints=1)
        plt.title("Time offset")
        plt.show()

        # 绘制clock diff
        # 真值
        plt.scatter(range(1, len(x_gt[8+7:mic_num * 8:8]) + 1), x_gt[8+7:mic_num * 8:8], marker='o', c='blue')
        # 初值
        plt.scatter(range(1, len(x[8+7:mic_num * 8:8]) + 1), x[8+7:mic_num * 8:8],marker='x',c='red')
        legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in range(2)]
        legend_labels = [methods[y] for y in range(2)]
        plt.legend(legend_lines, legend_labels, numpoints=1)
        plt.title("clock diff")
        plt.show()

    return x,measure_info,ID,mic_num,x_gt

if __name__=="__main__":
    get_value()








