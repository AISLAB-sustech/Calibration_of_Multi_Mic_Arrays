#!/usr/bin/env python2
#encoding:utf-8
import rospy
from std_msgs.msg import Header
import numpy as np
from nav_msgs.msg import Odometry
from numpy import sin as s
from numpy import cos as c
import os

class Lane_signal_extract:
    def __init__(self,batch, exp):
        self.batch = batch
        self.dataset = exp
        self.abs_path = os.path.realpath(__file__)
        self.path = self.abs_path.replace("vins/catkin_ws/src/VINS-Fusion/vins_estimator/script/sound_event.py"\
                    , "ROS_data/exp_{}/sound_seq_{}.npy".format(self.batch,self.dataset))
        self.sound_event = np.load(self.path)
        self.sound_event = np.insert(self.sound_event, 0, 0) 
        self.data  = np.zeros((len(self.sound_event),3))
        self.rows  = 0
        self.lane_sub = rospy.Subscriber('/Lane_signal',Header, self.callback,tcp_nodelay=True,queue_size=1)
        self.src_pos_sub = rospy.Subscriber('/vins_estimator/camera_pose',Odometry, self.src_cb,tcp_nodelay=True,queue_size=1)
        self.src_pos = np.zeros((0,3))
        self.src_pos_3 = np.zeros((0,3))
        self.trajectory = np.zeros((0,3))
        self.trajectory_switch = False
        self.pos_lock = True
        self.src_count = 0
        print("***************************************\nNote that if the script does not terminate automatically, rerun it.\n***************************************") 
        print("Waiting for data...")
        self.run()        
    def callback(self,msg):
        if len(self.sound_event)>0:
            seq = self.sound_event[0]
            if msg.seq == seq:
                if seq != 0:
                    self.pos_lock = False
                self.data[self.rows,0] = self.rows
                self.sound_event = np.delete(self.sound_event,[0])
                self.data[self.rows,1] = msg.stamp.secs
                self.data[self.rows,2] = msg.stamp.nsecs
                self.rows +=1                
                rospy.loginfo("get" + str(seq))
        elif len(self.src_pos) == len(self.data)-1:
            if self.batch == 1:
                if self.dataset in [1,2,3]:
                    theta_z = 0
                elif self.dataset in [4,5,6]:
                    theta_z = -np.pi /2
                elif self.dataset in [7,8,9]:
                    theta_z = 0
                elif self.dataset in [10,11,12,13,14,15]:
                    theta_z = np.pi /2
            elif self.batch == 2:
                theta_z = np.pi /2
            R_z = np.array([
                [c(theta_z), -s(theta_z),0],
                [s(theta_z), c(theta_z), 0],
                [0,0,1]
            ])
            for i in range(len(self.src_pos)):
                self.src_pos[i] = np.matmul( R_z.T , self.src_pos[i])
            for i in range(len(self.trajectory)):
                self.trajectory[i] = np.matmul( R_z.T , self.trajectory[i])
            pos_mea_path = self.abs_path.replace("vins/catkin_ws/src/VINS-Fusion/vins_estimator/script/sound_event.py"\
                    , "pos_mea/exp_{}/pose_{}.npy".format(self.batch,self.dataset))
            sound_marker_path = self.abs_path.replace("vins/catkin_ws/src/VINS-Fusion/vins_estimator/script/sound_event.py"\
                    , "sound_marker/exp_{}/sound_marker_{}.npy".format(self.batch,self.dataset))
            # np.save(sound_marker_path.format(self.dataset),self.data)
            # np.save(pos_mea_path.format(self.dataset),self.src_pos)
            rospy.signal_shutdown("Exit due to finish command")

    def src_cb(self,msg):
        position = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]).reshape(-1,3)
        if self.trajectory_switch:
            self.trajectory=np.vstack((self.trajectory,position))
        if not self.pos_lock:
            self.trajectory_switch = True
            if len(self.src_pos_3) < 3:
                self.src_pos_3 = np.vstack((self.src_pos_3,position))
            else:
                self.src_pos = np.vstack((self.src_pos,np.mean(self.src_pos_3,axis=0)))
                self.src_pos_3 = np.zeros((0,3))
                self.pos_lock = True
                self.src_count +=1

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node('listerner', anonymous=True)
    batch = input("Please input dataset batch: (1 or 2)\n")
    exp   = input("Please input the corresponding dataset:\n1 to 15 for batch1, 1 to 9 for batch2\n")
    extractor = Lane_signal_extract(batch, exp)
