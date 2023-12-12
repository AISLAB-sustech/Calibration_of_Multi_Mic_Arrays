## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)


### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).


## 2. Build VINS-Fusion
```
    cd vins/catkin_ws/src
    catkin_make
    source ~/vins/catkin_ws/devel/setup.bash
```
(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

## 3. Run VINS-Fusion and the dataset
Open four terminals, run vins odometry, sound event, rviz and play the bag file respectively.
```
Terminal 1:
roslaunch vins vins_rviz.launch 
Terminal 2:
rosrun vins sound_event
Terminal 3:
rosrun vins vins_node ~/vins/catkin_ws/src/VINS-Fusion/config/realsense_d435i/realsense_stereo_imu_config.yaml
Terminal 4:
rosbag play ROSBAG.bag
```

## 4. DOA & TDOA estimation
run the DOA_CAL.py and TDOA_CAL.m