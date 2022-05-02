# DDPG_AMR_Control

## Description
This repo contains implementation for DDPG based controller for continous mapless navigation for autonomous mobile robot. The agent used here is burger model of ROS turtlebot3.

State Space: The state vector is abstracted from 90-dimensional laser range findings, the previous action, the relative target position (relative dis-
tance and heading) are merged together as a 94-dimensional input vector. The laser range findings are sampled from the raw laser findings between 0 and 360 degrees in a trivial and fixed angle distribution of 4 degrees. The range information is normalized to (0, 1). The 2-dimensional target position is represented in polar coordinates (distance and angle) with respect to the mobile robot coordinate frame.

Action Space: The 2-dimensional action of every time step includes the angular and the linear velocities of the differential mobile robot. To constrain the range of angular velocity in (−0.5, 0.5), a hyperbolic tangent function (tanh) is used as the activation function. Moreover, the range of the linear velocity is constrained in (0, 0.5) through a sigmoid function for forward motion and the range of the linear velocity is constrained in (−0.5, 0.5)
through a tanh function for backward motion. Considering the real dynamics of the robot we clip the angular velocity at 1 radian per second and linear velocity at 0.5 meter per second.

## Dependencies for Running Locally
* ROS Noetic
  * All OSes: [click here for installation instructions](http://wiki.ros.org/Installation/Ubuntu)
* python >= 3.4
* Pytorch
  * Install using "pip install torch==1.6"
* Colorama
  * Install using "pip install colorama==0.4.4"
* Numpy
  * Install using "pip install numpy"
* Wandb - For logging graphs [click here for reading more about it](https://wandb.ai/site)
  * You will need to setup an account on wandb to use there platform. It is completely free for use.
  * Install using "pip instal wandb"

## Basic Build Instructions

1. Clone this repo.

2. Go into the repo directory: `cd DDPG_AMR_Control`

3. Compile the source code: `catkin_make -j4` 

## Run Simulation for stage 1
```
source devel/setup.bash
export TURTLEBOT3_MODEL = burger
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch 
```
<img src="env_images/RL1.png"/>

## Run Simulation for stage 4
```
source devel/setup.bash
export TURTLEBOT3_MODEL = burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch 
```
<img src="env_images/RL2.png"/>

## Run Training

1. Run the simulation using the above steps
2. In a saperate terminal, run
```
cd src/ddpg_control/scripts/
python3 train.py
```

## Run testing

1. Run the simulation using the above steps
2. In a saperate terminal, run
```
cd src/ddpg_control/scripts/
python3 test.py
```

## Code Instructions

1. Change the configs for your own settings for training from src/ddpg_control/scripts/config.py
2. Change the configs for your own settings for testing from src/ddpg_control/scripts/config_test.py
3. Change the stage number in src/ddpg_control/scripts/config.py according to stage you want to use for training
4. Turn the LOAD_PRETRAINED flag to false in src/ddpg_control/scripts/config.py if you want to run dtraining from scratch

