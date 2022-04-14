#!/usr/bin/env python3
from std_msgs.msg import Float32
from environments import Env
import rospy

def reset_callback(data):
    env.reset()

def step_callback(data):
    a=[data.data,0.1]
    pa=[0,0]
    env.step(a,pa)



rospy.init_node("ddpg_control")
rospy.Subscriber("Reset", Float32, reset_callback)

rospy.Subscriber("Step", Float32, step_callback)

env = Env()
rospy.spin()
