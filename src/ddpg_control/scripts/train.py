import copy
import numpy as np
import rospy
from std_msgs.msg import Float32
# ---Run agent---#

from noises import OUNoise
from mem_buffer import MemoryBuffer
from environments import Env
from agents import DDPGAgent
import config


def run_training():
    is_training = True

    #if is_training:
    #    var_v = config.ACTION_V_MAX * .5
    #    var_w = config.ACTION_W_MAX * 2 * .5
    ##else:
    #    var_v = config.ACTION_V_MAX * 0.10
    #    var_w = config.ACTION_W_MAX * 0.10

    print('State Dimensions: ' + str(config.STATE_DIMENSION))
    print('Action Dimensions: ' + str(config.ACTION_DIMENSION))
    print('Action Max: ' + str(config.ACTION_V_MAX) + ' m/s and ' + str(config.ACTION_W_MAX) + ' rad/s')
    
    memory_buffer = MemoryBuffer(config.MAX_BUFFER)
    
    agent = DDPGAgent(config.STATE_DIMENSION, 
                        config.ACTION_DIMENSION, 
                        config.ACTION_V_MAX,
                        config.ACTION_W_MAX, 
                        memory_buffer)
    
    noise = OUNoise(config.ACTION_DIMENSION, 
                    max_sigma = 0.1, 
                    min_sigma = 0.1, 
                    decay_period = 100000)
                    
    # agent.load_models(4880)
    
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    
    env = Env(action_dim = config.ACTION_DIMENSION)
    
    rewards_all_episodes = []
    
    for ep in range(config.MAX_EPISODES):
        if is_training:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            if memory_buffer.len >= config.WAIT_STEPS:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' evaluating')
                print('---------------------------------')
            else:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' adding to memory')
                print('---------------------------------')
        
        done = False
        state = env.reset()
        rewards_current_episode = 0.0
        past_action = np.zeros(config.ACTION_DIMENSION)

        for step in range(config.MAX_STEPS):
            print("step " , step)
            state = np.float32(state)

            action = agent.get_action(state)
            if is_training: #and not ep % 10 == 0
                N = copy.deepcopy(noise.get_noise(t = step))
                
                N[0] = N[0] * config.ACTION_V_MAX / 2
                N[1] = N[1] * config.ACTION_W_MAX
                
                action[0] = np.clip(action[0] + N[0], 0., config.ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -config.ACTION_W_MAX, config.ACTION_W_MAX)
                
            next_state, reward, done = env.step(action, past_action)
            past_action = copy.deepcopy(action)
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            
#            if not ep % 10 == 0 or not memory_buffer.len >= config.WAIT_STEPS:
#                if reward == 100.:
#                    print('***\n-------- Maximum Reward ----------\n****')
#                    for _ in range(3):
#                        memory_buffer.add(state, action, reward, next_state, done)
#                 elif reward == -100.:
#                     print('***\n-------- Collision ----------\n****')
#                     for _ in range():

            memory_buffer.add(state, action, reward, next_state, done)

#                else:
#                    memory_buffer.add(state, action, reward, next_state, done)

            state = copy.deepcopy(next_state)

            if memory_buffer.len >= config.WAIT_STEPS and is_training:
                agent.optimizer()
                
            if(ep % config.NETWORK_UPDATE == 0):
                agent.update_target()

            if done or step == config.MAX_STEPS - 1:
                print('reward per ep: ' + str(rewards_current_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                print('sigma: ' + str(noise.sigma))
                rewards_all_episodes.append(rewards_current_episode)
                if not ep % 10 == 0:
                    pass
                else:
                    result = rewards_current_episode
                    pub_result.publish(result)
                break
        if ep % config.SAVE_STEPS == 0:
            agent.save_models(ep)
    
    print('Completed Training')


if __name__ == '__main__':
    rospy.init_node('ddpg_stage_1')
    run_training()
