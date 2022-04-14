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

    if is_training:
        var_v = config.ACTION_V_MAX * .5
        var_w = config.ACTION_W_MAX * 2 * .5
    else:
        var_v = config.ACTION_V_MAX * 0.10
        var_w = config.ACTION_W_MAX * 0.10

    print('State Dimensions: ' + str(config.STATE_DIMENSION))
    print('Action Dimensions: ' + str(config.ACTION_DIMENSION))
    print('Action Max: ' + str(config.ACTION_V_MAX) + ' m/s and ' + str(config.ACTION_W_MAX) + ' rad/s')
    memory_buffer = MemoryBuffer(config.MAX_BUFFER)
    trainer = DDPGAgent(config.STATE_DIMENSION, config.ACTION_DIMENSION, config.ACTION_V_MAX,
                        config.ACTION_W_MAX, memory_buffer)
    noise = OUNoise(config.ACTION_DIMENSION, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
    # trainer.load_models(4880)

    rospy.init_node('ddpg_stage_1')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env(action_dim=config.ACTION_DIMENSION)
    before_training = 4

    past_action = np.zeros(config.ACTION_DIMENSION)

    for ep in range(config.MAX_EPISODES):
        done = False
        state = env.reset()
        if is_training and not ep % 10 == 0 and memory_buffer.len >= before_training * config.MAX_STEPS:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            if memory_buffer.len >= before_training * config.MAX_STEPS:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' evaluating')
                print('---------------------------------')
            else:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' adding to memory')
                print('---------------------------------')

        rewards_current_episode = 0.

        for step in range(config.MAX_STEPS):
            state = np.float32(state)

            # if is_training and not ep%10 == 0 and memory_buffer.len >= before_training*config.MAX_STEPS:
            if is_training and not ep % 10 == 0:
                action = trainer.get_exploration_action(state)
                # action[0] = np.clip(
                #     np.random.normal(action[0], var_v), 0., config.ACTION_V_MAX)
                # action[0] = np.clip(np.clip(
                #     action[0] + np.random.uniform(-var_v, var_v), action[0] - var_v, action[0] + var_v), 0., config.ACTION_V_MAX)
                # action[1] = np.clip(
                #     np.random.normal(action[1], var_w), -config.ACTION_W_MAX, config.ACTION_W_MAX)
                N = copy.deepcopy(noise.get_noise(t=step))
                N[0] = N[0] * config.ACTION_V_MAX / 2
                N[1] = N[1] * config.ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], 0., config.ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -config.ACTION_W_MAX, config.ACTION_W_MAX)
            else:
                action = trainer.get_exploration_action(state)

            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done = env.step(action, past_action)
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)

            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep % 10 == 0 or not memory_buffer.len >= before_training * config.MAX_STEPS:
                if reward == 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        memory_buffer.add(state, action, reward, next_state, done)
                # elif reward == -100.:
                #     print('***\n-------- Collision ----------\n****')
                #     for _ in range():
                #         memory_buffer.add(state, action, reward, next_state, done)
                else:
                    memory_buffer.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)

            if memory_buffer.len >= before_training * config.MAX_STEPS and is_training and not ep % 10 == 0:
                # var_v = max([var_v*0.99999, 0.005*config.ACTION_V_MAX])
                # var_w = max([var_w*0.99999, 0.01*config.ACTION_W_MAX])
                trainer.optimizer()

            if done or step == config.MAX_STEPS - 1:
                print('reward per ep: ' + str(rewards_current_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                # print('explore_v: ' + str(var_v) + ' and explore_w: ' + str(var_w))
                print('sigma: ' + str(noise.sigma))
                # rewards_all_episodes.append(rewards_current_episode)
                if not ep % 10 == 0:
                    pass
                else:
                    # if memory_buffer.len >= before_training*config.MAX_STEPS:
                    result = rewards_current_episode
                    pub_result.publish(result)
                break
        if ep % 20 == 0:
            trainer.save_models(ep)
    print('Completed Training')


if __name__ == '__main__':
    run_training()
