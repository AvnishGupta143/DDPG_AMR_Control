import copy
import numpy as np
import rospy
from std_msgs.msg import Float32
# ---Run agent---#

from noises import OUNoise
from mem_buffer import MemoryBuffer
from environments import Env
from agents import DDPGAgent
import config_test
import config
from colorama import Fore, Style
import wandb

def init_plotting():
    conf = dict(
        learning_rate_actor = config.ACTOR_LR,
        learning_rate_critic = config.CRITIC_LR,
        batch_size = config.BATCH_SIZE,
        architecture = "DDPG",
        infra = "Ubuntu",
        env = "ddpg"
    )

    wandb.init(
        project = "ddpg_turtlebot",
        entity="avnishgupta",
        tags = ["DDPG", "TURTLEBOT", "RL"],
        config = conf,
    )
    
    wandb.define_metric("steps")
    wandb.define_metric("episodes")
    wandb.define_metric("average_reward", step_metric = "episodes")
    wandb.define_metric("episode_reward", step_metric = "episodes")

def run_training():
    print('State Dimensions: ' + str(config.STATE_DIMENSION))
    print('Action Dimensions: ' + str(config.ACTION_DIMENSION))
    print('Action Max: ' + str(config.ACTION_V_MAX) + ' m/s and ' + str(config.ACTION_W_MAX) + ' rad/s')
        
    agent = DDPGAgent(config.STATE_DIMENSION, 
                        config.ACTION_DIMENSION, 
                        config.ACTION_V_MAX,
                        config.ACTION_W_MAX,
                        path_load = config_test.MODEL_LOAD_PATH)
    agent.load_models(config_test.STEPS_TO_LOAD)
    print("--------------- Loaded Model ------------------")
    
    env = Env(action_dim = config.ACTION_DIMENSION)
    rewards_all_episodes = []
    steps = config_test.STEPS_TO_LOAD
    
    for ep in range(config_test.EPISODES):
        print(f"---------------------- EPISODE {ep + 1} --------------------")
        done = False
        state = env.reset()
        rewards_current_episode = 0.0
        past_action = np.zeros(config.ACTION_DIMENSION)
        episode_steps = 0

        while not done:
            steps += 1
            episode_steps += 1
            state = np.float32(state)
            action = agent.get_action(state)
            
            if config.ALLOW_REVERSE:
                action[0] = np.clip(action[0], -config.ACTION_V_MAX, config.ACTION_V_MAX)
            else:
                action[0] = np.clip(action[0], 0.1, config.ACTION_V_MAX)
            action[1] = np.clip(action[1], -config.ACTION_W_MAX, config.ACTION_W_MAX)
            
            next_state, reward, done = env.step(action, past_action)
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            state = copy.deepcopy(next_state)
            
            print("step: {} | reward: {} | done: {} | action: {},{}".format(steps, reward, done, action[0], action[1]))
            
            past_action = copy.deepcopy(action)
            
            if config_test.STEPS <= episode_steps:
                done = True  

        rewards_all_episodes.append(rewards_current_episode)
        
        avg_reward = np.mean(rewards_all_episodes[max(0, ep - 100):(ep + 1)])
        
        wandb.log({"episodes": ep,
                   "episode_reward": rewards_current_episode,
                   "average_reward": avg_reward})
        
        print("------------------------------------- EPISODE END -----------------------------------------".format(ep + 1))
        
    print('Completed Testing')

if __name__ == '__main__':
    rospy.init_node('ddpg_test')
    init_plotting()
    run_training()
