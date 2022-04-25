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
    wandb.define_metric("critic_loss", step_metric = "steps")
    wandb.define_metric("actor_loss", step_metric = "steps")
    wandb.define_metric("average_reward", step_metric = "episodes")
    wandb.define_metric("episode_reward", step_metric = "episodes")

def run_training():
    global pub_average_reward, pub_episode_reward
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
    
    env = Env(action_dim = config.ACTION_DIMENSION)
    rewards_all_episodes = []
    steps = 0
    
    for ep in range(config.MAX_EPISODES):
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
            
            N = copy.deepcopy(noise.get_noise(t = steps))
            N[0] = N[0] * config.ACTION_V_MAX / 2
            N[1] = N[1] * config.ACTION_W_MAX
            if config.ALLOW_REVERSE:
                action[0] = np.clip(action[0] + N[0], -config.ACTION_V_MAX, config.ACTION_V_MAX)
            else:
                action[0] = np.clip(action[0] + N[0], 0.1, config.ACTION_V_MAX)
            action[1] = np.clip(action[1] + N[1], -config.ACTION_W_MAX, config.ACTION_W_MAX)
                
            next_state, reward, done = env.step(action, past_action)
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            memory_buffer.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            
            print("step: {} | reward: {} | done: {} | action: {},{}".format(steps, reward, done, action[0], action[1]))
            
            past_action = copy.deepcopy(action)
            
            if(steps % config.LEARN_RATE == 0 and steps > config.MIN_BUFFER_SIZE):
                for i in range(20):
                    agent.learn()
                    print(f"{Fore.BLUE}-------------------- Agent Learning ---------------{Style.RESET_ALL}")
                    
            if config.MAX_STEPS <= episode_steps:
                done = True
            
            if(steps % config.TARGET_UPDATE_RATE == 0 and steps > config.MIN_BUFFER_SIZE):
                agent.update_target()
                print(f"{Fore.RED}-------------------- Updating Target Networks ---------------{Style.RESET_ALL}")
                
            if (steps % config.NETWORK_SAVE_RATE == 0 and steps > config.MIN_BUFFER_SIZE):
                agent.save_models(steps)
                print(f"{Fore.GREEN}-------------------- SAVING THE MODEL ---------------{Style.RESET_ALL}")
                
            wandb.log({'steps': steps, 'critic_loss': agent.get_critic_loss(), 'actor_loss': agent.get_actor_loss()}, commit=False)

        rewards_all_episodes.append(rewards_current_episode)
        
        # episode_reward_msg = Float32()
        # episode_reward_msg.data = rewards_current_episode
        # pub_episode_reward.publish(episode_reward_msg)
        
        avg_reward = np.mean(rewards_all_episodes[max(0, ep - 100):(ep + 1)])
        # avg_reward_msg = Float32()
        # avg_reward_msg.data = avg_reward
        # pub_average_reward.publish(avg_reward_msg)
        
        wandb.log({"episodes": ep,
                   "episode_reward": rewards_current_episode,
                   "average_reward": avg_reward})
        
        print("------------------------------------- EPISODE END -----------------------------------------".format(ep + 1))
        
    print('Completed Training')
    print("saving...")
    agent.save_models(steps)
    print("saved")

if __name__ == '__main__':
    rospy.init_node('ddpg_train')
    pub_episode_reward = rospy.Publisher('episode_reward', Float32, queue_size=5)
    pub_average_reward = rospy.Publisher('average_reward', Float32, queue_size=5)
    init_plotting()
    run_training()
