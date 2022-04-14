exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 1000
MAX_BUFFER = 200000
rewards_all_episodes = []

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22  # m/s
ACTION_W_MAX = 2.  # rad/s
world = 'world_u'

BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
