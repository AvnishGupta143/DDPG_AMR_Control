MIN_BUFFER_SIZE = 1500 
MAX_BUFFER = 100000 

MAX_EPISODES = 50000 
MAX_STEPS = 250
TARGET_UPDATE_RATE = 200
NETWORK_SAVE_RATE = 1000
LEARN_RATE = 20

STATE_DIMENSION = 94
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.5  # m/s
ACTION_W_MAX = 0.5  # rad/s
world = 'world_u'

BATCH_SIZE = 256
ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
GAMMA = 0.99
TAU = 0.05
MODEL = 1
STAGE = 4
ALLOW_REVERSE = True
