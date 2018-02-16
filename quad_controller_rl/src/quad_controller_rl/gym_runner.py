import sys
import gym
from quad_controller_rl.agents.ddpg_keras import DDPGKeras

random_seed = 1234
env = gym.make('MountainCarContinuous-v0' if len(sys.argv) < 2 else sys.argv[1])
agent = DDPGKeras(env, random_seed)
env.seed(random_seed)

for _ in range(10000):
    state = env.reset()
    done = False
    reward = 0.0
    action = agent.step(state, reward, done)

    while not done:
        # env.render()
        state, reward, done, _ = env.step(action)
        action = agent.step(state, reward, done)