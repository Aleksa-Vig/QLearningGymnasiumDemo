# IMPORT GYMNASIUM
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1', render_mode='human')
obs = env.reset()

print(obs)
print(env.observation_space)
print(env.action_space)

env.render()

env.close()