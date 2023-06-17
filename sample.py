import gymnasium as gym

env = gym.make('Taxi-v3', render_mode='human')
state = env.reset()

epochs, penalties, reward = 0, 0, 0
terminated = False

while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if reward == -10:
        penalties += 1

    print(f'epoch: {epochs} penality: {penalties} previous reward: {reward}')
    epochs +=1

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print(state[0])