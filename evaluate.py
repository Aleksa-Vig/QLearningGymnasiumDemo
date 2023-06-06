import pickle
import gymnasium as gym
import random
import numpy as np

from utils import select_optimal_action

env = gym.make("Taxi-v3")

with open('q_table.pickle', 'rb') as f:
        q_table = pickle.load(f)

total_epochs, total_penalties = 0, 0

print("Running episodes...")
for episode in range(100):
    state = env.reset()
    stateNumber = state[0]
    epochs, num_penalties, reward = 0, 0, 0

    while reward != 20:
        #find the best action
        next_action = select_optimal_action(q_table,stateNumber)

        #do that action and note the state that was moved too
        stateNumber, reward, _, _ ,_ = env.step(next_action)

        if reward == -10:
            num_penalties += 1

        epochs += 1

        print(f'episode: {episode}, epochs: {epochs}, most current reward: {reward}')

    total_penalties += num_penalties
    total_epochs += epochs

average_time = total_epochs / float(100)
average_penalties = total_penalties / float(100)
print("Evaluation results after {} trials".format(100))
print("Average time steps taken: {}".format(average_time))
print("Average number of penalties incurred: {}".format(average_penalties))