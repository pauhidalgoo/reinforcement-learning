print('Doing imports...')

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy

print('Imports done!')

# Q-learning

env = gym.make('Taxi-v3')
env.reset()

print('Environment created...')

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)


"""
# Double-Q-learning
# Tries to avoid overestimation /maximization bias  
# Since it is a very small / simple experiment, there's almost no difference between the two
Q1 = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
Q2= np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
"""


# Q is a list that stores for each state, the reward obtained by taking each of the four actions available
print('Q shape:', Q.shape, Q)
gamma = 0.999

# You can change these values to see how they affect the results
lr=0.01
epsilon = 0.2
#epsilon = 1

G = 0
print('Training...')

for episode in range(1, 10001):

    # TODO: Implement the Q-learning algorithm that decides actions using the epsilon-greedy policy
    def random_movement():
        return env.action_space.sample()

    done = False
    state, _ = env.reset()
    while not done:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state]) #  np.argmax((Q1[state]+Q2[state])/2) for double Q learning
            
        next_state, reward, done, _, _ = env.step(action)

        Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        """
        # Double-Q-learning
        if np.random.rand() > 0.5:

            Q1[state, action] += lr * (reward + gamma * Q2[next_state, np.argmax(Q1[next_state])] - Q1[state, action])
        else:
            Q2[state, action] += lr * (reward + gamma * Q1[next_state, np.argmax(Q2[next_state])] - Q2[state, action])

        """
        G += reward
        state = next_state


    # End TODO

    # Every 500 episodes, print the average collected reward during training (stored in variable G)
    if episode % 500 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G/500), 'Epsilon =', epsilon, 'lr =', lr)
        G=0


# Q = Q1 + Q2 / 2 # For double Q learning

print('End training...')




"""
print('Visualizing episodes...')
# Let's test your policy!
for i in range(10):
    state, _  = env.reset()
    env.render()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()
"""

print('Collecting reward of the policy while testing (no exploration)...')
G=0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(1000):
    state, _  = env.reset()
    done = None
    truncated = False
    while not (done or truncated):
        action = np.argmax(Q[state])
        state, reward, done, truncated, info = env.step(action)
        G = G + reward
print('Average reward of the policy while testing:', G/1000)

