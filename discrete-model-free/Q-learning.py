import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Q-learning

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)


"""
# Double-Q-learning
# Tries to avoid overestimation /maximization bias  
# Since it is a very small / simple experiment, there's almost no difference between the two
Q1 = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
Q2= np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
"""
gamma = 0.999

# You can change these values to see how they affect the results
lr=0.01 # lr 0.05 is better
epsilon = 0.2
epsilon = 1

G = 0
print('Training...')

for episode in range(1, 50001):

    # TODO: Implement the Q-learning algorithm that decides actions using the epsilon-greedy policy

    state = env.reset()
    done = False

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


print('Visualizing policy...')
# Create and print policy and V for valid states
policy = np.zeros((env.world_row,env.world_col))
V = np.zeros((env.world_row,env.world_col))
policy[0,3]=-1  # Special value for terminal state
policy[1,3]=-1  # Special value for terminal state
policy[1,1]=-1  # Not defined for non-valid states

state = np.nditer(env.map, flags=['multi_index'])
while not state.finished:
    if env.map[state.multi_index]==0:
        policy[state.multi_index] = np.argmax(Q[env.cell_id(state.multi_index)])
        V[state.multi_index]= np.max(Q[env.cell_id(state.multi_index)])
    state.iternext()

print_policy(policy,V)
print(policy)
print(V)

print('Visualizing episodes...')
# Let's test your policy!
for i in range(10):
    state = env.reset()
    env.render()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()


print('Collecting reward of the policy while testing (no exploration)...')
G=0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(10000):
    state = env.reset()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G = G + reward
print('Average reward of the policy while testing:', G/10000)

