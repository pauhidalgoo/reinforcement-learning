import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Sarsa

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
gamma = 0.999

# You can change these values to see how they affect the results
lr=0.01
epsilon = 0.2

G = 0
print('Training...')

for episode in range(1, 10001):

    # TODO: Implement the SARSA algorithm

    state = env.reset()
    done = False
    
    
    action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

    while not done:
        next_state, reward, done, _, _ = env.step(action)

        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])


        
        Q[state, action] += lr * (reward + gamma * (Q[next_state, next_action]) - Q[state, action])


        """
        # Expected Sarsa
        # Same convergence, less variability

        action_probs = np.ones(env.action_space.n) * (epsilon / env.action_space.n)
        best_action = np.argmax(Q[next_state])
        action_probs[best_action] += (1 - epsilon) # epsilon greedy

        Q[state, action] += lr * (reward + gamma * np.dot(action_probs, Q[next_state]) - Q[state, action]) # np.dot fa el mateix que el sumatori en aquest cas
        """
        

        G += reward
        state = next_state
        action = next_action


    # End TODO

    # Every 500 episodes, print the average collected reward during training (stored in variable G)
    if episode % 500 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G/500), 'Epsilon =', epsilon, 'lr =', lr)
        G=0

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
for i in range(1000):
    state = env.reset()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G = G + reward
print('Average reward of the policy while testing:', G/1000)

