
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from replaybuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import itertools


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3:
    """The TD3 Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=100, gamma=0.99): # OLd was batch_size 32
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.      
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_freq = 2 # with 3 i found it to be better?

        # Initialize the Actor and Critic networks. 
        
        # Initialize Critic network and target network. Should be named self.Critic 

        # TWINs
        self.Critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic1_target, self.Critic1)
        copy_target(self.Critic2_target, self.Critic2)

        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor_target, self.Actor)

        # Define the optimizers for the actor and critic networks as proposed in the paper

        q_params = itertools.chain(self.Critic1.parameters(), self.Critic2.parameters())
        self.optim_dqn = optim.Adam(q_params, lr=0.001) # TD3 doesn't seem to have weight decay 
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.001) 



    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps
        

        # We use here OUNoise instead of Gaussian to add some exploration to the agent. OU noise is a stochastic process
        # that generates a random sample from a Gaussian distribution whose value at time t depends on the previous value
        # x(t) and the time elapsed since the previous value y(t). It helps to explore the environment better than Gaussian noise.
        # This line initializes the noise with mean 0 and sigma 0.15 (see Noise.py file)

        # CHANGE: CLIPPED GAUSSIAN
        noise = NormalActionNoise(mean=np.zeros(self.act_dim), sigma=0.1)
        

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            epsilon= noise.sample()

            # NOISE IS ALSO CLIPPED HERE
            # epsilon = np.clip(epsilon, -0.5, 0.5) # In OPENAI SPINNINGUP this noise (ACTION NOISE) isn't clipped, it is the target noise that it is

            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)

                # Get the batch data
                batch = self.replay_buffer.get(self.batch_size)


                
                self.optim_dqn.zero_grad()
                # Compute the loss for the critic and update the critic network
                critic_loss = self.compute_critic_loss(batch)
                critic_loss.backward()
                self.optim_dqn.step()




                if timestep % self.policy_freq == 0:
                    for p in self.Critic1.parameters(): # INCLUDED IN OPENAI IMPLEMENTATION
                        # Freeze Q-network so you don't waste computational effort 
                        # computing gradients for it during the policy learning step.
                        p.requires_grad = False # 
                    
                    for p in self.Critic2.parameters():
                        p.requires_grad = False # 


                    # Compute the loss for the actor and update the actor network 
                    self.optim_actor.zero_grad()
                    actor_loss = self.compute_actor_loss(batch)
                    actor_loss.backward()
                    self.optim_actor.step()

                    for p in self.Critic1.parameters():
                        p.requires_grad = True
                    for p in self.Critic2.parameters():
                        p.requires_grad = True


            # Sync the target networks with soft updates and tau=0.001 according to details of the TD3 paper

            # In TD3 implementation, tau is tau=0.005
            with torch.no_grad():
                soft_update(self.Critic1_target, self.Critic1, 0.005)
                soft_update(self.Critic2_target, self.Critic2, 0.005)
                soft_update(self.Actor_target, self.Actor, 0.005)


            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-5:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        #Implement the function to choose an action given a state. It is deterministic because exploration is added
        # by the OrnsteinUhlenbeckActionNoise in the main loop.
        a = self.Actor(torch.FloatTensor(s).to(device)).cpu().detach().numpy()
        return a


    def compute_critic_loss(self, batch):
        """
        The function computes the critic loss using the Mean Squared Bellman Error (MSBE) calculation.
        
        :param batch: The `batch` parameter is a tuple containing the data for computing the loss.
        :return: the critic loss, which is calculated using the mean squared error (MSE) loss between
        the expected Q-values (q_expected) and the target Q-values (target).
        """
    

        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch # self.replay_buffer.get(self.batch_size)

        # Move data to Tensor and also to device to take profit of GPU if available
        state_batch = torch.FloatTensor(state_batch).to(device) # o
        action_batch = torch.Tensor(action_batch).to(device) #.unsqueeze(1) # a
        next_state_batch = torch.FloatTensor(next_state_batch).to(device) # o2
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1) # r
        terminated_batch = torch.FloatTensor(terminated_batch).to(device).unsqueeze(1) #d
        truncated_batch = torch.FloatTensor(truncated_batch).to(device).unsqueeze(1) #d

        noise = NormalActionNoise(mean=np.zeros(self.act_dim), sigma=0.2) # Target noise: 0.2



        # Compute the Q-values for the state_batch according to the DQN network
        q_expected1 = self.Critic1(state_batch, action_batch) # q
        q_expected2 = self.Critic2(state_batch, action_batch) # q

        


        with torch.no_grad():
            next_action = self.Actor_target(next_state_batch).to(device) # ac_targ.pi(o2)
            

            epsilon= noise.sample()

            # NOISE IS ALSO CLIPPED HERE
            epsilon = np.clip(epsilon, -0.5, 0.5) # In OPENAI SPINNINGUP and the official repo, they use 0.5

            next_action = torch.as_tensor(np.clip(next_action + epsilon, -1, 1), dtype=torch.float32)

            # Compute the Q-values for the next_state_batch to compute the target
            #q_targets_next = self.Critic_target(next_state_batch, next_action)   # Q pi targ

            target_Q1 = self.Critic1_target(next_state_batch, next_action)
            target_Q2 = self.Critic2_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            #q_targets_next = torch.max(self.dqn_target_net(next_state_batch).detach(),dim=1,keepdim = True)[0]                                              # Standard DQN

            # Compute targets. Target for Q(s,a) is standard but when episode terminates target should be only the reward.
            target = reward_batch + (1-(terminated_batch)) * (1-(truncated_batch)) *self.gamma*target_Q



        # Compute the MSE loss between q_expected and target
        criterion = nn.MSELoss()

        #print("q_expected shape", q_expected.shape)
        #print("target shape", target.shape)

        # Target is the same for both: the pessimistic
        loss1 = criterion(q_expected1, target)
        loss2 = criterion(q_expected2, target)

        loss = loss1 + loss2


        return loss
    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # implement the actor loss. You have to sample from the replay buffer first a set of states.

        state_batch, _, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch # self.replay_buffer.get(self.batch_size)

        # Move data to Tensor and also to device to take profit of GPU if available
        state_batch = torch.FloatTensor(state_batch).to(device)
        
        action_batch = self.Actor(state_batch) # ac.pi(o)
        loss = - torch.mean(self.Critic1(state_batch, action_batch)) # Critic target, q_pi


        # END TODO (5) 

        return loss



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)




if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v2",continuous=True, render_mode='rgb_array')

    td3 = TD3(env,replay_size=1000000, batch_size=100, gamma=0.99)

    td3.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v2",continuous=True, render_mode='rgb_array'),'video', episode_trigger=lambda x: True)
    video_agent(env, td3,n_episodes=5)  
    pass
