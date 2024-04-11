import numpy as np
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from IPython import display

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.append('../utils/')
from networks import Actor, Critic
from replay import ExperienceReplayMemory

    
# Agent
class DDPGTrainer:
    def __init__(self, env, input_dims, alpha=0.001, beta=0.002, gamma=0.99, tau=0.05, 
                 batch_size=256, replay_size=10**6, update_actor_every=2, exploration_period=500, 
                 noise_factor=0.1, agent_name='agent', model_save_path=None, model_load_path=None):
        
        # hyperparameters
        self.alpha = alpha  # actor learning rate
        self.beta = beta    # critic learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update factor
        self.batch_size = batch_size  # training batch size
        self.input_dims = input_dims
        self.exploration_period = exploration_period  # exploration period
        self.training_step_count = 0
        self.update_actor_every = update_actor_every
        self.noise_factor = noise_factor   # exploration noise factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_score = 0
        self.agent_name = agent_name
        self.is_trained = False
        if model_save_path is None:
            self.model_save_path = f'../Data/{agent_name}'
        else:
            self.model_save_path = model_save_path

        # environment
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        # replay buffer memory
        self.memory = ExperienceReplayMemory(replay_size, input_dims, self.n_actions)

        # initialize actor and critic networks
        if model_load_path:
            self.initialize_networks(self.n_actions, checkpoints_dir=model_load_path)
            self.load_model()
        else:
            self.initialize_networks(self.n_actions)
            self.update_target_parameters(tau=1)


    def initialize_networks(self, n_actions, checkpoints_dir=None):
        """
        Initialize actor and critic networks for TD3 agent.
        """
        if checkpoints_dir is None:
            checkpoints_dir=self.model_save_path
            
        self.actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                           name="actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.critic = Critic(state_action_shape=self.input_dims+self.n_actions,
                               name="critic", checkpoints_dir=checkpoints_dir).to(self.device)

        self.target_actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                                  name="target_actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.target_critic = Critic(state_action_shape=self.input_dims+self.n_actions, 
                                      name="target_critic", checkpoints_dir=checkpoints_dir).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.beta)

        self.target_actor_optimizer = optim.Adam(self.target_actor.parameters(), lr=self.alpha)
        self.target_critic_optimizer = optim.Adam(self.target_critic.parameters(), lr=self.beta)
    
    
    def soft_update(self, target_network, source_network, tau):
        """
        Update the weights of a target neural network using a soft update rule according to the formula:
            new_weight = tau * old_weight + (1 - tau) * old_target_weight
            θ′ ← τ θ + (1 −τ )θ′
        """
        target_params = target_network.state_dict()
        source_params = source_network.state_dict()

        for key in source_params:
            target_params[key] = tau * source_params[key] + (1.0 - tau) * target_params[key]

        target_network.load_state_dict(target_params)
        
    def update_target_parameters(self, tau=None):
        """
        Update the weights of the target actor and both target critic networks using soft update rule.
        """
        if tau is None:
            tau = self.tau

        # update weights of the target actor
        self.soft_update(self.target_actor, self.actor, tau)

        # update weights of the first target critic network
        self.soft_update(self.target_critic, self.critic, tau)
        
        
    def select_action(self, observation, train=True):
        """
        Select an action for the agent.
         
        """
        if self.is_trained:
            train = False
        
        state = torch.tensor([observation], dtype=torch.float32).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        if train:
            action += np.random.normal(loc=0, scale=self.noise_factor, size=self.n_actions)
        
        action = np.clip(action, self.min_action, self.max_action)

        return action
    
    
    def optimize_model(self):
        """
        Function for agent learning that implements the TD3 algorithm.

        Randomly sample a batch of past experiences from memory.
        Perform gradient descent on the two critic networks.
        Perform gradient descent on the actor network with a delayed update schedule; 
        the actor is updated once for every two updates of the critic networks.
        """

        # check if there are enough experiences in memory
        if self.memory.size < self.batch_size:
            return

        # sample a random batch of experiences from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # calculate critic network loss
        target_actions = self.target_actor(next_states)
        new_critic_value = self.target_critic(next_states, target_actions).squeeze(1)
        target = rewards + self.gamma * new_critic_value * (1 - dones)
        critic_value = self.critic(states, actions).squeeze(1)
        critic_loss = F.mse_loss(target, critic_value)
        
        # apply gradient descent with the calculated critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Calculate actor network loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Apply gradient descent with the calculated actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update actor/critic target networks weights with soft update rule
        self.update_target_parameters()
        
        
    def ddpg_train(self, n_episodes=1500, opt_steps=64, reward_weights=None, 
                  print_every=100, render_save_path=None, plot_save_path=None):
        
        if render_save_path:
            env = gym.wrappers.RecordVideo(self.env, video_folder=render_save_path, 
                              episode_trigger=lambda t: t % (n_episodes//10) == 0, disable_logger=True)
        else:
            env = self.env
        
        score_history = []
        avg_score_history = []

        for i in tqdm(range(n_episodes), desc='Training..'):
            done = False
            truncated = False
            score = 0
            step = 0

            obs_array = []
            actions_array = []
            next_obs_array = []

            observation, info = env.reset()

            while not (done or truncated):
                current_observation, achieved_goal, desired_goal = observation.values()
                state = np.concatenate((current_observation, achieved_goal, desired_goal))
                # print(state)

                # Choose an action
                action = self.select_action(state)

                # Execute the chosen action in the environment
                next_observation, reward, done, truncated, _ = env.step(np.array(action))
                next_obs, next_achieved_goal, next_desired_goal = next_observation.values()
                next_state = np.concatenate((next_obs, next_achieved_goal, next_desired_goal))
                # print(next_observation)
                
                if reward_weights is not None:
                    features = self.construct_feature_vector(observation).to(self.device)
                    reward_weights = reward_weights.to(self.device)
                    reward = (reward_weights.t()) @ features                 # w^T ⋅ φ
                    
                    #print(reward_weights.t().shape, features.shape, reward.shape)

                # Store experience in the replay buffer
                self.memory.push(state, action, reward, next_state, done)

                obs_array.append(observation)
                actions_array.append(action)
                next_obs_array.append(next_observation)

                observation = next_observation
                if reward_weights is not None:
                    score += reward.cpu().numpy()[0]
                else:
                    score += reward
                step += 1

            # augment replay buffer with HER
            self.her_augmentation(obs_array, actions_array, next_obs_array)

            # train the agent in multiple optimization steps
            for _ in range(opt_steps):
                self.optimize_model()

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            if avg_score > self.best_score:
                self.best_score = avg_score

            if i % print_every==0 and i!=0:
                print(f"Episode: {i} \t Steps: {step} \t Score: {score:.1f} \t Average score: {avg_score:.1f}")
            
            if self.model_save_path and i % (n_episodes//10)==0:
                self.save_model()
                
        # Plot training performance
        self.plot_scores(scores=score_history, avg_scores=avg_score_history, plot_save_path=plot_save_path)

        return score_history, avg_score_history
    
            
    def her_augmentation(self, observations, actions, next_observations, k = 4):
        """
        Augment the agent's replay buffer using Hindsight Experience Replay (HER).

        This function iterates through the provided observations, actions, and next observations,
        performing HER augmentation to create multiple training examples from each experience.
        """
        # augment the replay buffer
        num_samples = len(actions)
        for index in range(num_samples):
            for _ in range(k):
                # sample a future state (observation and goal) from later in the episode
                future_index = np.random.randint(index, num_samples)
                future_observation, future_achieved_goal, _ = next_observations[future_index].values()
                # print(future_achieved_goal)

                # extract current observation and action from the experience
                observation, _, _ = observations[future_index].values()
                
                # create state representation including the future achieved goal (as if it were the intended goal)
                state = torch.tensor(np.concatenate((observation, future_achieved_goal, future_achieved_goal)), 
                                     dtype=torch.float32).to(self.device)

                next_observation, _, _ = next_observations[future_index].values()
                
                # create next state representation with the same goal for consistency
                next_state = torch.tensor(np.concatenate((next_observation, future_achieved_goal, 
                                                          future_achieved_goal)), dtype=torch.float32).to(self.device)

                # extract action from the experience
                action = torch.tensor(actions[future_index], dtype=torch.float32).to(self.device)
                
                # calculate reward based on achieving the future goal from the current state and action
                reward = self.env.unwrapped.compute_reward(future_achieved_goal, future_achieved_goal, 1.0)

                # store augmented experience in buffer
                state = state.cpu().numpy()
                action = action.cpu().numpy()
                next_state = next_state.cpu().numpy()

                self.memory.push(state, action, reward, next_state, True)
                
                
    def construct_feature_vector(self, observation):
        """
        Normalize observation components and construct a feature vector for the given observation.
        """
        # Normalize observation components
        obs = observation['observation']
        achieved_goal = observation['achieved_goal']
        desired_goal = observation['desired_goal']

        normalized_obs = (obs - self.env.observation_space['observation'].low) / \
                         (self.env.observation_space['observation'].high - self.env.observation_space['observation'].low)
        normalized_achieved_goal = (achieved_goal - self.env.observation_space['achieved_goal'].low) / \
                                    (self.env.observation_space['achieved_goal'].high - self.env.observation_space['achieved_goal'].low)
        normalized_desired_goal = (desired_goal - self.env.observation_space['desired_goal'].low) / \
                                   (self.env.observation_space['desired_goal'].high - self.env.observation_space['desired_goal'].low)

        # Construct feature vector
        feature_vector = np.concatenate((normalized_obs, normalized_achieved_goal, normalized_desired_goal))

        return torch.tensor(feature_vector, dtype=torch.float32)
                
                
    def test_model(self, steps, env=None, save_states=False, render_save_path=None, fps=30):
        """
        Run the trained agent in the environment.
        """
        if env is None:
            env = self.env
        episode_score = 0
        state_list = []     # List to store state feature vectors
        
        observation, info = env.reset()
        current_observation, current_achieved_goal, current_desired_goal = observation.values()
        state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))
                        
        if save_states:
            state_list.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
        
        images = []
        done = False
        truncated = False
        
        with torch.inference_mode():
            for i in range(steps):
                if render_save_path:
                    images.append(env.render())

                action = self.select_action(state)

                observation, reward, done, truncated, _ = env.step(np.array(action))
                
                current_observation, current_achieved_goal, current_desired_goal = observation.values()
                state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))

                if save_states:
                    state_list.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                
                episode_score += reward

                if done or truncated:
                    if render_save_path:
                        images.append(env.render())
                    break

        if render_save_path:
            # env.close()
            imageio.mimsave(f'{render_save_path}.gif', images, fps=fps, loop=0)
            with open(f'{render_save_path}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))
                
        if not save_states:
            return episode_score
        else:
            return episode_score, state_list
    
    
    def plot_scores(self, scores, avg_scores, plot_save_path):
        """
        Plot performance of agent.
        """
        plt.figure(figsize=(10,8))
        plt.plot(scores)
        plt.plot(avg_scores)
        plt.title(f'Performance of {self.agent_name}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
                
                
    def save_model(self):
        """
        Save trained models.
        """
        torch.save(self.actor.state_dict(), self.actor.checkpoints_file)
        torch.save(self.critic.state_dict(), self.critic.checkpoints_file)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoints_file)
        torch.save(self.target_critic.state_dict(), self.target_critic.checkpoints_file)

    def load_model(self):
        """
        Load trained models.
        """
        self.is_trained = True
        self.actor.load_state_dict(torch.load(self.actor.checkpoints_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoints_file))
        self.target_actor.load_state_dict(torch.load(self.target_actor.checkpoints_file))
        self.target_critic.load_state_dict(torch.load(self.target_critic.checkpoints_file))