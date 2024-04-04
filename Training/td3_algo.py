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

from networks import Actor, Critic
from replay import ExperienceReplayMemory

    
# Agent
class TD3Trainer:
    def __init__(self, env, input_dims, alpha=0.001, beta=0.002, gamma=0.99, tau=0.05, 
                 batch_size=256, replay_size=10**6, update_actor_every=2, warmup=500, 
                 noise_factor=0.1, agent_name='agent', model_save_path=None):
        
        # hyperparameters
        self.alpha = alpha  # actor learning rate
        self.beta = beta    # critic learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update factor
        self.batch_size = batch_size  # training batch size
        self.time_step = 0
        self.input_dims = input_dims
        self.warmup = warmup  # exploration period
        self.training_step_count = 0
        self.update_actor_every = update_actor_every
        self.noise_factor = noise_factor   #exploration noise factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_score = 0
        self.agent_name = agent_name
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
        self.initialize_networks(self.n_actions)
        self.update_target_parameters(tau=1)


    def initialize_networks(self, n_actions):
        """
        Initialize actor and critic networks for TD3 agent
        """
        model = "TD3"
        self.actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                           name="actor", checkpoints_dir=self.model_save_path).to(self.device)
        self.critic_1 = Critic(state_action_shape=self.input_dims+self.n_actions,
                               name="critic_1", checkpoints_dir=self.model_save_path).to(self.device)
        self.critic_2 = Critic(state_action_shape=self.input_dims+self.n_actions,
                               name="critic_2", checkpoints_dir=self.model_save_path).to(self.device)

        self.target_actor = Actor(state_shape=self.input_dims, num_actions=n_actions, 
                                  name="target_actor", checkpoints_dir=self.model_save_path).to(self.device)
        self.target_critic_1 = Critic(state_action_shape=self.input_dims+self.n_actions, 
                                      name="target_critic_1", checkpoints_dir=self.model_save_path).to(self.device)
        self.target_critic_2 = Critic(state_action_shape=self.input_dims+self.n_actions, 
                                      name="target_critic_2", checkpoints_dir=self.model_save_path).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.beta)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.beta)

        self.target_actor_optimizer = optim.Adam(self.target_actor.parameters(), lr=self.alpha)
        self.target_critic_1_optimizer = optim.Adam(self.target_critic_1.parameters(), lr=self.beta)
        self.target_critic_2_optimizer = optim.Adam(self.target_critic_2.parameters(), lr=self.beta)
    
    
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
        self.soft_update(self.target_critic_1, self.critic_1, tau)

        # update weights of the second target critic network
        self.soft_update(self.target_critic_2, self.critic_2, tau)
        
        
    def select_action(self, observation):
        """
        Select an action for the agent.
         
        """
        # Selects random action to promote exploration for the warmup period
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise_factor, size=(self.n_actions,))
        else:
            state = torch.tensor([observation], dtype=torch.float32).to(self.device)
            mu = self.actor(state).detach().cpu().numpy()[0]
            
        mu_star = mu + np.random.normal(scale=self.noise_factor, size=self.n_actions)
        mu_star = np.clip(mu_star, self.min_action, self.max_action)
        self.time_step += 1

        return mu_star
    
    
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

        # apply gradient descent on the two critic networks
        target_actions = self.target_actor(next_states) + torch.clamp(torch.randn_like(actions) * 0.2, -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action, self.max_action)

        with torch.no_grad():
            q1_new = self.target_critic_1(next_states, target_actions).squeeze(1)
            q2_new = self.target_critic_2(next_states, target_actions).squeeze(1)
            target = rewards + self.gamma * torch.min(q1_new, q2_new) * (1 - dones)

        q1 = self.critic_1(states, actions).squeeze(1)
        q2 = self.critic_2(states, actions).squeeze(1)
        
        # critic loss
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)

        # gradient descent
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # update the actor network only once for every two updates of critic networks
        self.training_step_count += 1
        if self.training_step_count % self.update_actor_every != 0:
            return

        actor_loss = -torch.mean(self.critic_1(states, self.actor(states)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update actor/critic target networks weights with soft update rule
        self.update_target_parameters()
        
        
    def td3_train(self, n_episodes=1500, opt_steps=64, print_every=100, render_save_path=None, plot_save_path=None):
        
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
                next_obs, next_achgoal, next_desgoal = next_observation.values()
                next_state = np.concatenate((next_obs, next_achgoal, next_desgoal))
                # print(next_observation)

                # Store experience in the replay buffer
                self.memory.push(state, action, reward, next_state, done)

                obs_array.append(observation)
                actions_array.append(action)
                next_obs_array.append(next_observation)

                observation = next_observation
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

            if i % print_every==0:
                print(f"Episode: {i} \t Steps: {step} \t Score: {score:.1f} \t Average score: {avg_score:.1f}")
            
            if self.model_save_path and i % (n_episodes//10)==0:
                self.save_model()
                
        # Plot training performance
        if plot_save_path:
            self.plot_scores(scores=score_history, avg_scores=avg_score_history, plot_save_path=plot_save_path)

        return score_history, avg_score_history
    
            
    def her_augmentation(self, observations, actions, next_observations):
        """
        Augment the agent's replay buffer using Hindsight Experience Replay (HER).
        """
        # hyperparameter for future goals sampling
        k = 4

        # augment the replay buffer
        num_samples = len(actions)
        for index in range(num_samples):
            for _ in range(k):
                future_index = np.random.randint(index, num_samples)
                future_observation, future_achieved_goal, _ = next_observations[future_index].values()
                # print(future_achieved_goal)

                observation, _, _ = observations[future_index].values()
                
                state = torch.tensor(np.concatenate((observation, future_achieved_goal, future_achieved_goal)), 
                                     dtype=torch.float32).to(self.device)

                next_observation, _, _ = next_observations[future_index].values()
                
                next_state = torch.tensor(np.concatenate((next_observation, future_achieved_goal, 
                                                          future_achieved_goal)), dtype=torch.float32).to(self.device)

                action = torch.tensor(actions[future_index], dtype=torch.float32).to(self.device)
                reward = self.env.unwrapped.compute_reward(future_achieved_goal, future_achieved_goal, 1.0)

                # Store augmented experience in buffer
                state = state.cpu().numpy()
                action = action.cpu().numpy()
                next_state = next_state.cpu().numpy()

                # Store augmented experience in buffer
                self.memory.push(state, action, reward, next_state, True)
                
                
    def test_model(self, env=None, render_save_path=None, fps=30):
        """
        Run the trained agent in the environment.
        """
        if env is None:
            env = self.env
        episode_score = 0
        observation, info = env.reset()
        images = [env.render()]
        done = False
        truncated = False
        
        with torch.inference_mode():
            while not done:
                if render_save_path:
                    images.append(env.render())
                curr_obs, curr_achgoal, curr_desgoal = observation.values()
                state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

                action = self.select_action(state)

                next_observation, reward, done, truncated, _ = env.step(np.array(action))
                observation = next_observation
                episode_score += reward

                if done or truncated:
                    images.append(env.render())

        if render_save_path:
            # env.close()
            imageio.mimsave(f'{render_save_path}.gif', images, fps=fps, loop=0)
            with open(f'{render_save_path}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))
                
        return episode_score
    
    
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
        torch.save(self.critic_1.state_dict(), self.critic_1.checkpoints_file)
        torch.save(self.critic_2.state_dict(), self.critic_2.checkpoints_file)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoints_file)
        torch.save(self.target_critic_1.state_dict(), self.target_critic_1.checkpoints_file)
        torch.save(self.target_critic_2.state_dict(), self.target_critic_2.checkpoints_file)

    def load_model(self):
        """
        Load trained models.
        """
        self.actor.load_state_dict(torch.load(self.actor.checkpoints_file))
        self.critic_1.load_state_dict(torch.load(self.critic_1.checkpoints_file))
        self.critic_2.load_state_dict(torch.load(self.critic_2.checkpoints_file))
        self.target_actor.load_state_dict(torch.load(self.target_actor.checkpoints_file))
        self.target_critic_1.load_state_dict(torch.load(self.target_critic_1.checkpoints_file))
        self.target_critic_2.load_state_dict(torch.load(self.target_critic_2.checkpoints_file))