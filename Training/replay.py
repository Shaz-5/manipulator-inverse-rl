import numpy as np

# Replay Memory
class ExperienceReplayMemory:
    def __init__(self, capacity, state_shape, num_actions):
        self.capacity = capacity
        self.size = 0
        self.state_memory = np.zeros((capacity, state_shape))
        self.action_memory = np.zeros((capacity, num_actions))
        self.reward_memory = np.zeros(capacity)
        self.next_state_memory = np.zeros((capacity, state_shape))
        self.dones_memory = np.zeros(capacity)
    
    def push(self, state, action, reward, next_state, done):
        index = self.size % self.capacity
        # print(state)
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.dones_memory[index] = done

        self.size += 1

    def sample(self, batch_size):
        max_capacity = min(self.capacity, self.size)
        batch = np.random.choice(max_capacity, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.dones_memory[batch]

        return states, actions, rewards, next_states, dones