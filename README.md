# Learning Continuous Control using Inverse Reinforcement Learning

This project focuses on training a 7 DOF robotic arm agent from the PandaReach-v3 environment available in the panda-gym toolkit. The PandaReach-v3 task involves controlling a robotic arm to reach target objects in a simulated environment.We explore advanced algorithms for continuous control: Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3) to train the agent for this task. Further, we use the projection-based inverse reinforcement learning algorithm based on the paper “Apprenticeship Learning via Inverse Reinforcement Learning" by P. Abbeel and A. Y. Ng to train apprentice agents for the same task by using the trained agents as the expert agents. The apprentice agents are trained succesfully in the continouous domain and attain performances close to the expert with one apprentice agent (trained using TD3 in the IRL step) surpassing even the expert's performance. This demonstrates the successful application of inverse reinforcement learning in continuous control tasks.

<p align="center">
  <img src="assets/Trained%20Agent.gif"/>
</p>

## Reinforcement Learning algorithms for Continuous Control

Continuous reinforcement learning algorithms are designed to handle environments where actions are continuous, like controlling robotic arm joints with precision. These algorithms aim to discover policies that effectively map observed states to continuous actions, optimizing the accumulation of expected rewards. 

### DDPG

DDPG is an actor-critic algorithm designed for continuous action spaces. It combines the strengths of policy gradients and Q-learning. In DDPG, an actor network learns the policy, while a critic network approximates the action-value (Q-function). The actor network directly outputs continuous actions, which are evaluated by the critic network to output optimal actions.

### TD3

TD3 builds upon DDPG, addressing issues such as overestimation bias. It introduces twin critics to estimate the Q-value, employing two critic networks instead of one as in DDPG. Additionally, it utilizes target networks with delayed updates to stabilize training. TD3 is recognized for its robustness and enhanced performance compared to DDPG.

## Hindsight Experience Replay (HER)
Hindsight Experience Replay (HER) is a technique developed to tackle the challenge of sparse and binary rewards in reinforcement learning (RL) environments. In many robotic tasks, achieving the desired goal is rare, leading traditional RL algorithms to struggle with learning from such feedback. HER addresses this by repurposing past experiences for learning, regardless of whether they resulted in the desired goal. By relabeling failed attempts as succesful ones and storing both experiences in a replay buffer, the agent can learn from both successful and failed attempts, significantly improving the learning process.

## Inverse Reinforcement Learning

Apprenticeship Learning via Inverse Reinforcement Learning combines principles of reinforcement learning and inverse reinforcement learning to enable agents to learn from expert demonstrations. The agent learns to perform a task by observing demonstrations provided by an expert, without explicit guidance or reward signals. Instead of learning directly from rewards, the algorithm seeks to infer the underlying reward function from the expert demonstrations and then optimize the agent's behavior based on this inferred reward function.

One approach to implementing this is the Projection Method Algorithm, which iteratively refines the agent's policy based on the difference between the expert's behavior and the agent's behavior. At each iteration, the algorithm computes a weight vector that maximally separates the expert's feature expectations from the agent's feature expectations, subject to a constraint on the norm of the weight vector. This weight vector is then used to derive rewards and train the agent's policy using the above stated algorithms, and the process repeats until convergence. At least one of the trained apprentices performs at least as well as the expert within ϵ.

## Results:

### DDPG

- The expert is trained for 500 episodes
- Average reward of the expert over 1000 episodes = -1.768

<p align="center">
  <img src="Results/DDPG/Expert%20Performance.png" width="300" />
  <img src="Results/DDPG/Expert%20Policy.gif" width="350"/>
  <p align="center">CartPole expert trained using Q learning</p>
</p>

#### Apprentice agents

- Ten apprentices were trained using the IRL algorithm.
- The best performing apprentice agent has an average reward of -1.852 over 500 episodes.

<p align="center">
  <img src="Results/DDPG/Apprentice_1%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_2%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_3%20Performance.png" width="250"/>

  <img src="Results/DDPG/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/DDPG/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/DDPG/Apprentice_7%20Performance.png" width="250" />
  <img src="Results/DDPG/Apprentice_9%20Performance.png" width="250"/>
  <img src="Results/DDPG/Apprentice_10%20Performance.png" width="250"/>

  <img src="Results/DDPG/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%209%20Policy.gif" width="250"/>
  <img src="Results/DDPG/Apprentice%2010%20Policy.gif" width="250"/>
</p>

### TD3

- The expert is trained for 500 episodes
- Average reward of the expert over 1000 episodes = -1.932

<p align="center">
  <img src="Results/TD3/Expert%20Performance.png" width="300" />
  <img src="Results/TD3/Expert%20Policy.gif" width="350"/>
  <p align="center">CartPole expert trained using Q learning</p>
</p>

#### Apprentice agents

- Ten apprentices were trained using the IRL algorithm.
- The best performing apprentice agent surpasses the expert and has an average reward of -1.852 over 500 episodes.

<p align="center">
  <img src="Results/TD3/Apprentice_1%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_2%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_3%20Performance.png" width="250"/>

  <img src="Results/TD3/Apprentice%201%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%202%20Policy.gif" width="250" />
  <img src="Results/TD3/Apprentice%203%20Policy.gif" width="250"/>
</p>

<p align="center">
  <img src="Results/TD3/Apprentice_7%20Performance.png" width="250" />
  <img src="Results/TD3/Apprentice_9%20Performance.png" width="250"/>
  <img src="Results/TD3/Apprentice_10%20Performance.png" width="250"/>

  <img src="Results/TD3/Apprentice%207%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%209%20Policy.gif" width="250"/>
  <img src="Results/TD3/Apprentice%2010%20Policy.gif" width="250"/>
</p>

## Documentation

For an overview of the project and its implementation, refer to the [presentation](docs/Learning%20Continuous%20Control%20using%20IRL.pdf) file.

## References:
- Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, & Daan Wierstra. (2015). Continuous control with deep reinforcement learning.
- Scott Fujimoto, Herke van Hoof, & David Meger (2018). Addressing Function Approximation Error in Actor-Critic Methods. CoRR, abs/1802.09477.
- Quentin Gallouédec, Nicolas Cazin, Emmanuel Dellandréa, & Liming Chen. (2021). panda-gym: Open-source goal-conditioned environments for robotic learning.
- Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, & Wojciech Zaremba. (2017). Hindsight Experience Replay.
- Abbeel, P. & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.
- Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. In International Conference on Machine Learning (pp. 1582–1591).
- Omkar Chittar. (n.d.). Omkarchittar/manipulator_control_DDPG - GitHub.