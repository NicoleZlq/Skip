"""Envelope Q-Learning implementation."""
import os
from typing import List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import time
from typing import Dict, Optional, Union



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


# The network of the critic
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        v_s = self.l2(s)
        return v_s


class CA2C(object):
    def __init__(self,
                 env, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 initial_epsilon: float = 0.01,
                final_epsilon: float = 0.01,
                batch_size: int = 256,
                learning_starts: int = 100,
                buffer_size: int = int(1e6),
                epsilon_decay_steps: int = None,  # None == fixed epsilon
                 log: bool = True,
                 seed: Optional[int] = None,
                 project_name: str = "CRCPO",
                experiment_name: str = "td",
                wandb_entity: Optional[str] = None,
                 ):
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.hidden_width = 64  # The number of neurons in hidden layers of the neural network
        self.lr = learning_rate  # learning rate
        self.GAMMA = gamma  # discount factor
        self.I = 1
        self.env = env
        self.seed = seed
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        
        self.np_random = np.random.default_rng(self.seed)

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.info ={}
        
        
        self.replay_buffer = self.ReplayBuffer(
                self.state_dim,
                1,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )
        
        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.actor(s).detach().numpy().flatten()  # probability distribution(numpy)
        
        if self.np_random.random() < self.epsilon:
            a = np.random.choice(range(self.action_dim), p=prob_weights)
        else:
            a = np.argmax(prob_weights)
        return a

        
    def learn(self,
        total_timesteps: int,
        env: Optional[gym.Env] = None,):
        
        
        
        
        """Train the agent.
            Args:
            total_timesteps: total number of timesteps to train for.
            env: environment to use for evaluation. If None, it is ignored.
            eval_freq: policy evaluation frequency (in number of steps). 
            """
        self.global_step = 0
        self.num_episodes = 0
        num_episodes = 0
        s, z = env.reset()
        
        for _ in range(1, total_timesteps+1):
            a = self.choose_action(s, deterministic=False)
            s_, r, missing, traveling, done, trcthu, info = env.step(a)  #return observation, reward, missing, traveling, terminated, truncated, info
            self.global_step += 1
            
            self.add(s, a, r, s_, done)
            
            if self.global_step >= self.learning_starts:
                self.linearly_decaying_value(self.initial_epsilon,
                                                self.epsilon_decay_steps,
                                                self.global_step,
                                                self.learning_starts,
                                                self.final_epsilon,)
            
            
            
            if done:
                print(self.global_step)
                s, z = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1
                
                self.info={
                    "n": sum(missing)/len(missing),
                     "tr": sum(traveling)/len(traveling),
                    "m": max(missing),
                    "o": sum([x**2 for x in missing])/len(missing), 
                    "r": r, 
                    "p": len(traveling)}


                if self.log :
                    print(self.global_step)
                    self.log_episode_info(self.info, self.global_step)

            else:
                s = s_
                
        print("Done training!")
        self.env.close()
        if self.log:
            self.close_wandb()
                
                
    def log_episode_info(
        self,
        info: dict,
        global_timestep: int,
        id: Optional[int] = None,
        verbose: bool = True,
    ):
        """Logs information of the last episode from the info dict (automatically filled by the RecordStatisticsWrapper).

        Args:
            info: info dictionary containing the episode statistics
            scalarization: scalarization function
            weights: weights to be used in the scalarization
            : global timestep
            id: agent's id
            verbose: whether to print the episode info
        """
        missing_num = info["n"]
        max_num = info["m"]
        objective = info["o"]
        travel = info["tr"]
        passenger = info["p"]
        reward = info["r"]

        if verbose:
            print("Episode infos:")
            print(f"Total Reward: {reward}")

        if id is not None:
            idstr = "_" + str(id)
        else:
            idstr = ""
        wandb.log(
            {
               "eval/missing train": missing_num,
                "eval/maximum": max_num,
               "eval/objective": objective,
               "eval/travel time": travel,
               "eval/passenger_num":passenger,
                "eval//reward": reward,
                "global_step": global_timestep,
            },
            commit=None,
            step = global_timestep
        )


                
    def ReplayBuffer(self, 
        obs_shape,
        action_dim,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):

        """Initialize the replay buffer.

        Args:
            obs_shape: Shape of the observations
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        print(max_size,obs_shape)
        print(type(max_size), type(obs_shape))
        self.obs = np.zeros((max_size, obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,  obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""

        wandb.finish()

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Whether to use CER
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)

        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        """Sample a batch of observations from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A batch of observations
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return th.tensor(self.obs[inds]).to(device)
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None):
        """Get all the data in the buffer (with a maximum specified).

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        return (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )

    def __len__(self):
        """Get the size of the buffer."""
        return self.size

            



    def abc(self, s, a, r, s_, dw):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float), 0)
        v_s = self.critic(s).flatten()  # v(s)
        v_s_ = self.critic(s_).flatten()  # v(s')

        with torch.no_grad():  # td_target has no gradient
            td_target = r + self.GAMMA * (1 - dw) * v_s_

        # Update actor
        log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
        actor_loss = -self.I * ((td_target - v_s).detach()) * log_pi  # Only calculate the derivative of log_pi
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        critic_loss = (td_target - v_s) ** 2  # Only calculate the derivative of v(s)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.I *= self.GAMMA  # Represent the gamma^t in th policy gradient theorem
        
        
    def register_additional_config(self, conf: Dict = {}) -> None:
        """Registers additional config parameters to wandb. For example when calling train().

        Args:
            conf: dictionary of additional config parameters
        """
        for key, value in conf.items():
            wandb.config[key] = value
            
    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.lr,
            "initial_epsilon": self.initial_epsilon,
             "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "gamma": self.GAMMA,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def setup_wandb(self, project_name: str, experiment_name: str, entity: Optional[str] = None, group: Optional[str] = None) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.
            entity: wandb entity. Usually your username but useful for reporting other places such as openrlbenmark.

        Returns:
            None
        """
        self.experiment_name = experiment_name
        env_id =  self.env.spec.id
        self.full_experiment_name = f"{env_id}__{experiment_name}__{self.seed}__{int(time.time())}"


        config = self.get_config()

        config["algo"] = self.experiment_name

        for i in range(20):
            try:
                wandb.init(
                    project=project_name,
                    entity=entity,
                    sync_tensorboard=True,
                    config=config,
                    name=self.full_experiment_name,
                    monitor_gym=True,
                    save_code=True,   )
                break
            except:
                print('wandb init failed')
                continue

        
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""
        import wandb

        wandb.finish()
        
        
    def linearly_decaying_value(self, initial_value, decay_period, step, warmup_steps, final_value):
        """Returns the current value for a linearly decaying parameter.

        This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
        al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.

        Args:
            decay_period: float, the period over which the value is decayed.
            step: int, the number of training steps completed so far.
            warmup_steps: int, the number of steps taken before the value is decayed.
            final value: float, the final value to which to decay the value parameter.

        Returns:
            A float, the current value computed according to the schedule.
        """
        steps_left = decay_period + warmup_steps - step
        bonus = (initial_value - final_value) * steps_left / decay_period
        value = final_value + bonus
        value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
        return value


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)



class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_shape[0] + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions

        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs / 255.0)
            input = torch.cat((features, w), dim=w.dim() - 1)
        else:
            input = torch.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards


