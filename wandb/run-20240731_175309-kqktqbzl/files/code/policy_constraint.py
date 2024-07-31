import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C,PPO,DQN
from ca2c import CA2C
from stable_baselines3.common.vec_env import SubprocVecEnv
import data_generator
import argparse
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import random
import wandb

import warnings
warnings.filterwarnings("ignore")


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
            "eval/mean_reward": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True



def main(args):

# Instantiate the env

    read_OD = data_generator.process_csv(args.instance)

    env = gym.make("Skip-v0", num_trains=args.train,   num_stations=args.station, num_time = args.time, pass_od = read_OD)
    
    model = CA2C(env, 
                 learning_rate=1e-4, 
                 gamma=0.99,
                 initial_epsilon=1.0,
                 final_epsilon=0.01,
                 seed=984,
                 instance = "3",
                epsilon_decay_steps=20000,
                learning_starts=100,
                 log=True,
                 project_name="skip",
                experiment_name="RCPO",)

   # model = A2C("MlpPolicy", env, learning_rate=0.0001, use_rms_prop=False, gamma=1.0,gae_lambda=0.5,max_grad_norm=0.6)
    model.learn(int(1e3), env, train=True, critic_path=None, actor_path=None)
    
    model.learn(int(1e3), env, train=False, critic_path=args.critic_path, actor_path=args.actor_path)
    
    
    n_steps =5

    # using the vecenv
    obs,_ = env.reset()
    
    model = A2C.load("skip_small_{}".format(args.instance))
    
    
    for step in range(n_steps):

        action, _ = model.predict(obs, deterministic=False)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, missing,  travel, done,_, info = env.step(action)
        #print("obs=", obs, "reward=", reward, "done=", done)
        env.render()
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break

# Instantiate the env
parser = argparse.ArgumentParser(
        prog='skip_env', # 程序名
        description='Skip'
    )
# Train the agent
    # 此参数必须为int类型:
parser.add_argument('--port', default='3306', type=int)
# 允许用户输入简写的-u:
parser.add_argument('-t',  '--train', default=6, type=int )
parser.add_argument('-s','--station', default=6, type=int )
parser.add_argument('--time',  default=60, type=int)
parser.add_argument('--instance',  default=3, type=int)
parser.add_argument('--critic_path',  default='save_model/actor_3.path', type=str)
parser.add_argument('--actor_path',  default='save_model/actor_3.path', type=str)

# 解析参数:
args = parser.parse_args()

main(args)


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# The network of the actor

if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 0
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 9
    # Set random seed
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode

    agent = A2C(state_dim, action_dim)
    
    agent.train(
        total_timesteps= int(1000 * 50),
        eval_env=env,
        timesteps_per_iter=int(2000),
    )


    max_train_steps = 3e5  # Maximum number of training steps
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        s = env.reset()
        done = False
        agent.I = 1
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)

            # When dead or win or reaching the max_epsiode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            agent.learn(s, a, r, s_, dw)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/A2C_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1