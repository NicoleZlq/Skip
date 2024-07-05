import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import data_generator
import argparse
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

import warnings
warnings.filterwarnings("ignore")


import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


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

    read_OD = data_generator.process_csv()
    
    print(read_OD)
    
    env = gym.make("Skip-v0", num_trains=args.train, num_stations=args.station, num_time = args.time, pass_od = read_OD)
    

    model = PPO("MlpPolicy", env, verbose=1, seed=1, tensorboard_log="./skip_small_01_tensorboard/")
    
    model.learn(50000, callback=HParamCallback(BaseCallback))
    
    model.save("a2c_skip_small")
    
    del model
    
    n_steps =500

    # using the vecenv
    obs,_ = env.reset()
    
    model = PPO.load('a2c_skip_small')
    
    
    for step in range(n_steps):

        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done,_, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
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

# 解析参数:
args = parser.parse_args()

main(args)