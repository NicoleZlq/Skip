import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C,PPO,DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import data_generator
import argparse
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import random

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
    
    
    np.random.seed(20240101)

    model = A2C("MlpPolicy", env, learning_rate = 0.0004, use_rms_prop=False, verbose=1,tensorboard_log="./skip_small_01_tensorboard/")

    #model = A2C("MlpPolicy", env, verbose=1, seed=seed1)
    model.learn(1e5, log_interval=100,  callback=HParamCallback(BaseCallback), tb_log_name="A2C_instance{}".format(args.instance),reset_num_timesteps=True)

    
    model.save("skip_small_{}".format(args.instance))
    
    del model # remove to demonstrate saving and loading
    
    n_steps =5

    # using the vecenv
    obs,_ = env.reset()
    
    model = A2C.load("skip_small_{}".format(args.instance))
    
    
    for step in range(n_steps):

        action, _ = model.predict(obs, deterministic=True)
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
parser.add_argument('--instance',  default=1, type=int)

# 解析参数:
args = parser.parse_args()

main(args)