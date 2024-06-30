import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from gymnasium.envs.registration import register

import pass_process

from gymnasium.experimental.vector import VectorEnv
class Skip(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    STOP = 1
    SKIP= 0

    def __init__(self, num_trains=6, num_stations=6, num_time =60,  pass_od = None, render_mode="console"):
        super(Skip, self).__init__()
        self.render_mode = render_mode
        #passengers flow, dict
        self.PassOD = pass_od

        # numbers of trains
        self.train = num_trains
        # number of station
        self.station = num_stations
        # peak hour time (1 time inverval = 1 min)
        self.num_time = num_time
        
        
        # when the train decides stop at the station, the stop time is 1 min
        self.stop_time = 1
        # when train running in the section, the running time is 10 -1 = 9 min
        self.running_time = 9
        # the train capacity 
        self.train_cap = 20
        
        
        
        

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
        low=np.array([0, 0, 0, 0]), high=np.array([self.num_time+180, 1000, 2000, 2000]), shape=(4,), dtype=np.int64
    )

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # count the train, station, time unit
        self.train_index = 1
        self.station_index = 1
        self.time_index = 1
        
        #the initial state s_0
        #departue time of the current train, its the first item of state
        self.DepartTime = 1
        #remining passengers who cannot boarding the train
        self.RemainPass = 0
        #passengers on the train
        self.OntrainPass = 0
        #passenger waiting on the station when train arrive at the station
        self.WaitPass = 0
        

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.DepartTime,self.RemainPass, self.OntrainPass, self.WaitPass ]).astype(np.float32), {}  # empty info dict

    def step(self, action):
        if self.station_index == 0:
            self.action = 1
        else:
            if action == self.LEFT:
                self.agent_pos -= 1
            elif action == self.RIGHT:
                self.agent_pos += 1
            else:
                raise ValueError(
                    f"Received invalid action={action} which is not part of the action space"
                )

        current_station_pass = self.PassOD[self.time_index]
        
        current_train = pass_process.boarding()
        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        terminated = bool(self.agent_pos == 0)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.int32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
    
    
    
      

      
