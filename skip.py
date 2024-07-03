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
        self.train_num = num_trains
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
        #in the toy instance, the departure time is fixed
        self.depart_interval = 10
        
        # dict for onboard passenger
        self.onboard_dict = {key: [] for key in range(0, self.train_num)}
        
        
        
        

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
        # count the train, station, time unit, the count start from 0
        self.train_index = 0
        self.time_index = 0
        # count the station, the count start from 1.
        #the od also strat from 1 to 6
        self.station_index = 0
        #the passeners waiting at the in every station
        self.PassengerEvery = {key: [] for key in range(0, self.train_num )}
        
        #the initial state s_0
        #departue time of the current train, its the first item of state
        self.DepartTime = 0
        #passengers on the train 
        self.PassengerRemain = {key: [] for key in range(0, self.train_num )}
        #remining passengers who cannot boarding the train
        self.PassengerWaiting = {key: [] for key in range(0, self.station )}
        #the total passenger waiting on the station when train arrive at the station
        self.PassengerTotal = {key: [] for key in range(0, self.station )}
        
        
        self.FirstBoardProcess()

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        #return np.array([self.DepartTime,self.PassengerWaiting, self.PassengerRemain, self.PassengerTotal]).astype(np.float32), {}  # empty info dict
        return np.array([self.DepartTime,len(self.PassengerWaiting[self.station_index]), len(self.PassengerRemain[self.station_index]), len(self.PassengerTotal[self.station_index])]).astype(np.float32), {} 

    def step(self, action):
        
        self.station_index += 1
        #train running process
        self.time_index = self.time_index +self.running_time
        #the passenger waiting as the current station p^n_ks

            
        time_interval = [self.DepartTime, self.time_index+1]
        
        
        for key in self.PassOD:
            if time_interval[0] <= key <= time_interval[1]:
                values = self.PassOD[key]
                for value in values:
                    new_tuple = value[:0] + (key,) + value[1:]   #transfer the first item 'start station' to arrive time 'time', the information 'start staion' is showed in the Passenger dic key
                    self.PassengerTotal[value[0]].append(new_tuple)
            
            #boarding process, alighting process, calculate the passenger who fail to board on the train
           
            self.PassengerBoardProcess()
                
        
        
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
    
    
    def FirstBoardProcess(self):
        
        #The earliest passenger arrival time, to calculate number of the passenger 
        #who arrive at the first station when the train departure this station
        
        #when the first train start from the first train, there no passenger
        
        if self.time_index in self.PassOD.keys():
            values = self.PassOD[self.time_index]           
            for value in values:
                if value[0] == self.station_index:
                    self.onboard_dict[self.train_index].append(value)
                    values.remove(value)
    
    def PassengerBoardProcess(self):
        
        #Alighting Process
        onboard_passengers = self.onboard_dict[self.train_index]
        
        #先下车， 下车时间不限
        #下完车，用while 判断列车是否坐满了，再上车
        #再更新没有上车的字典
        
        
        if self.time_index in self.PassOD.keys():
            
            for value in values:
                if value[0] == self.station_index:
                    self.onboard_dict[self.train_index].append(value)
                    values.remove(value)
                    break
            if len(values) == 0:
                del self.PassOD[self.time_index]

            
        
        else:
            pass
            
    

    
    
      

      
