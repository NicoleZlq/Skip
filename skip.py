import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import copy

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
        self.OriginalOD = pass_od.copy()

        # numbers of trains
        self.train_num = num_trains
        # number of station
        self.station_num = num_stations
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
        # initial state
        state_length = 1+1+self.station_num+self.train_cap+self.train_cap+1
        
        self.ini_state = [0] * state_length
        
        max_value = [2000] * state_length
        
        self.observation_space = spaces.Box(
        low=np.array(self.ini_state), high=np.array(max_value), shape=(state_length,), dtype=np.int64
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
        #departure time of the current train, its the first item of state
        self.DepartTime = 0
        #the initial departure time from the first station
        self.StartDepartTime = 0
        #predifined departure interval
        self.PreDepartInterval =10
        #the total passenger waiting on the station when train arrive at the station
        self.PassengerTotal = {key: [] for key in range(0, self.station_num )}
        
        #waiting time, use to calculate the missing train
        self.TotalWaitingTime = []
        
        
        self.PassOD = self.OriginalOD.copy()
        
        
        self.FirstBoardProcess()

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        #return np.array([self.DepartTime,self.PassengerWaiting, self.PassengerRemain, self.PassengerTotal]).astype(np.float32), {}  # empty info dict
        return np.array(self.ini_state).astype(np.float32), {} 

    def step(self, action):
        
        self.station_index += 1
        
        print('station:', self.station_index)
        print('train:', self.train_index)
        
        #train running process
        self.time_index = self.DepartTime +self.running_time
        #the passenger waiting as the current station p^n_ks
        
        self.UpdateTrain()

        
        self.PassengerArriveProcess() #calculate the total passenger who waiting at the station

        if action == 1 and self.station_index !=0:
            self.time_index +=1
            self.PassengerBoardProcess() #alighting, boarding, updating the waiting passengers
            
            
        self.DepartTime = self.time_index
            

        # predefine the constrain condition
        terminated = bool(self.station_index == self.station_num-2 and self.train_index == self.train_num-1)
        
        reward = -sum([(num // 9) ** 2 for num in self.TotalWaitingTime])/1000
        
        
        
        print('*************', terminated)
        if terminated:
            added_reward = self.CalculateFinalReward()/1000  #these passenger connot boarding on the train
            reward -= added_reward
            
            for key in self.PassengerTotal:
                length = len(self.PassengerTotal[key])
                print(f"The length of key {key} is {length}.")
        truncated = False  # we do not limit the number of steps here
        

        print('reward:', reward)
        
        
        

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        state = self.UpdateState()
        
        

        return (
            np.array(state).astype(np.float32),
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
    
    def UpdateState(self):
        pass
    #station_index + train_index　＋　numbers of remaining passengers in each station 
    #+ onboard passengers arriving time +  onboard passengers end station
    #  1 + 1 + train_num + station_num + capacity + capacity
    
        info = [self.train_index, self.station_index]
        
        remaining_pass =  [len(value_list) for value_list in self.PassengerTotal.values()]
        
        arriving_time = [value_list[0] for value_list in self.onboard_dict[self.train_index]]

        arriving_time.extend([-1] * (self.train_cap - len(arriving_time)))
        
        end_station = [value_list[1] for value_list in self.onboard_dict[self.train_index]]

        end_station.extend([-1] * (self.train_cap - len(end_station)))
        
        time = [self.DepartTime]
        
        return  info + remaining_pass + arriving_time + end_station + time
    
    
    def FirstBoardProcess(self):
        
        #The earliest passenger arrival time, to calculate number of the passenger 
        #who arrive at the first station when the train departure this station
        
        #when the first train start from the first train, there no passenger
        
        if self.time_index in self.PassOD.keys():
            values = self.PassOD[self.time_index]  
            boarding = [value for value in values if value[0] == self.station_index]
            updated_values = [value for value in values if value[0] != self.station_index]
            self.onboard_dict[self.train_index] = boarding
            self.PassOD[self.time_index] =  updated_values
        
        print(self.PassOD[self.time_index])
        print(self.onboard_dict[self.train_index])
            
            
                  

                    
    def PassengerArriveProcess(self):
        
        time_interval = [self.DepartTime, self.time_index+1]
        
        
        for key in self.PassOD:
            if time_interval[0] <= key <= time_interval[1]:
                while len(self.PassOD[key])>0:
                    value = self.PassOD[key].pop(0)     #Avoid passengers being counted repeatedly   
                    new_tuple = value[:0] + (key,) + value[1:]   #transfer the first item 'start station' to arrive time 'time', the information 'start staion' is showed in the Passenger dic key
                    self.PassengerTotal[value[0]].append(new_tuple)
                    
                    
        for key in self.PassengerTotal:
                    
            self.PassengerTotal[key] = sorted(self.PassengerTotal[key], key=lambda x: x[0])
    
    
    def PassengerBoardProcess(self):
        
        #Alighting Process
        Passengers = self.onboard_dict[self.train_index]

        self.onboard_dict[self.train_index] = [value for value in Passengers if value[1] > self.station_index]
        
        print('before')
        print('waiging passengers:',len(self.PassengerTotal[self.station_index]))
        print('numbers of onboard passenger', len(self.onboard_dict[self.train_index])) 
               
       #Boarding Process
        while len(self.onboard_dict[self.train_index]) < self.train_cap and len(self.PassengerTotal[self.station_index]) >0:   #constrain: train capacity
            boarding_passenger = self.PassengerTotal[self.station_index].pop(0)  
            self.onboard_dict[self.train_index].append(boarding_passenger)
            
        #calcualte the waiting time until the passenger board on the train
        ArrivingTime = [value[0] for value in self.onboard_dict[self.train_index]]
        WaitingTime = [self.time_index - item for item in ArrivingTime]
        self.TotalWaitingTime.extend(WaitingTime)
        
        print('after')
        print('numbers of remaining passenger', len(self.PassengerTotal[self.station_index]))
        print('numbers of onboard passenger', len(self.onboard_dict[self.train_index]))
        

            
    def UpdateTrain(self):
        
        if self.station_index == self.station_num-1:
            
            self.onboard_dict[self.train_index] = []    #in the end station, all of passenger should be alighting
            self.train_index +=1
            self.station_index = 0
        
            self.DepartTime = self.time_index = self.StartDepartTime = self.StartDepartTime + self.PreDepartInterval
            
            #take the passenger in the first station
            
            print('waiging passengers:',len(self.PassengerTotal[self.station_index]))
            
            passengers = self.PassengerTotal[self.station_index]
            
            selected_passenger = [value for value in passengers if value[0] <= self.time_index]
            updated_passengers = [value for value in passengers if value[0] > self.time_index]
            
            
            num_to_passenger =  min(len(selected_passenger), self.train_cap)
            
            self.onboard_dict[self.train_index] =  selected_passenger[:num_to_passenger]
            
            updated_passengers += selected_passenger[num_to_passenger:]
            
            
            self.PassengerTotal[self.station_index] = updated_passengers

            
            print('in the first staition')
            print('waiging passengers:',len(self.PassengerTotal[self.station_index]))
            print('numbers of onboard passengers:',len(self.onboard_dict[self.train_index]))
            
            
            

    def CalculateFinalReward(self):
        sum_of_squares = 0

        # Iterate through the dictionary values
        for key, value in self.PassengerTotal.items():
    # Check if the list is not empty
            if value:  # This ensures there is at least one element in the list
                # Take the first item, which could be a number or a tuple
                print(value)
                for i in value:
                    first_item = i[0]
                    # Divide the first item by 9, square it, and add to the sum
                    sum_of_squares += (first_item // 9) ** 2
        
        return sum_of_squares
         

            

            

            
           
            

    

    
    
      

      
