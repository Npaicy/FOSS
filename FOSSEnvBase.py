import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config

config = Config()

class FOSSEnvBase(gym.Env):
    def __init__(self,env_config):
        
        # self.planhelper = PlanHelper()
        self.tablenum = env_config['tableNum']#self.planhelper.gettablenum()
        
        self.action_space_size = int(((self.tablenum + 5) * (self.tablenum)) / 2) # n(n-1)/2 + 3n
        # self.observation_space = spaces.Box(-np.inf,np.inf,dtype = np.float32,shape = (config.hidden_dim + 1,))
        self.observation_space = spaces.Dict({
            'x':spaces.Box(-np.inf,np.inf,dtype = np.float32,shape = (config.maxnode, 10 + 5 + config.maxjoins)),
            'attn_bias':spaces.Box(0,1,dtype = np.float32,shape = (config.maxnode + 1,config.maxnode + 1)),
            'heights':spaces.Box(0,config.maxnode,dtype = np.int64,shape = (config.maxnode,)),
            'action_mask':spaces.Box(0,1,dtype = np.int32,shape = (self.action_space_size,)),
            'steps':spaces.Box(0,1,dtype = np.float32,shape = (1,))
            })
        self.action_space = spaces.Discrete(self.action_space_size)
        # 设置动作分割区间
        self.action_inteval = [0]
        for i in range(1,self.tablenum):
            self.action_inteval.append(self.action_inteval[-1] + self.tablenum - i)

    def reset(self, seed=None, options=None):
    
        return None,None
    def step(self,action):
       
        return None,None,None,None,None