from ray.rllib.env.multi_agent_env import MultiAgentEnv
from FOSSEnvBase import FOSSEnvBase
import numpy as np
from copy import deepcopy
from config import Config
config = Config()
class FOSSEnvTest(MultiAgentEnv):
    def __init__(self,env_config):
        super().__init__()
        self.numagents = config.num_agents
        self.planhelper = env_config['planhelper']
        tableNum = self.planhelper.gettablenum()
        self.unwrapped_env = FOSSEnvBase({'tableNum':tableNum})
        self._agent_ids = set(range(self.numagents))
        self.observation_space = self.unwrapped_env.observation_space
        self.action_space = self.unwrapped_env.action_space
        self.resetted = False
        
    def reset(self, *, seed=None, options=None):
        # if not self.resetted or self.isloop:
        #     self.resetted = True
        #     self.isloop = False
        #     return {},{'over':True}
        
        super().reset(seed=seed)
        self.stepnum = 0
        self.sql = options['sql']
        self.query_id = options['query_id']

        feature_dict,self.hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',self.sql,True,query_id = self.query_id)
        self.plantime = cost_plan_json['Planning Time']

        self.count_table = len(self.hintdict['join order'])
        
        if left_deep and self.count_table > 2:
            assert self.count_table == len(self.hintdict['join operator']) + 1
            self.use_FOSS = True
        else:
            self.use_FOSS = False 
            print('BAN FOSS')
        
        # init action
        self.action_mask = np.zeros(self.unwrapped_env.action_space_size)
        for i in range(self.count_table - 1):
            self.action_mask[ self.unwrapped_env.action_inteval[i]: self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1
        # self.unwrapped.action_mask[0] = 0  # 交换前两个元素没有意义
        self.action_mask[-3 * (self.count_table - 1):] = 1
        for i,jo in enumerate(self.hintdict['join operator']):   
            self.action_mask[-3 * i - config.OperatorDict[jo]] = 0
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
        state = feature_dict.copy()
        state['action_mask'] = self.action_mask
        state_all = {}
        self.action_mask_tatal = []
        self.hintdict_total = []
        self.es_hint = []
        
        for i in range(self.numagents):
            self.hintdict_total.append(deepcopy(self.hintdict))
            self.action_mask_tatal.append(deepcopy(self.action_mask))
            state_all.update({i:deepcopy(state)})
            self.es_hint.append('')
        info = {'hint':'','useFOSS':self.use_FOSS}
        return (state_all,info)
    def step(self, action_dict):
        self.stepnum += 1
        state = {}
        rew, terminated, truncated, info =  {}, {'__all__':False}, {'__all__':False},{}
        for agent_id, action in action_dict.items():
            if action >= (self.unwrapped_env.tablenum * (self.unwrapped_env.tablenum - 1)) / 2:
                idx = abs(action - self.unwrapped_env.action_space_size + 1)
                self.hintdict_total[agent_id]['join operator'][int(idx/3)] = config.Operatortype[idx%3]
                for i in range(self.count_table - 1):
                    self.action_mask_tatal[agent_id][self.unwrapped_env.action_inteval[i]:self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1
                self.action_mask_tatal[agent_id][-3 * (self.count_table - 1):] = 1
                for i,jo in enumerate(self.hintdict_total[agent_id]['join operator']):   
                    self.action_mask_tatal[agent_id][-3 * i - config.OperatorDict[jo]] = 0
            # 如果是交换两个表的顺序
            else:
                tag = -1
                for i in range(len(self.unwrapped_env.action_inteval)):
                    if action < self.unwrapped_env.action_inteval[i]:
                        tag = i
                        break
                if tag != -1:
                    t1 = tag - 1
                    t2 = action - self.unwrapped_env.action_inteval[t1] + tag
                    temp = self.hintdict_total[agent_id]['join order'][t1]
                    self.hintdict_total[agent_id]['join order'][t1] = self.hintdict_total[agent_id]['join order'][t2]
                    self.hintdict_total[agent_id]['join order'][t2] = temp 
                    # 第tag个与第action - self.action_inteval[tag - 1] + 1 + tag交换
                    self.action_mask_tatal[agent_id].fill(0)
                    if t1 == 0 or t1 == 1 or t2 == 1:
                        self.action_mask_tatal[agent_id][-3:] = 1
                    else:
                        self.action_mask_tatal[agent_id][-3 * (t1):-3 * (t1 - 1)] = 1
                        self.action_mask_tatal[agent_id][-3 * (t2):-3 * (t2 - 1)] = 1
                    
            exechint = self.planhelper.to_exechint(self.hintdict_total[agent_id])
            # latency_timeout,_ = self.planhelper.getLatency(exechint,self.sql,self.query_id)
            # print('test_{}_agent_{}_step_{},Latency:{:.4f}'.format(self.query_id,agent_id,self.stepnum,latency_timeout[0]))
            feature_dict,_,_,cost_plan_json = self.planhelper.get_feature(exechint,self.sql,False,query_id = self.query_id)
            self.plantime += cost_plan_json['Planning Time']
            feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
            state[agent_id] = feature_dict.copy()
            state[agent_id]['action_mask'] = self.action_mask_tatal[agent_id]
            info[agent_id] = {'hint':exechint}
            rew[agent_id] = 0
        if self.stepnum >= config.maxsteps:
            terminated['__all__'] = True
            truncated['__all__'] = True

        return state, rew, terminated, truncated, info