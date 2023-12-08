from ray.rllib.env.multi_agent_env import MultiAgentEnv
from util import min_steps
from FOSSEnvBase import FOSSEnvBase
import numpy as np
from copy import deepcopy
from config import Config
config = Config()
class FOSSEnvExplore(MultiAgentEnv):

    def __init__(self,env_config):
        super().__init__()
        self.planhelper = env_config['planhelper']
        tableNum = self.planhelper.gettablenum()
        self.unwrapped_env = FOSSEnvBase({'tableNum':tableNum})
        self.numagents = config.num_agents
        self.observation_space = self.unwrapped_env.observation_space
        self.action_space = self.unwrapped_env.action_space
        self._agent_ids = set(range(self.numagents))
        self.resetted = False
        
    def reset(self, *, seed=None, options=None):
        # if not self.resetted:
        #     self.resetted = True
        #     return {},{}
        super().reset(seed=seed)
        self.stepnum = 0
        self.query_id = options['query_id']
        self.sql = options['sql']
        feature_dict,self.basehintdict,_ = deepcopy(options['feature'])

        self.count_table = len(self.basehintdict['join order'])
        assert self.count_table == len(self.basehintdict['join operator']) + 1
        # init action
        self.action_mask = np.zeros( self.unwrapped_env.action_space_size)
        for i in range(self.count_table - 1):
            self.action_mask[ self.unwrapped_env.action_inteval[i]: self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1
        self.action_mask[-3 * (self.count_table - 1):] = 1
        for i,jo in enumerate(self.basehintdict['join operator']):   
            self.action_mask[-3 * i - config.OperatorDict[jo]] = 0
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
        feature_dict['action_mask'] = self.action_mask        

        state_all = {}
        self.action_mask_tatal = []
        self.hintdict_total = []

        for i in range(self.numagents):
            self.hintdict_total.append(deepcopy(self.basehintdict))
            self.action_mask_tatal.append(self.action_mask.copy())
            state_all.update({i:deepcopy(feature_dict)})
        return (state_all,{})
    def step(self, action_dict):
        self.stepnum += 1
        ExecHint = []
        CostPlanJson = []
        min_step = []
        state, rew, terminated, truncated = {}, {}, {'__all__':False}, {'__all__':False}
        for agent_id, action in action_dict.items():
            # =============action===================
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
            # ========================================
            min_step.append(min_steps(self.basehintdict, self.hintdict_total[agent_id]))
            exechint = self.planhelper.to_exechint(self.hintdict_total[agent_id])
            feature_dict,_,_,cost_plan_json = self.planhelper.get_feature(exechint,self.sql,False,query_id = self.query_id)
            ExecHint.append(exechint)
            feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
            feature_dict['action_mask'] = self.action_mask_tatal[agent_id]
            cost_plan_json['steps'] = self.stepnum * 1.0 / config.maxsteps
            CostPlanJson.append(deepcopy(cost_plan_json))
            state[agent_id] = deepcopy(feature_dict)


        if self.stepnum >= config.maxsteps:
            terminated['__all__'] = True
            truncated['__all__'] = True
        return state, rew, terminated, truncated, {'exechint':ExecHint,'cost_plan_json':CostPlanJson,'min_step':min_step}