import gymnasium as gym
import numpy as np
from copy import deepcopy
from util import min_steps,get_label
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os,random
from pairestimator import PairTrainer
from manager import QueryManager
from planhelper import PlanHelper
import ray
class FOSSEnvBase(gym.Env):
    def __init__(self,env_config):
        self.tablenum = env_config['tableNum']
        self.config   = env_config['genConfig']
        self.action_space_size = int(((self.tablenum + 6) * (self.tablenum - 1)) / 2) # n(n-1)/2 + 3(n - 1)
        self.observation_space = spaces.Dict({
            'x':spaces.Box(-np.inf,np.inf,dtype = np.float32,shape = (self.config.maxnode, self.config.num_node_feature)),
            'attn_bias':spaces.Box(0,1,dtype = np.float32,shape = (self.config.maxnode + 1, self.config.maxnode + 1)),
            'heights':spaces.Box(0,self.config.heightsize,dtype = np.int64,shape = (self.config.maxnode,)),
            'action_mask':spaces.Box(0,1,dtype = np.int32,shape = (self.action_space_size,)),
            'steps':spaces.Box(0,1,dtype = np.float32,shape = (1,))
            })
        self.action_space = spaces.Discrete(self.action_space_size)
        self.action_inteval = [0]
        for i in range(1,self.tablenum):
            self.action_inteval.append(self.action_inteval[-1] + self.tablenum - i)

    def reset(self, seed=None, options=None):
    
        return None,None
    
    def step(self,action):
       
        return None,None,None,None,None

class FOSSEnvTrainBase(gym.Wrapper):
    def __init__(self,env_config) -> None:
        unwrapped_env = FOSSEnvBase(env_config)
        super().__init__(unwrapped_env) 
        self.planhelper  = env_config['planhelper']
        self.querymanger = env_config['querymanger']
        self.pairtrainer = env_config['pairtrainer']
        self.config      = env_config['genConfig']
        self.bpm         = env_config['bestplanmanager']
        self.isCollectSamples = self.config.update_evaluator
        splitpoint = [1.00] + self.config.splitpoint
        self.bouns_weight = [0.00]
        for i in range(len(splitpoint) - 1, 0, -1):
            self.bouns_weight.append((splitpoint[i] + splitpoint[i - 1]) / 2)
    def reset(self, seed=None, options=None):
        self.stepnum = 0
        self.evaluator_update_times = options['evaluator_times']
        self.candidatehint = []
        self.sql, base_train_feature,self.query_id,self.evaluator_best,self.RL_best = self.querymanger.get2train()
        self.baselatency  = self.planhelper.tryGetLatency('',self.query_id)
        self.optimlatency = self.baselatency
        feature_dict,self.hintdict, _ = deepcopy(base_train_feature)
        self.count_table = len(self.hintdict['join order'])
        assert self.count_table == len(self.hintdict['join operator']) + 1
        #=========== init action and init action mask==========
        self.action_mask = np.zeros(self.action_space_size)
        for i in range(self.count_table - 1):
            self.action_mask[self.action_inteval[i]:self.action_inteval[i] + self.count_table - i - 1] = 1
        self.action_mask[-3 * (self.count_table - 1):] = 1
        for i,jo in enumerate(self.hintdict['join operator']):   
            self.action_mask[-3 * i - self.config.OperatorDict[jo]] = 0
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
        self.baseplan = deepcopy(feature_dict)
        
        feature_dict['action_mask'] = self.action_mask
        # self.evaluator_best[0]['action_mask'] = self.action_mask # 
        # self.RL_best[0]['action_mask'] = self.action_mask # 
        self.esbestplan = self.baseplan
        self.esbesthint = ''
        self.beststeps = 0
        self.isswapL = False
        
        self.candidatehint.append(deepcopy(self.hintdict))

        self.bonustobest    = self.config.maxbounty * (self.evaluator_best[1] / self.baselatency)
        self.bestbonus      = self.config.maxbounty - self.bonustobest
        self.bonusignal = 0
        if self.RL_best[1] >= self.baselatency:
            self.comparedplan = self.baseplan
            self.basebonus = 0
            self.bonustoexplore = self.config.maxbounty * 0.5
            self.comparedlatency = self.baselatency
        else:
            self.bonustoexplore = (self.RL_best[1] / self.baselatency) * self.config.maxbounty
            self.basebonus      = self.config.maxbounty - self.bonustoexplore
            self.comparedplan = self.RL_best[0]
            self.comparedlatency = self.RL_best[1]
            self.baseidx = 0
            
        self.is_done = False
        self.truncated = False
        return feature_dict,{}
    def step(self,action):
        self.stepnum += 1
        # =============act on ICP and update action mask===========
        try:
            if action >= (self.unwrapped.tablenum * (self.unwrapped.tablenum - 1)) / 2:
                idx = abs(action - self.unwrapped.action_space_size + 1)
                self.hintdict['join operator'][int(idx / 3)] = self.config.Operatortype[idx % 3]
                for i in range(self.count_table - 1):
                    self.action_mask[self.action_inteval[i]:self.action_inteval[i] + self.count_table - i - 1] = 1
                self.action_mask[-3 * (self.count_table - 1):] = 1
                for i,jo in enumerate(self.hintdict['join operator']):   
                    self.action_mask[-3 * i - self.config.OperatorDict[jo]] = 0
                # self.isswapC = False
            else:
                tag = -1
                # self.isswapC = True
                for i in range(len(self.unwrapped.action_inteval)):
                    if action < self.unwrapped.action_inteval[i]:
                        tag = i
                        break
                if tag != -1:
                    t1 = tag - 1
                    t2 = action - self.unwrapped.action_inteval[t1] + tag
                    temp = self.hintdict['join order'][t1]
                    self.hintdict['join order'][t1] = self.hintdict['join order'][t2]
                    self.hintdict['join order'][t2] = temp 
                    # 第tag个与第action - self.action_inteval[tag - 1] + 1 + tag交换
                    self.action_mask.fill(0)
                    if t1 == 0 or t1 == 1 or t2 == 1:
                        self.action_mask[-3:] = 1
                    else:
                        self.action_mask[-3 * (t1):-3 * (t1 - 1)] = 1
                        self.action_mask[-3 * (t2):-3 * (t2 - 1)] = 1
        except:
            print(self.query_id, self.hintdict, self.action_mask)
            raise ValueError('Action Invalid')
        # ========Determine if there are duplicates=====
        isloop = False
        for hint in self.candidatehint:
            if hint['join order'] == self.hintdict['join order'] and hint['join operator'] == self.hintdict['join operator']:
                isloop = True
                break
        if not isloop:
            self.candidatehint.append(deepcopy(self.hintdict))
        #=====get CP from ICP=========
        exechint = self.planhelper.to_exechint(self.hintdict)
        self.currlatency = self.planhelper.tryGetLatency(exechint,self.query_id)
        feature_dict,_,_,_ = self.planhelper.get_feature(exechint, self.sql, False, query_id = self.query_id)
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
        currplan = deepcopy(feature_dict)
        feature_dict['action_mask'] = self.action_mask
        #=========calculate penalty=======
        # if not self.isswapL:
        minsteps = min_steps(self.candidatehint[0], self.hintdict)
        penalty = (minsteps - self.stepnum) * self.config.penalty_coeff
        # else:
        #     penalty = 0
        # self.isswapL = self.isswapC
        #========== reward style 1===========
        # bounty = 0
        # if self.currlatency != None and self.optimlatency != None:
        #     betteridx = get_label(self.optimlatency, self.currlatency)
        #     if betteridx >= 1:
        #         self.optimlatency = self.currlatency
        #         self.esbesthint   = exechint
        #         self.beststeps    = self.stepnum
        #         self.esbestplan   = currplan
        #         self.bonusignal   = get_label(self.comparedlatency, self.optimlatency)
        #         # if bsidx > self.bonusignal:
        #         #     bonus = bonus + (bsidx - self.bonusignal) 
        #         # self.bonusignal = bsidx

        # else:
        #     betteridx = self.pairtrainer.predict_pair(self.esbestplan, currplan)
        #     if betteridx >= 1:
        #         self.optimlatency = self.currlatency
        #         self.esbestplan   = currplan
        #         self.esbesthint   = exechint
        #         self.beststeps    = self.stepnum
        #         self.bonusignal   = self.pairtrainer.predict_pair(self.comparedplan, self.esbestplan)
                
        # if self.bonusignal == 1:
        #     bounty = bounty + (0.25 * self.bonustoexplore + self.basebonus)# * (self.stepnum / self.config.maxsteps)
        # elif self.bonusignal == 2:
        #     bounty = bounty + (0.75 * self.bonustoexplore + self.basebonus)# * (self.stepnum / self.config.maxsteps)
        # else:
        #     if self.basebonus != 0:
        #         if self.optimlatency != None:
        #             self.baseidx = get_label(self.baselatency, self.optimlatency)
        #         else:
        #             self.baseidx = self.pairtrainer.predict_pair(self.esbestplan, currplan)
        #         bounty = bounty + self.baseidx * (1.0 / 2) * self.basebonus * (3.0 / 4)# * (self.stepnum / self.config.maxsteps) 
        # if self.stepnum >= self.config.maxsteps:
        #     self.is_done = True
        #     self.truncated = True
        #     if self.optimlatency != None:
        #         evaluatorbestidx = get_label(self.evaluator_best[1], self.optimlatency)
        #     else:
        #         evaluatorbestidx = self.pairtrainer.predict_pair(self.evaluator_best[0], self.esbestplan) 
        #     bounty += (evaluatorbestidx * 3)
        #     if self.optimlatency == None:
        #         if evaluatorbestidx >= 1:# self.pairtrainer.predict_pair(self.RL_best[0],self.esbestplan) >= 1:
        #             self.bpm.add_candidatebest.remote(self.query_id,self.esbesthint,self.beststeps)
        # reward = penalty + bounty #+ self.basebonus
        #========== reward style 2================
        # bounty = 0
        # bounscoeff = 0.5
        # if not isloop:
        #     if self.currlatency != None and self.optimlatency != None:
        #         betteridx = get_label(self.optimlatency, self.currlatency)
        #     else:
        #         betteridx = self.pairtrainer.predict_pair(self.esbestplan, currplan)
        #     if betteridx >= 1:
        #         self.optimlatency = self.currlatency
        #         self.esbesthint   = exechint
        #         self.beststeps    = self.stepnum
        #         self.esbestplan   = currplan
        #     if self.stepnum < self.config.maxsteps:
        #         if self.currlatency != None:
        #             bonusignal = get_label(self.comparedlatency, self.currlatency)
        #             evaluator_idx = get_label(self.evaluator_best[1],self.currlatency) 
        #         else:
        #             bonusignal = self.pairtrainer.predict_pair(self.comparedplan, currplan)
        #             evaluator_idx = self.pairtrainer.predict_pair(self.evaluator_best[0],currplan) 
        #         if evaluator_idx == 1:
        #             bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
        #         elif evaluator_idx == 2:
        #             bounty = bounty + (0.85 * self.bonustobest + self.bestbonus)
        #         else:
        #             if bonusignal != 0:
        #                 bounty = bounty + (bonusignal * (1.0 / 2) * (self.bestbonus - self.basebonus) + self.basebonus)
        #             else:
        #                 if self.basebonus != 0:
        #                     if self.currlatency != None:
        #                         baseidx = get_label(self.baselatency, self.currlatency)
        #                     else:
        #                         baseidx = self.pairtrainer.predict_pair(self.baseplan, currplan)
        #                     bounty = bounty + baseidx * (1.0 / 2) * self.basebonus # * (3.0 / 4) 
        #         bounty = 0.2 * bounty
        # # bounty = betteridx * bounscoeff
        # if self.stepnum >= self.config.maxsteps:
        #     self.is_done = True
        #     self.truncated = True
        #     if self.optimlatency != None:
        #         self.bonusignal = get_label(self.comparedlatency, self.optimlatency)
        #         evaluator_idx = get_label(self.evaluator_best[1],self.optimlatency) 
        #     else:
        #         self.bonusignal = self.pairtrainer.predict_pair(self.comparedplan, self.esbestplan)
        #         evaluator_idx = self.pairtrainer.predict_pair(self.evaluator_best[0],self.esbestplan) 
        #     if evaluator_idx == 1:
        #         bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
        #     elif evaluator_idx == 2:
        #         bounty = bounty + (0.85 * self.bonustobest + self.bestbonus)
        #     else:
        #         if self.bonusignal != 0:
        #             bounty = bounty + (self.bonusignal * (1.0 / 2) * (self.bestbonus - self.basebonus) + self.basebonus)
        #         # if self.bonusignal == 1:
        #         #     bounty = bounty + (0.25 * self.bonustoexplore + self.basebonus)
        #         # elif self.bonusignal == 2:
        #         #     bounty = bounty + (0.85 * self.bonustoexplore + self.basebonus)
        #         else:
        #             if self.basebonus != 0:
        #                 if self.optimlatency != None:
        #                     self.baseidx = get_label(self.baselatency, self.optimlatency)
        #                 else:
        #                     self.baseidx = self.pairtrainer.predict_pair(self.baseplan, self.esbestplan)
        #                 bounty = bounty + self.baseidx * (1.0 / 2) * self.basebonus # * (3.0 / 4) 
        #     if self.optimlatency == None:            
        #         if evaluator_idx >= 1:
        #             self.bpm.add_candidatebest.remote(self.query_id,self.esbesthint,self.beststeps)

        #     # bounty += evaluator_idx * (self.self.baselatency)
        # reward = penalty + bounty #+ self.basebonus
        #========== reward style 3===========
        bounty = 0
        if not isloop:
            # if self.currlatency == None and random.random() <= 0.1 and self.evaluator_update_times <= 2:
                # if random.random() <= 0.05:
                # self.bpm.add_candidatebest.remote(self.query_id, exechint, currplan, self.sql)

            if self.currlatency != None and self.optimlatency != None:
                betteridx = get_label(self.optimlatency, self.currlatency)
            else:
                betteridx, prob = self.pairtrainer.predict_pair(self.esbestplan, currplan)
            if betteridx >= 1:
                self.optimlatency = self.currlatency
                self.esbesthint   = exechint
                self.beststeps    = minsteps
                self.esbestplan   = currplan
            if self.currlatency == None and self.isCollectSamples:
                self.bpm.add_iterCandidate.remote(self.query_id, exechint, currplan, self.sql, prob)
            # if self.currlatency != None:
            #     evaluator_idx = get_label(self.currlatency,self.evaluator_best[1]) 
            # else:
            #     evaluator_idx = self.pairtrainer.predict_pair(currplan,self.evaluator_best[0]) 
            # if evaluator_idx == 0:
            #     if self.currlatency != None:
            #         evaluator_idx = get_label(self.evaluator_best[1],self.currlatency) 
            #     else:
            #         evaluator_idx = self.pairtrainer.predict_pair(self.evaluator_best[0], currplan) 
            #     if evaluator_idx == 1:
            #         bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
            #     elif evaluator_idx == 2:
            #         bounty = bounty + (0.85 * self.bonustobest + self.bestbonus)
            #     else:
            #         bounty = bounty + self.bestbonus
            # else:
            #     if self.currlatency != None:
            #         Out_RL_idx = get_label(self.currlatency, self.comparedlatency)
            #     else:
            #         Out_RL_idx = self.pairtrainer.predict_pair(currplan, self.comparedplan)
            #     if Out_RL_idx == 0:
            #         if self.currlatency != None:
            #             RL_idx = get_label(self.comparedlatency, self.currlatency)
            #         else:
            #             RL_idx = self.pairtrainer.predict_pair(self.comparedplan, currplan)
            #         if RL_idx == 1:
            #             bounty = bounty + (0.25 * (self.bestbonus - self.basebonus) + self.basebonus)
            #         elif RL_idx == 2:
            #             bounty = bounty + (0.85 * (self.bestbonus - self.basebonus) + self.basebonus)
            #         else:
            #             bounty = bounty + self.basebonus
            #         # bounty = bounty + (bonusignal * (1.0 / 2) * (self.bestbonus - self.basebonus) + self.basebonus)
            #     else:
            #         if self.basebonus != 0:
            #             if self.currlatency != None:
            #                 base_idx = get_label(self.baselatency, self.currlatency)
            #             else:
            #                 base_idx = self.pairtrainer.predict_pair(self.baseplan, currplan)
            #             # bounty = bounty + baseidx * (1.0 / 2) * self.basebonus # * (3.0 / 4) 
            #             if base_idx == 1:
            #                 bounty = bounty + (0.25 * self.basebonus)
            #             elif base_idx == 2:
            #                 bounty = bounty + (0.85 * self.basebonus)
            bounty = betteridx * 0.5
        #bounty = 0.2 * bounty
        # bounty = betteridx * bounscoeff
        if self.stepnum >= self.config.maxsteps:
            self.is_done = True
            self.truncated = True
            isvalidate = False
            if self.optimlatency != None:
                Out_evaluator_idx = get_label(self.optimlatency,self.evaluator_best[1]) 
            else:
                Out_evaluator_idx,prob = self.pairtrainer.predict_pair(self.esbestplan,self.evaluator_best[0]) 
            if Out_evaluator_idx == 0:
                if self.optimlatency != None:
                    evaluator_idx = get_label(self.evaluator_best[1],self.optimlatency) 
                else:
                    evaluator_idx,prob = self.pairtrainer.predict_pair(self.evaluator_best[0], self.esbestplan) 
                
                if evaluator_idx != 0:
                    bounty = bounty + (self.bouns_weight[evaluator_idx] * self.bonustobest + self.bestbonus)
                    isvalidate = True
                # elif evaluator_idx == 2:
                #     bounty = bounty + (0.85 * self.bonustobest + self.bestbonus)
                #     isvalidate = True
                else:
                    bounty = bounty + self.bestbonus
            else:
                if self.optimlatency != None:
                    Out_RL_idx = get_label(self.optimlatency, self.comparedlatency)
                else:
                    Out_RL_idx,prob = self.pairtrainer.predict_pair(self.esbestplan, self.comparedplan)
                if Out_RL_idx == 0:
                    if self.optimlatency != None:
                        RL_idx = get_label(self.comparedlatency, self.optimlatency)
                    else:
                        RL_idx,prob = self.pairtrainer.predict_pair(self.comparedplan, self.esbestplan)

                    bounty = bounty + (self.bouns_weight[RL_idx] * (self.bestbonus - self.basebonus) + self.basebonus)
                    # if RL_idx == 1:
                    #     bounty = bounty + (0.25 * (self.bestbonus - self.basebonus) + self.basebonus)
                    # elif RL_idx == 2:
                    #     bounty = bounty + (0.85 * (self.bestbonus - self.basebonus) + self.basebonus)
                    # else:
                    #     bounty = bounty + self.basebonus
                    # bounty = bounty + (bonusignal * (1.0 / 2) * (self.bestbonus - self.basebonus) + self.basebonus)
                else:
                    if self.basebonus != 0:
                        if self.optimlatency != None:
                            base_idx = get_label(self.baselatency, self.optimlatency)
                        else:
                            base_idx,prob = self.pairtrainer.predict_pair(self.baseplan, self.esbestplan)
                        # bounty = bounty + baseidx * (1.0 / 2) * self.basebonus # * (3.0 / 4) 
                        bounty = bounty + (self.bouns_weight[base_idx] * self.basebonus)
                        # if base_idx == 1:
                        #     bounty = bounty + (0.25 * self.basebonus)
                        # elif base_idx == 2:
                        #     bounty = bounty + (0.85 * self.basebonus)
            # if self.optimlatency == None:            
            #     if isvalidate and self.isCollectGlobal:
            #         self.bpm.add_globalCandidate.remote(self.query_id,self.esbesthint,self.esbestplan,self.sql)

        reward = penalty + bounty #+ self.basebonus
        return feature_dict, reward, self.is_done,self.truncated,{}


class FOSSEnvTrain(MultiAgentEnv):

    def __init__(self,out_config):
        super().__init__()
        self.config      = out_config['genConfig']
        self.pairtrainer = PairTrainer(self.config, device='cpu')
        self.planhelper  = PlanHelper(self.config)
        self.querymanger = QueryManager(self.config, planhelper = self.planhelper,isremote=False)
        self.bpm         = out_config['bestplanmanager']
        evaluator_esbest_feature = ray.get(self.bpm.get_evaluator_esbest.remote())
        self.querymanger.update_evaluator_esbest(evaluator_esbest_feature)
        median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
        self.querymanger.update_Median(median_hint_latency)
        # RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
        # self.querymanger.update_RL_esbest(RL_esbest_feature)
        if os.path.exists(self.config.evaluator_path):
            self.pairtrainer.load_model(self.config.evaluator_path)
            self.evaluatorversion = os.path.getmtime(self.config.evaluator_path)
        else:
            self.evaluatorversion = None
        tableNum = self.planhelper.get_table_num()
        env_config = {'pairtrainer':self.pairtrainer,'querymanger':self.querymanger,
                      'bestplanmanager':self.bpm,'planhelper':self.planhelper,'tableNum':tableNum, 'genConfig':self.config}
        self.agents = [FOSSEnvTrainBase(env_config) for _ in range(self.config.num_agents)]
        self._agent_ids = set(range(self.config.num_agents))
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.evaluator_update_times = 0
        self.resetted = False
    def reset(self, *, seed=None, options=None):
        if not self.resetted:
            self.resetted = True
            return {},{}
        super().reset(seed=seed)
        if os.path.exists(self.config.evaluator_path):
            now_version = os.path.getmtime(self.config.evaluator_path)
            if now_version != self.evaluatorversion:
                queryImportance = ray.get(self.bpm.update_weightsByRLesbest.remote())
                self.querymanger.updateBuffer(queryImportance) 
                latencyBuffer = ray.get(self.bpm.get_latencyBuffer.remote())
                self.planhelper.updatePGLatencyBuffer(latencyBuffer)
                evaluator_esbest_feature = ray.get(self.bpm.get_evaluator_esbest.remote())
                self.querymanger.update_evaluator_esbest(evaluator_esbest_feature)
                median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
                self.querymanger.update_Median(median_hint_latency)
                # RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
                # self.querymanger.update_RL_esbest(RL_esbest_feature)
                self.pairtrainer.load_model(self.config.evaluator_path)
                # print('Load New evaluator|Now VersionTime:{}'.format(now_version))
                self.evaluatorversion = now_version
                self.evaluator_update_times += 1
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        reset_results = [a.reset(options={'evaluator_times':self.evaluator_update_times}) for a in self.agents]
        # self.epi += 1
        return (
            {i: oi[0] for i, oi in enumerate(reset_results)},
            {i: oi[1] for i, oi in enumerate(reset_results)},
        )

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info


class FOSSEnvTest(MultiAgentEnv):
    def __init__(self,env_config):
        super().__init__()
        self.planhelper = env_config['planhelper']
        self.config     = env_config['genConfig']
        self.numagents  = self.config.num_agents
        
        tableNum = ray.get(self.planhelper.GetTableNum.remote())
        self.unwrapped_env = FOSSEnvBase({'tableNum':tableNum, 'genConfig':self.config})
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

        feature_dict,self.hintdict,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('',self.sql,True,query_id = self.query_id))
        self.plantime = cost_plan_json['Planning Time']

        self.count_table = len(self.hintdict['join order'])
        
        if self.count_table > 2:
            assert self.count_table == len(self.hintdict['join operator']) + 1
            self.use_FOSS = True
        else:
            self.use_FOSS = False 
            print('BAN FOSS')
        
        # init action
        self.action_mask = np.zeros(self.unwrapped_env.action_space_size)
        for i in range(self.count_table - 1):
            self.action_mask[ self.unwrapped_env.action_inteval[i]: self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1
        self.action_mask[-3 * (self.count_table - 1):] = 1
        for i,jo in enumerate(self.hintdict['join operator']):   
            self.action_mask[-3 * i - self.config.OperatorDict[jo]] = 0
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
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
                self.hintdict_total[agent_id]['join operator'][int(idx/3)] = self.config.Operatortype[idx % 3]
                for i in range(self.count_table - 1):
                    self.action_mask_tatal[agent_id][self.unwrapped_env.action_inteval[i]:self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1
                self.action_mask_tatal[agent_id][-3 * (self.count_table - 1):] = 1
                for i,jo in enumerate(self.hintdict_total[agent_id]['join operator']):   
                    self.action_mask_tatal[agent_id][-3 * i - self.config.OperatorDict[jo]] = 0
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
                    
            exechint = ray.get(self.planhelper.GetExechint.remote(self.hintdict_total[agent_id]))
            # latency_timeout,_ = self.planhelper.getLatency(exechint,self.sql,self.query_id)
            # print('test_{}_agent_{}_step_{},Latency:{:.4f}'.format(self.query_id,agent_id,self.stepnum,latency_timeout[0]))
            feature_dict,_,_,cost_plan_json = ray.get(self.planhelper.GetFeature.remote(exechint,self.sql,False,query_id = self.query_id))
            self.plantime += cost_plan_json['Planning Time']
            feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
            state[agent_id] = feature_dict.copy()
            state[agent_id]['action_mask'] = self.action_mask_tatal[agent_id]
            info[agent_id] = {'hint':exechint}
            rew[agent_id] = 0
        if self.stepnum >= self.config.maxsteps:
            terminated['__all__'] = True
            truncated['__all__'] = True

        return state, rew, terminated, truncated, info