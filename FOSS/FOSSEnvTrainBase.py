import gymnasium as gym
from FOSSEnvBase import FOSSEnvBase
import numpy as np
from config import Config
from copy import deepcopy
from util import min_steps,get_label
config = Config()
class FOSSEnvTrainBase(gym.Wrapper):
    def __init__(self,env_config) -> None:
        unwrapped_env = FOSSEnvBase(env_config)
        super().__init__(unwrapped_env) 
        self.planhelper  = env_config['planhelper']
        self.querymanger = env_config['querymanger']
        self.pairtrainer = env_config['pairtrainer']
        self.bpm = env_config['bestplanmanager']
    def reset(self, seed=None, options=None):
        self.stepnum = 0
        # self.bonusPlus = options['bonusPlus']
        self.candidatehint = []
        self.sql, base_train_feature,self.query_id,self.AAM_best,self.RL_best = self.querymanger.get2train()
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
            self.action_mask[-3 * i - config.OperatorDict[jo]] = 0
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
        self.baseplan = deepcopy(feature_dict)
        
        feature_dict['action_mask'] = self.action_mask
        # self.AAM_best[0]['action_mask'] = self.action_mask # 
        # self.RL_best[0]['action_mask'] = self.action_mask # 
        self.esbestplan = self.baseplan
        self.esbesthint = ''
        self.beststeps = 0
        self.isswapL = False
        
        self.candidatehint.append(deepcopy(self.hintdict))

        self.bonustobest    = config.maxbounty * (self.AAM_best[1] / self.baselatency)
        self.bestbonus      = config.maxbounty - self.bonustobest
        self.bonusignal = 0
        if self.RL_best[1] >= self.baselatency:
            self.comparedplan = self.baseplan
            self.basebonus = 0
            self.bonustoexplore = config.maxbounty * 0.5
            self.comparedlatency = self.baselatency
        else:
            self.bonustoexplore = (self.RL_best[1] / self.baselatency) * config.maxbounty
            self.basebonus      = config.maxbounty - self.bonustoexplore
            self.comparedplan = self.RL_best[0]
            self.comparedlatency = self.RL_best[1]
            self.baseidx = 0
            
        self.is_done = False
        self.truncated = False
        return feature_dict,{}
    def step(self,action):
        self.stepnum += 1
        # =============act on ICP and update action mask===========
        if action >= (self.unwrapped.tablenum * (self.unwrapped.tablenum - 1)) / 2:
            idx = abs(action - self.unwrapped.action_space_size + 1)
            self.hintdict['join operator'][int(idx/3)] = config.Operatortype[idx%3]
            for i in range(self.count_table - 1):
                self.action_mask[self.action_inteval[i]:self.action_inteval[i] + self.count_table - i - 1] = 1
            self.action_mask[-3 * (self.count_table - 1):] = 1
            for i,jo in enumerate(self.hintdict['join operator']):   
                self.action_mask[-3 * i - config.OperatorDict[jo]] = 0
            # self.isswapC = False
        # 如果是交换两个表的顺序
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
        feature_dict,_,_,_ = self.planhelper.get_feature(exechint, self.sql, False,query_id=self.query_id)
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / config.maxsteps])
        currplan = deepcopy(feature_dict)
        feature_dict['action_mask'] = self.action_mask
        #=========calculate penalty=======
        # if not self.isswapL:
        minsteps = min_steps(self.candidatehint[0], self.hintdict)
        penalty = (minsteps - self.stepnum) * 2
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
        #     bounty = bounty + (0.25 * self.bonustoexplore + self.basebonus)# * (self.stepnum / config.maxsteps)
        # elif self.bonusignal == 2:
        #     bounty = bounty + (0.75 * self.bonustoexplore + self.basebonus)# * (self.stepnum / config.maxsteps)
        # else:
        #     if self.basebonus != 0:
        #         if self.optimlatency != None:
        #             self.baseidx = get_label(self.baselatency, self.optimlatency)
        #         else:
        #             self.baseidx = self.pairtrainer.predict_pair(self.esbestplan, currplan)
        #         bounty = bounty + self.baseidx * (1.0 / 2) * self.basebonus * (3.0 / 4)# * (self.stepnum / config.maxsteps) 
        # if self.stepnum >= config.maxsteps:
        #     self.is_done = True
        #     self.truncated = True
        #     if self.optimlatency != None:
        #         AAMbestidx = get_label(self.AAM_best[1], self.optimlatency)
        #     else:
        #         AAMbestidx = self.pairtrainer.predict_pair(self.AAM_best[0], self.esbestplan) 
        #     bounty += (AAMbestidx * 3)
        #     if self.optimlatency == None:
        #         if AAMbestidx >= 1:# self.pairtrainer.predict_pair(self.RL_best[0],self.esbestplan) >= 1:
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
        #     if self.stepnum < config.maxsteps:
        #         if self.currlatency != None:
        #             bonusignal = get_label(self.comparedlatency, self.currlatency)
        #             AAM_idx = get_label(self.AAM_best[1],self.currlatency) 
        #         else:
        #             bonusignal = self.pairtrainer.predict_pair(self.comparedplan, currplan)
        #             AAM_idx = self.pairtrainer.predict_pair(self.AAM_best[0],currplan) 
        #         if AAM_idx == 1:
        #             bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
        #         elif AAM_idx == 2:
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
        # if self.stepnum >= config.maxsteps:
        #     self.is_done = True
        #     self.truncated = True
        #     if self.optimlatency != None:
        #         self.bonusignal = get_label(self.comparedlatency, self.optimlatency)
        #         AAM_idx = get_label(self.AAM_best[1],self.optimlatency) 
        #     else:
        #         self.bonusignal = self.pairtrainer.predict_pair(self.comparedplan, self.esbestplan)
        #         AAM_idx = self.pairtrainer.predict_pair(self.AAM_best[0],self.esbestplan) 
        #     if AAM_idx == 1:
        #         bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
        #     elif AAM_idx == 2:
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
        #         if AAM_idx >= 1:
        #             self.bpm.add_candidatebest.remote(self.query_id,self.esbesthint,self.beststeps)

        #     # bounty += AAM_idx * (self.self.baselatency)
        # reward = penalty + bounty #+ self.basebonus
        #========== reward style 3===========
        bounty = 0
        if not isloop:
            if self.currlatency != None and self.optimlatency != None:
                betteridx = get_label(self.optimlatency, self.currlatency)
            else:
                betteridx = self.pairtrainer.predict_pair(self.esbestplan, currplan)
            if betteridx >= 1:
                self.optimlatency = self.currlatency
                self.esbesthint   = exechint
                self.beststeps    = minsteps
                self.esbestplan   = currplan
            # if self.currlatency != None:
            #     AAM_idx = get_label(self.currlatency,self.AAM_best[1]) 
            # else:
            #     AAM_idx = self.pairtrainer.predict_pair(currplan,self.AAM_best[0]) 
            # if AAM_idx == 0:
            #     if self.currlatency != None:
            #         AAM_idx = get_label(self.AAM_best[1],self.currlatency) 
            #     else:
            #         AAM_idx = self.pairtrainer.predict_pair(self.AAM_best[0], currplan) 
            #     if AAM_idx == 1:
            #         bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
            #     elif AAM_idx == 2:
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
        if self.stepnum >= config.maxsteps:
            self.is_done = True
            self.truncated = True
            isvalidate = False
            if self.optimlatency != None:
                Out_AAM_idx = get_label(self.optimlatency,self.AAM_best[1]) 
            else:
                Out_AAM_idx = self.pairtrainer.predict_pair(self.esbestplan,self.AAM_best[0]) 
            if Out_AAM_idx == 0:
                if self.optimlatency != None:
                    AAM_idx = get_label(self.AAM_best[1],self.optimlatency) 
                else:
                    AAM_idx = self.pairtrainer.predict_pair(self.AAM_best[0], self.esbestplan) 
                if AAM_idx == 1:
                    bounty = bounty + (0.25 * self.bonustobest + self.bestbonus)
                    isvalidate = True
                elif AAM_idx == 2:
                    bounty = bounty + (0.85 * self.bonustobest + self.bestbonus)
                    isvalidate = True
                else:
                    bounty = bounty + self.bestbonus
            else:
                if self.optimlatency != None:
                    Out_RL_idx = get_label(self.optimlatency, self.comparedlatency)
                else:
                    Out_RL_idx = self.pairtrainer.predict_pair(self.esbestplan, self.comparedplan)
                if Out_RL_idx == 0:
                    if self.optimlatency != None:
                        RL_idx = get_label(self.comparedlatency, self.optimlatency)
                    else:
                        RL_idx = self.pairtrainer.predict_pair(self.comparedplan, self.esbestplan)
                    if RL_idx == 1:
                        bounty = bounty + (0.25 * (self.bestbonus - self.basebonus) + self.basebonus)
                    elif RL_idx == 2:
                        bounty = bounty + (0.85 * (self.bestbonus - self.basebonus) + self.basebonus)
                    else:
                        bounty = bounty + self.basebonus
                    # bounty = bounty + (bonusignal * (1.0 / 2) * (self.bestbonus - self.basebonus) + self.basebonus)
                else:
                    if self.basebonus != 0:
                        if self.optimlatency != None:
                            base_idx = get_label(self.baselatency, self.optimlatency)
                        else:
                            base_idx = self.pairtrainer.predict_pair(self.baseplan, self.esbestplan)
                        # bounty = bounty + baseidx * (1.0 / 2) * self.basebonus # * (3.0 / 4) 
                        if base_idx == 1:
                            bounty = bounty + (0.25 * self.basebonus)
                        elif base_idx == 2:
                            bounty = bounty + (0.85 * self.basebonus)
            if self.optimlatency == None:            
                if isvalidate:
                    self.bpm.add_candidatebest.remote(self.query_id,self.esbesthint,self.beststeps)

        reward = penalty + bounty #+ self.basebonus
        return feature_dict, reward, self.is_done,self.truncated,{}

