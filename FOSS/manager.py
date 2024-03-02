import random
import json
import os
from planhelper import PlanHelper
import numpy as np
from copy import deepcopy
import pandas as pd
import ray
import time
import math
from util import get_median
# @ray.remote
class QueryManager:
    def __init__(self, globalconfig, planhelper = None, isremote = True):
        self.isremote = isremote
        if planhelper != None:
            self.planhelper = planhelper
        else:
            self.planhelper = PlanHelper(globalconfig)
            isremote        = False
        self.config      = globalconfig
        self.train_dir   = os.listdir(self.config.train_workload_path)
        self.test_dir    = os.listdir(self.config.test_workload_path)
        self.trainSet    = pd.DataFrame(columns=['tablenum','sql','base_train_feature'])
        self.validateSet = pd.DataFrame(columns=['sql'])
        self.testSet     = pd.DataFrame(columns=['sql'])
        self.AAM_esbest_feature = {}
        self.RL_esbest_feature  = {}
        self.median_feature     = {}
        for i in range(len(self.train_dir)):
            with open(self.config.train_workload_path + self.train_dir[i],'r') as f:
                sql = f.read()
            query_id = self.train_dir[i].split('.')[0]
            self.validateSet.loc[query_id] = [sql]
            if isremote:
                feature_dict,hintdict,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,True,query_id = query_id))
            else:
                feature_dict,hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',sql,True,query_id = query_id)
            if left_deep:
                feature_dict['steps']   = np.array([0])
                cost_plan_json['steps'] = 0
                self.trainSet.loc[query_id] = [len(hintdict['join order']), sql, (feature_dict, hintdict, cost_plan_json)]
            else:
                print(f'Query:{query_id} is not left-deep ')
        for i in range(len(self.test_dir)):
            with open(self.config.test_workload_path + self.test_dir[i],'r') as f:
                sql = f.read()
            query_id = self.test_dir[i].split('.')[0]
            self.testSet.loc[query_id] = [sql]
        self.numOfTrain = len(self.trainSet.index)
        self.cur_test = 0
        self.numOfTest = len(self.testSet.index)
        self.cur_validate = 0
        self.numOfvalidate = len(self.validateSet.index)
        self.buffer = list(range(self.numOfTrain))

        if not os.path.exists(self.config.encoding_path):
            if isremote:
                self.planhelper.SaveEncoding.remote(self.config.encoding_path)
            else:
                self.planhelper.encoding.save_to_file(self.config.encoding_path)

    def get2eval(self):
        query_id = self.testSet.index[self.cur_test]
        oneloop = False
        self.cur_test = (self.cur_test + 1) % self.numOfTest
        if self.cur_test == 0:
            oneloop = True
        return self.testSet.loc[query_id,'sql'], query_id, oneloop
    
    def get2validate(self):
        query_id = self.validateSet.index[self.cur_validate]
        oneloop = False
        self.cur_validate = (self.cur_validate + 1) % self.numOfvalidate
        if self.cur_validate == 0:
            oneloop = True
        return self.validateSet.loc[query_id,'sql'], query_id, oneloop
        
    # def get2train(self):
    #     query_id = random.choices(self.buffer)[0]
    #     return self.trainSet.loc[query_id,'sql'],self.trainSet.loc[query_id,'base_train_feature'],\
    #             query_id, self.AAM_esbest_feature[query_id],\
    #             self.RL_esbest_feature[query_id]
    def get2train(self):
        query_id = random.choices(self.buffer)[0]
        return self.trainSet.loc[query_id,'sql'],self.trainSet.loc[query_id,'base_train_feature'],\
                query_id, self.AAM_esbest_feature[query_id],\
                self.median_feature[query_id]
    
    def updateBuffer(self,queryImportance):
        self.buffer = []
        for k, v in queryImportance.items():
            self.buffer.extend([k] * v)
    
    def update_AAM_esbest(self,AAM_esbest_feature):
        self.AAM_esbest_feature = deepcopy(AAM_esbest_feature)

    def update_Median(self,median_hint_latency):
        for query_id,hint_latency in median_hint_latency.items():
            if self.isremote:
                feature_dict,_,_,_ = ray.get(self.planhelper.GetFeature.remote(hint_latency['hint'],self.validateSet.loc[query_id,'sql'],False,query_id = query_id))
            else:
                feature_dict,_,_,_ = self.planhelper.get_feature(hint_latency['hint'],self.validateSet.loc[query_id,'sql'],False,query_id = query_id)
            feature_dict['steps'] = np.array([0])
            self.median_feature[query_id] = (feature_dict, hint_latency['latency'])

    def update_RL_esbest(self,RL_esbest_feature):
        self.RL_esbest_feature = deepcopy(RL_esbest_feature)
    
class ResultManager():
    def __init__(self, genConfig, writer):
        self.config     = genConfig
        self.resultFile = open(self.config.outfile_path,'w')
        self.expPool    = open(self.config.ExperiencePool,'w')
        self.bestplan   = open(self.config.beststeps_record,'a') # TO_Delete
        self.testQuery  = []
        self.trainQuery = []
        self.writer     = writer
        self.output     = {}
        self.timeRecord = time.time()

    def recordQuery(self,queryId, istest):
        if istest:
            self.testQuery.append(queryId)
        else:
            self.trainQuery.append(queryId)

    def recordRuning(self,key,value):
        self.resultFile.write(json.dumps([key,value])+"\n")
        self.resultFile.flush()

    def recordExp(self,queryId, hint, agentNo, steps):
        self.expPool.write(json.dumps([queryId,'|'.join([hint,str(agentNo),str(steps)])])+"\n")
        self.expPool.flush()

    def recordTime(self,name):
        _time = str(round(time.time() - self.timeRecord, 3))
        self.timeRecord = time.time()
        self.resultFile.write(json.dumps([name,_time])+"\n")
        self.resultFile.flush()

    def recordeval(self,queryId, latency):
        latency = round(latency, 3)
        if queryId not in self.output:
            self.output[queryId] = [latency]
        else:
            self.output[queryId].append(latency)

    def recordwrl(self, valIter):
        wrl_test   = 0
        wrl_train  = 0
        total_test  = 0
        total_train = 0
        for k,v in self.output.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in self.testQuery:
                    wrl_test    += v[-1]
                    total_test  += v[0]
                elif k in self.trainQuery:
                    wrl_train   += v[-1]
                    total_train += v[0]
        if total_test != 0:
            self.writer.add_scalar('TestWorkload',wrl_test,valIter)
            wrl_test = wrl_test / total_test
            self.recordeval("Test_WRL", wrl_test)
            self.writer.add_scalar('Test_WRL',wrl_test,valIter)
        if total_train != 0:
            self.writer.add_scalar('TrainWorkload',wrl_train,valIter)
            wrl_train = wrl_train / total_train
            self.recordeval("Train_WRL", wrl_train)
            self.writer.add_scalar('Train_WRL',wrl_train,valIter)

    def recordgmrl(self, valIter):
        gmrl_train   = 1
        gmrl_test    = 1
        counts_train = 0
        counts_test  = 0
        for k,v in self.output.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in self.testQuery:
                    gmrl_test = gmrl_test * (v[-1] / v[0])
                    counts_test += 1
                elif k in self.trainQuery:
                    gmrl_train = gmrl_train * (v[-1] / v[0])
                    counts_train += 1
        if counts_test != 0:
            gmrl_test = pow(gmrl_test, 1 / counts_test)
            self.recordeval("Test_GMRL", gmrl_test)
            self.writer.add_scalar('Test_GMRL',gmrl_test,valIter)
        if counts_train != 0:
            gmrl_train = pow(gmrl_train, 1 / counts_train)
            self.recordeval("Train_GMRL", gmrl_train)
            self.writer.add_scalar('Train_GMRL',gmrl_train,valIter)

    def recordMetric(self,valIter):
        self.recordwrl(valIter)
        self.recordgmrl(valIter)
    def recordBestPlanSteps(self,query_id,steps,latency):
        self.bestplan.write(json.dumps([query_id, '|'.join([str(steps),str(round(latency,3))])]) + '\n') # TO_Delete
        self.bestplan.flush()
    def writeout(self):
        with open(self.config.eval_output_path, "w") as out:
            for k,v in self.output.items():
                out.write(json.dumps([k, v]) + '\n')
                out.flush()
    def close(self):
        self.resultFile.close()
        self.expPool.close()

@ray.remote
class BestPlanManager:
    def __init__(self, genConfig):
        self.config = genConfig
        self.candidateBest = pd.DataFrame(columns = ['queryid', 'hint','feature','sql'])
        self.balances = pd.DataFrame(columns = ['queryid', 'hint','feature','sql'])
        self.sampleNum = self.config.maxsamples
        self.currNo    = 0
        self.masked    = {}
        self.AAM_esbest_feature = {}
        self.RL_estbest = {}
        self.medianplan = {}
        self.queryImportance = {}
        self.latencyBuffer = {}
        self.endSignal = False
        self.validationSet = None
        self.coeff_1 = 0.4
        self.coeff_2 = 0.6

    def update_AAM_esbest(self,query_id,feature_dict, exectime):
        self.AAM_esbest_feature[query_id] = (feature_dict,exectime)

    def add_candidatebest(self,query_id,hint,feature_dict,sql):
        self.candidateBest.loc[self.currNo] = \
        {'queryid':query_id,'hint':hint,'feature':feature_dict,'sql':sql}
        self.currNo += 1

    def add_balances(self,query_id,hint,feature_dict,sql):
        self.balances.loc[len(self.balances)] = \
        {'queryid':query_id,'hint':hint,'feature':feature_dict,'sql':sql}

    def get_balances(self):
        if len(self.balances) > 0:
            samples = self.balances.sample(frac=1)
            self.balances = pd.DataFrame(columns = self.balances.columns)
            return samples
        else:
            return None
        
    def get_currNo(self):
        return self.currNo
    
    def update_validationSet(self,validationSet):
        self.validationSet = validationSet

    def get_candidatePair(self):
        # self.candidatebest.drop_duplicates(subset=['queryid', 'hint', 'steps'],keep='first').reset_index(drop=True)
        inputs = []
        idxlist = []
        for idx, samples in self.candidateBest.iterrows():
            inputs.append({'left': self.AAM_esbest_feature[samples['queryid']][0],'right':samples['feature']})
            idxlist.append(idx)
        return idxlist, inputs, self.currNo
    
    def get_AAM_esbest(self,query_id = None):
        if query_id == None:
            return self.AAM_esbest_feature
        else:
            return self.AAM_esbest_feature[query_id]
        
    def get_AAM_best(self):
        total = 0
        best_steps = {}
        for k,v in self.AAM_esbest_feature.items():
            total += v[1]
            best_steps[k] = v[0]['steps']
        return total, best_steps
    
    def update_RL_esbest(self,query_id,featuredict,exectime):
         self.RL_estbest[query_id] = (featuredict, exectime)

    def get_RL_esbest(self,query_id = None):
        if query_id == None:
            return self.RL_estbest
        else:
            return self.RL_estbest[query_id]
    def get_median_plan(self):
        for k in self.AAM_esbest_feature:
            name = k
            hint_latency = self.latencyBuffer[name]
            baselatency = hint_latency[''][0]
            hints  = ['']
            latency= [baselatency]
            for hint,lt in hint_latency.items():
                if lt[0] < baselatency:
                    hints.append(hint)
                    latency.append(lt[0])
            median_value, median_hint = get_median(latency,hints)
            self.medianplan[name] = {'hint':median_hint, 'latency':median_value}
        return self.medianplan
        
    def get_candidatebest(self):
        if self.validationSet == None:
            self.validationSet = list(self.candidateBest.index)

        if len(self.validationSet) >= self.sampleNum:
            sample_idx = random.sample(self.validationSet, self.sampleNum)
            samples = self.candidateBest.loc[sample_idx]
            self.candidateBest.drop(sample_idx, inplace=True)
        else:
            random.shuffle(self.validationSet)
            samples = self.candidateBest.loc[self.validationSet]
            self.candidateBest.drop(self.validationSet, inplace=True)

        self.validationSet = []
        return samples
    
    def update_weightsByRLesbest(self):
        # decay = 0.05
        # self.coeff_1 = min(1.0, self.coeff_1 + decay)
        # self.coeff_2 = max(0.0, self.coeff_2 - decay)
        for k,v in self.RL_estbest.items():
            weights_1 = max(1, v[1] - self.AAM_esbest_feature[k][1])
            weights_2 = max(1, v[1])
            # weights = 1
            if k not in self.masked or not self.masked[k]:
                self.queryImportance[k] = int(self.coeff_1 * (2 ** (math.floor(math.log10(weights_1)))) + self.coeff_2 * (2 ** (math.floor(math.log10(weights_2)))))
            else:
                self.queryImportance[k] = 0
        return self.queryImportance
    
    def getqueryImportance(self):
        return self.queryImportance
    
    def update_latencyBuffer(self,latencybuffer):
        self.latencyBuffer = latencybuffer

    def get_latencyBuffer(self):
        return self.latencyBuffer
    
    def update_schedule(self,endSignal):
        self.endSignal = endSignal

    def get_schedule(self):
        return self.endSignal
    
    def updateMask(self, toTrain = None, toMask = None):
        if toTrain:
            for k in toTrain:
                self.masked[k] = False
        if toMask:
            for k in toMask:
                self.masked[k] = True