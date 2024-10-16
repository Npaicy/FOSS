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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from util import get_median
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
        self.evaluator_esbest_feature = {}
        self.RL_esbest_feature  = {}
        self.median_feature     = {}
        self.buffer             = []
        for i in range(len(self.train_dir)):
            with open(self.config.train_workload_path + self.train_dir[i],'r') as f:
                sql = f.read()
            query_id = self.train_dir[i].split('.')[0]
            self.buffer.append(query_id)
            
            self.validateSet.loc[query_id] = [sql]
            if isremote:
                feature_dict,hintdict,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,True,query_id = query_id))
            else:
                feature_dict,hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',sql,True,query_id = query_id)
            feature_dict['steps']   = np.array([0])
            # self.buffer.extend([query_id] * len(hintdict['join order']))
            cost_plan_json['steps'] = 0
            self.trainSet.loc[query_id] = [len(hintdict['join order']), sql, (feature_dict, hintdict, cost_plan_json)]
            if not left_deep:
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
        
    def get2train(self):
        query_id = random.choices(self.buffer)[0]
        return self.trainSet.loc[query_id,'sql'],self.trainSet.loc[query_id,'base_train_feature'],\
                query_id, self.evaluator_esbest_feature[query_id], self.median_feature[query_id]
    
    def updateBuffer(self,queryImportance):
        self.buffer = []
        for k, v in queryImportance.items():
            self.buffer.extend([k] * v)
    
    def update_evaluator_esbest(self,evaluator_esbest_feature):
        self.evaluator_esbest_feature = deepcopy(evaluator_esbest_feature)

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
        self.test_expPool    = open(self.config.TestExperiencePool,'w')
        # self.bestplan   = open(self.config.beststeps_record,'a') # TO_Delete
        self.testQuery  = []
        self.trainQuery = []
        self.writer     = writer
        self.ExecutionTime  = {}
        self.PlanningTime   = {}
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
        self.test_expPool.write(json.dumps([queryId,'|'.join([hint,str(agentNo),str(steps)])])+"\n")
        self.test_expPool.flush()

    def recordTime(self,name):
        _time = str(round(time.time() - self.timeRecord, 3) * 1000)
        self.timeRecord = time.time()
        self.resultFile.write(json.dumps([name,_time])+"\n")
        self.resultFile.flush()

    def recordeval(self,queryId, execution_time, planning_time):
        execution_time = round(execution_time, 3)
        planning_time = round(planning_time, 3)
        if queryId not in self.ExecutionTime:
            self.ExecutionTime[queryId] = [execution_time]
            self.PlanningTime[queryId]  = [planning_time]
        else:
            self.ExecutionTime[queryId].append(execution_time)
            self.PlanningTime[queryId].append(planning_time)

    def recordwrl(self, valIter):
        foss_test   = 0
        foss_train  = 0
        pg_test  = 0
        pg_train = 0
        for k,v in self.ExecutionTime.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in self.testQuery:
                    foss_test = foss_test + v[-1] + self.PlanningTime[k][-1]
                    pg_test   = pg_test + v[0] + self.PlanningTime[k][0]
                elif k in self.trainQuery:
                    foss_train = foss_train + v[-1] + self.PlanningTime[k][-1]
                    pg_train   = pg_train + v[0] + self.PlanningTime[k][0]
        if pg_test != 0:
            self.writer.add_scalar('Test/Workload',foss_test,valIter)
            wrl_test = foss_test / pg_test
            self.recordeval("Test_WRL", wrl_test, 0)
            self.writer.add_scalar('Test/WRL',wrl_test,valIter)
        if pg_train != 0:
            self.writer.add_scalar('Train/Workload',foss_train,valIter)
            wrl_train = foss_train / pg_train
            self.recordeval("Train_WRL", wrl_train, 0)
            self.writer.add_scalar('Train/WRL',wrl_train,valIter)

    def recordgmrl(self, valIter):
        gmrl_train   = 1
        gmrl_test    = 1
        counts_train = 0
        counts_test  = 0
        for k,v in self.ExecutionTime.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in self.testQuery:
                    gmrl_test = gmrl_test * (v[-1] / v[0])
                    counts_test += 1
                elif k in self.trainQuery:
                    gmrl_train = gmrl_train * (v[-1] / v[0])
                    counts_train += 1
        if counts_test != 0:
            gmrl_test = pow(gmrl_test, 1 / counts_test)
            self.recordeval("Test_GMRL", gmrl_test, 0)
            self.writer.add_scalar('Test/GMRL',gmrl_test,valIter)
        if counts_train != 0:
            gmrl_train = pow(gmrl_train, 1 / counts_train)
            self.recordeval("Train_GMRL", gmrl_train, 0)
            self.writer.add_scalar('Train/GMRL',gmrl_train,valIter)

    def recordMetric(self,valIter):
        self.recordwrl(valIter)
        self.recordgmrl(valIter)
    # def recordBestPlanSteps(self,query_id,steps,latency):
    #     self.bestplan.write(json.dumps([query_id, '|'.join([str(steps),str(round(latency,3))])]) + '\n') # TO_Delete
    #     self.bestplan.flush()
    def writeout(self):
        with open(self.config.eval_output_path, "w") as out:
            for k,v in self.ExecutionTime.items():
                out.write(json.dumps([k, v]) + '\n')
                out.flush()
    def close(self):
        self.resultFile.close()
        self.test_expPool.close()

@ray.remote
class BestPlanManager:
    def __init__(self, genConfig):
        self.config = genConfig
        self.globalCandidate = pd.DataFrame(columns = ['queryid', 'hint','feature','sql'])
        self.iterCandidate = pd.DataFrame(columns = ['queryid', 'hint','feature','sql','prob'])
        self.balances = pd.DataFrame(columns = ['queryid', 'hint','feature','sql'])
        self.sampleNum = self.config.maxsamples
        self.iterNo      = 0
        self.globalNo    = 0
        self.masked    = {}
        self.evaluator_esbest_feature = {}
        self.RL_estbest = {}
        self.medianplan = {}
        self.queryImportance = {}
        self.latencyBuffer = {}
        self.endSignal = False
        # self.validationSet = None
        self.coeff_1 = 0.4
        self.coeff_2 = 0.6
        self.iterCandidate.to_csv(self.config.ExperiencePool, mode='a', index=False, columns = ['queryid','hint','prob'])

    def update_evaluator_esbest(self,query_id,feature_dict, exectime):
        self.evaluator_esbest_feature[query_id] = (feature_dict,exectime)

    def add_globalCandidate(self,query_id,hint,feature_dict,sql):  # No drop
        self.globalCandidate.loc[self.globalNo] = \
        {'queryid':query_id,'hint':hint,'feature':feature_dict,'sql':sql}
        self.globalNo += 1
    
    def clear_globalCandidate(self):
        self.globalCandidate.drop(self.globalCandidate.index, inplace = True)

    def add_iterCandidate(self,query_id,hint,feature_dict,sql,prob):  #Drop every iter or every evaluator upadtes
        if not ((self.iterCandidate['queryid'] == query_id) & (self.iterCandidate['hint'] == hint)).any():
            self.iterCandidate.loc[self.iterNo] = \
            {'queryid':query_id,'hint':hint,'feature':feature_dict,'sql':sql,'prob':prob}
            self.iterNo += 1
            
    def clear_iterCandidate(self):
        self.iterCandidate.drop(self.iterCandidate.index, inplace = True)

    def write_iterCandidate(self):
        self.iterCandidate.to_csv(self.config.ExperiencePool, mode='a', index=False, columns = ['queryid','hint','prob'], header = 0)

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
        
    def get_stateNo(self):
        return self.globalNo, self.iterNo
    
    # def update_validationSet(self,validationSet):
    #     self.validationSet = validationSet

    # def get_candidatePair(self):
    #     # self.candidatebest.drop_duplicates(subset=['queryid', 'hint', 'steps'],keep='first').reset_index(drop=True)
    #     inputs = []
    #     idxlist = []
    #     for idx, samples in self.candidateBest.iterrows():
    #         inputs.append({'left': self.evaluator_esbest_feature[samples['queryid']][0],'right':samples['feature']})
    #         idxlist.append(idx)
    #     return idxlist, inputs, self.currNo
    
    def get_evaluator_esbest(self,query_id = None):
        if query_id == None:
            return self.evaluator_esbest_feature
        else:
            return self.evaluator_esbest_feature[query_id]
        
    def get_evaluator_best(self):
        total = 0
        best_steps = {}
        for k,v in self.evaluator_esbest_feature.items():
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
        for k in self.evaluator_esbest_feature:
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

    
    def update_weightsByRLesbest(self):
        # decay = 0.05
        # self.coeff_1 = min(1.0, self.coeff_1 + decay)
        # self.coeff_2 = max(0.0, self.coeff_2 - decay)
        for k,v in self.RL_estbest.items():
            weights_1 = max(1, (v[1] - self.evaluator_esbest_feature[k][1]) / 10)
            weights_2 = max(1, v[1]/ 10)
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
    def random_sample(self, sampleNum = 0, frac = 0.1):
        if sampleNum == 0:
            return None, 0
        if len(self.iterCandidate) < sampleNum:
            sampleNum = len(self.iterCandidate)
        if sampleNum != 0:
            samples = self.iterCandidate.sample(sampleNum)
            self.iterCandidate.drop(samples.index, inplace=True)
            print(f'Random Sample:{len(samples)}')
            return samples, len(samples)
        else:
            samples = self.iterCandidate.sample(frac = frac)
            self.iterCandidate.drop(samples.index, inplace=True)
            print(f'Random Sample:{len(samples)}')
            return samples, len(samples)
    def uncertainty_sample(self, sampleNum = 0, threshold = 0.85):
        # choose the max prob
        if sampleNum == 0:
            return None, 0
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        self.iterCandidate_filter = self.iterCandidate[self.iterCandidate['prob'] < threshold]
        if len(self.iterCandidate_filter) == 0:
            return None, 0
        samples = self.iterCandidate_filter.nsmallest(sampleNum, 'prob')
        self.iterCandidate.drop(samples.index, inplace = True)
        print('Uncertainty Samples: ', len(samples))
        return samples, len(samples)
    
    def hybrid_sample(self, predictor, sampleNum, threshold = 0.9):
        if sampleNum == 0:
            return None, 0
        if len(self.iterCandidate) == 0:
            return None, 0
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        self.iterCandidate_filter = self.iterCandidate[self.iterCandidate['prob'] < threshold]
        if len(self.iterCandidate_filter) == 0:
            return None, 0
        uncertainty_samples = self.iterCandidate_filter.nsmallest(sampleNum * 10, 'prob')
        embeddings = ray.get(predictor.get_embed.remote(uncertainty_samples['feature']))
        if len(uncertainty_samples) > sampleNum:
            cosine_sim_matrix = cosine_similarity(embeddings)
            similarity_scores = np.sum(cosine_sim_matrix, axis=1) - 1
            q_idxs = np.argsort(similarity_scores)[:sampleNum]
            samples = uncertainty_samples.iloc[q_idxs]
        else:
            samples = uncertainty_samples
        self.iterCandidate.drop(samples.index, inplace = True)
        print(f'Hybrid Sample:{len(samples)}')
        return samples, len(samples)
    
    def hybrid_sample_global(self, predictor, sampleNum, currPool):
        if sampleNum == 0 or len(self.iterCandidate) == 0:
            return None, 0
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        uncertainty_samples = self.iterCandidate.nsmallest(sampleNum * 10, 'prob')
        undetermined = ray.get(predictor.get_embed.remote(uncertainty_samples['feature']))
        curr_embeddings = ray.get(predictor.get_embed.remote(currPool))
        if len(uncertainty_samples) > sampleNum:
            farthest_points_indices = []
            cosine_sim_matrix = cosine_similarity(undetermined, curr_embeddings)
            sum_similarities = np.sum(cosine_sim_matrix, axis=1)
            for _ in range(sampleNum):
                farthest_point_idx = np.argmin(sum_similarities)
                farthest_points_indices.append(farthest_point_idx)
                # curr_embeddings = np.vstack([curr_embeddings, undetermined[farthest_point_idx]])
                new_similarities = cosine_similarity(undetermined, undetermined[farthest_point_idx].reshape(1, -1)).flatten()
                sum_similarities += new_similarities
                sum_similarities[farthest_point_idx] += len(curr_embeddings) # avoid chosen in the next iter
            samples = uncertainty_samples.iloc[farthest_points_indices]
        else:
            samples = uncertainty_samples
        self.iterCandidate.drop(samples.index, inplace=True)
        print(f'Hybrid Sample Global: {len(samples)}')
        return samples, len(samples)
    
    def heuristic_sample(self, predictor, sampleNum):
        if sampleNum == 0:
            return None, 0
        inputs = []
        idxlist = []
        for idx, samples in self.globalCandidate.iterrows():
            inputs.append({'left': self.evaluator_esbest_feature[samples['queryid']][0],'right':samples['feature']})
            idxlist.append(idx)
        to_validate = []
        prediction = ray.get(predictor.GetListPrediction.remote(inputs))
        for i, pred in enumerate(prediction):
            if pred != 0:
                to_validate.append(idxlist[i])
        if len(to_validate) >= sampleNum:
            sample_idx = random.sample(to_validate, sampleNum)
            samples = self.globalCandidate.loc[sample_idx]
            self.globalCandidate.drop(sample_idx, inplace=True)
        else:
            random.shuffle(to_validate)
            samples = self.globalCandidate.loc[to_validate]
            self.globalCandidate.drop(to_validate, inplace=True)
        print(f'Heuristic Sample:{len(samples)}')
        return samples, len(samples)
    
    def get_toExecuted(self, strategy, predictor = None, sampleNum = None, currPool = None):
        start = time.time()
        if strategy == 'random':
            samples, num_samples = self.random_sample(sampleNum = sampleNum)
        elif strategy == 'uncertainty':
            samples, num_samples = self.uncertainty_sample(sampleNum)
        elif strategy == 'hybrid_global':
            samples, num_samples = self.hybrid_sample_global(predictor, sampleNum, currPool)
        elif strategy == 'heuristic':
            samples, num_samples = self.heuristic_sample(predictor, sampleNum)
        elif strategy == 'hybrid':
            samples, num_samples = self.hybrid_sample(predictor, sampleNum)
        sample_time = time.time() - start
        # print(f'{strategy} sample time:{sample_time}')
        return samples, num_samples