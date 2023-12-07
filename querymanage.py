import random
import json
from config import Config
import os
from planhelper import PlanHelper
import numpy as np
from copy import deepcopy
import pandas as pd
config = Config()
# @ray.remote
class QueryManager:
    def __init__(self,writer = None) -> None:
        self.train_dir = os.listdir(config.train_workload_path)
        self.test_dir = os.listdir(config.test_workload_path)
        self.writer = writer
        self.planhelper = PlanHelper()
        self.trainSet = pd.DataFrame(columns=['tablenum','sql','query_id','base_train_feature'])
        self.validateSet = pd.DataFrame(columns=['sql','query_id'])
        self.testSet = pd.DataFrame(columns=['sql','query_id'])
        # self.train_query = []
        # self.test_query = []
        # self.filter_train_dir = []
        # self.base_train_feature = []
        self.AAM_esbest_feature = {}
        self.RL_esbest_feature = {}
        
        # self.candidatebest = {}
        # self.train_dir = sorted(self.train_dir)
        # random.seed(config.seed)
        # random.shuffle(self.train_dir)
        # self.train_queryid = []
        # self.test_queryid = []
        self.queryid2sql = {}
        self.queryid2No = {}
        # self.validate_query = {}
        # self.validate_query_id = {}self.queryid2No[query_id] = counts
        for i in range(len(self.train_dir)):
            with open(config.train_workload_path + self.train_dir[i],'r') as f:
                sql = f.read()
                query_id = self.train_dir[i].split('.')[0]
                feature_dict,hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',sql,True,query_id=query_id)
                self.validateSet.loc[len(self.validateSet.index)] = [sql, query_id]
                if left_deep:
                    feature_dict['steps'] = np.array([0])
                    cost_plan_json['steps'] = 0
                    self.trainSet.loc[len(self.trainSet.index)] = [len(hintdict['join order']), sql, query_id,(feature_dict,hintdict,cost_plan_json)]
                    # self.train_query.append(sql)
                    # self.train_queryid.append(query_id)
                    self.queryid2sql[query_id] = sql
                    # self.filter_train_dir.append(self.train_dir[i])
                    # self.base_train_feature.append((feature_dict,hintdict,cost_plan_json))
        for i in range(len(self.test_dir)):
            with open(config.test_workload_path + self.test_dir[i],'r') as f:
                sql = f.read()
                query_id = self.test_dir[i].split('.')[0]
                # self.test_queryid.append(query_id)
                # self.test_query.append(sql)
                self.queryid2sql[query_id] = sql
                self.testSet.loc[len(self.testSet.index)] = [sql, query_id]
        # self.trainSet = self.trainSet.sort_values(by='tablenum').reset_index(drop=True)
        self.queryid2No = pd.Series(index=self.trainSet['query_id'], data = self.trainSet.index).to_dict()
        # self.numOfTrain = len(self.train_query)
        self.cur_train = 0
        self.numOfTrain = len(self.trainSet.index)
        
        # self.numOfTest = len(self.test_query)
        self.cur_test = 0
        self.numOfTest = len(self.testSet.index)
        
        self.cur_validate = 0
        self.numOfvalidate = len(self.validateSet.index)
        # self.numOfvalidate = len(self.validate_query)
        
        self.output = {}
        self.buffer = list(range(self.numOfTrain))
        #self.planhelper.encoding.save_to_file(config.encoding_path)
    def get2explore(self):
        No = self.cur_train % self.numOfTrain
        # sql = self.train_query[No]
        self.cur_train += 1
        if No == self.numOfTrain - 1:
            oneloop = True
        else:
            oneloop = False
        return self.trainSet.loc[No,'sql'], No, self.trainSet.loc[No,'query_id'],self.trainSet.loc[No,'base_train_feature'],oneloop
    def get2exploreByqueryId(self,query_id):
        No = self.queryid2No[query_id]
        sql = self.queryid2sql[query_id]
        return sql, No, self.trainSet.loc[No,'query_id'],self.trainSet.loc[No,'base_train_feature']
    def get2eval(self):
        No = self.cur_test % self.numOfTest
        # sql = self.test_query[No]
        self.cur_test += 1
        if No == self.numOfTest - 1:
            return self.testSet.loc[No,'sql'], No, self.testSet.loc[No,'query_id'],True
        else:
            return self.testSet.loc[No,'sql'], No, self.testSet.loc[No,'query_id'],False
    def get2validate(self):
        No = self.cur_validate % self.numOfvalidate
        # sql = self.validate_query[No]
        self.cur_validate += 1
        if No == self.numOfvalidate - 1:
            return self.validateSet.loc[No,'sql'], No, self.validateSet.loc[No,'query_id'],True
        else:
            return self.validateSet.loc[No,'sql'], No, self.validateSet.loc[No,'query_id'],False
    def get2train(self):
        rand_select = random.choices(self.buffer)[0] % self.numOfTrain
        query_id = self.trainSet.loc[rand_select, 'query_id']
        return self.trainSet.loc[rand_select,'sql'],self.trainSet.loc[rand_select,'base_train_feature'],\
                query_id,self.AAM_esbest_feature[query_id],\
                self.RL_esbest_feature[query_id]
    def updateBuffer(self,queryImportance):
        self.buffer = []
        for k,v in queryImportance.items():
            if k in self.queryid2No:
                self.buffer.extend([self.queryid2No[k]] * v)
        # print(self.buffer)
    def getsqlbyqueryid(self,query_id):
        return self.queryid2sql[query_id]
    def recordeval(self,queryId,latency):
        latency = round(latency,3)
        if queryId not in self.output:
            self.output[queryId] = [latency]
        else:
            self.output[queryId].append(latency)
    def recordwrl(self):
        FOSS_test = 0
        FOSS_train = 0
        total_test = 0
        total_train = 0
        test_query = self.testSet['query_id'].to_list()
        validate_query = self.validateSet['query_id'].to_list()
        for k,v in self.output.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in test_query:
                    FOSS_test += v[-1]
                    total_test += v[0]
                elif k in validate_query:
                    FOSS_train += v[-1]
                    total_train += v[0]
        if total_test != 0:
            wrl_test = FOSS_test / total_test
            self.recordeval("Test_WRL", wrl_test)
            self.writer.add_scalar('Test_WRL',wrl_test,len(self.output['Test_WRL']))
        if total_train!= 0:
            wrl_train = FOSS_train / total_train
            self.recordeval("Train_WRL", wrl_train)
            self.writer.add_scalar('Train_WRL',wrl_train,len(self.output['Train_WRL']))

    def recordgmrl(self):
        gmrl_train = 1
        gmrl_test = 1
        counts_train = 0
        counts_test = 0
        test_query = self.testSet['query_id'].to_list()
        validate_query = self.validateSet['query_id'].to_list()
        for k,v in self.output.items():
            if k != "Train_WRL" and k != "Train_GMRL" and k != "Test_WRL" and k != "Test_GMRL":
                if k in test_query:
                    gmrl_test = gmrl_test * (v[-1] / v[0])
                    counts_test += 1
                elif k in validate_query:
                    gmrl_train = gmrl_train * (v[-1] / v[0])
                    counts_train += 1
        if counts_test != 0:
            gmrl_test = pow(gmrl_test, 1/counts_test)
            self.recordeval("Test_GMRL", gmrl_test)
            self.writer.add_scalar('Test_GMRL',gmrl_test,len(self.output['Test_GMRL']))
        if counts_train != 0:
            gmrl_train = pow(gmrl_train, 1/counts_train)
            self.recordeval("Train_GMRL", gmrl_train)
            self.writer.add_scalar('Train_GMRL',gmrl_train,len(self.output['Train_GMRL']))
        
    def writeout(self):
        with open(config.eval_output_path,"w") as out:
            for k,v in self.output.items():
                out.write(json.dumps([k, v]) + '\n')
                out.flush()
    def update_AAM_esbest(self,AAM_esbest_feature):
        self.AAM_esbest_feature = deepcopy(AAM_esbest_feature)
    def update_RL_esbest(self,RL_esbest_feature):
        self.RL_esbest_feature = deepcopy(RL_esbest_feature)
    
    