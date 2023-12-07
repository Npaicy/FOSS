import ray
import math
@ray.remote
class BestPlanManager:
    def __init__(self):
        self.candidatebest = {}
        self.AAM_esbest_feature = {}
        self.RL_estbest = {}
        self.queryImportance = {}
        self.latencyBuffer = {}
        self.startEval = True
    def update_AAM_esbest(self,query_id,feature_dict, exectime):
        self.AAM_esbest_feature[query_id] = (feature_dict,exectime)
    def add_candidatebest(self,query_id,hint,steps):
        if query_id not in self.candidatebest:
            self.candidatebest[query_id] = [(hint,steps)]
        else:
            if (hint,steps) not in self.candidatebest[query_id]:
                self.candidatebest[query_id].append((hint,steps))
    def clear_candidatebest(self):
        self.candidatebest.clear()
    def get_AAM_esbest(self,query_id = None):
        if query_id == None:
            return self.AAM_esbest_feature
        else:
            return self.AAM_esbest_feature[query_id]
    def get_AAM_best_totaltime(self):
        total = 0
        for k,v in self.AAM_esbest_feature.items():
            total += v[1]
        return total
    def get_candidatebest(self):
        return self.candidatebest
    def update_RL_esbest(self,query_id,featuredict,exectime):
         self.RL_estbest[query_id] = (featuredict, exectime)
    def get_RL_esbest(self,query_id = None):
        if query_id == None:
            return self.RL_estbest
        else:
            return self.RL_estbest[query_id]
    def update_weightsByRLesbest(self):
        # if self.startEval:
        for k,v in self.RL_estbest.items():
            self.queryImportance[k] = int(2 ** (math.floor(math.log10(v[1]))))
        return self.queryImportance
    def getqueryImportance(self):
        return self.queryImportance
    def updatequeryImportance(self,queryImportance, starteval = True):
        if not starteval:
            self.queryImportance = queryImportance
        else:
            self.queryImportance = {}
        self.startEval = starteval
    def update_latencyBuffer(self,latencybuffer):
        self.latencyBuffer = latencybuffer
    def get_latencyBuffer(self):
        return self.latencyBuffer
    