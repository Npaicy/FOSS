import ray
from datacollector import DataColletor
from pairestimator import PairTrainer,MyDataset
import pandas as pd
import json
from copy import deepcopy
@ray.remote
class RemoteAAM():
    def __init__(self, config):
        self.config      = config
        self.pairtrainer = PairTrainer(config)
        
    def TrainModel(self, dataset):
        self.pairtrainer.retrainmodel()
        self.pairtrainer.fit(dataset,mybatch_size = 128, epochs = 15)
        self.pairtrainer.save_model(self.config.model_path)
        
    def GetPrediction(self,hint_feature):
        return self.pairtrainer.predict_epi(hint_feature)
    
    def GetListPrediction(self,inputs):
        return self.pairtrainer.predict_list(inputs)
    
    def LoadModel(self):
        self.pairtrainer.load_model(self.config.model_path)

    def SaveModel(self):
        self.pairtrainer.save_model(self.config.model_path)
        
@ray.remote
class Learner():
    def __init__(self, bpm, planhelper, predictor, genConfig):
        self.bpm           = bpm
        self.planhelper    = planhelper
        self.predictor     = predictor
        self.config        = genConfig
        self.baseline      = None
        self.datacollector = DataColletor(self.config)
        self.remoteAAM     = RemoteAAM.remote(genConfig)
        self.trainref      = None
        self.IsReady       = []
        self.CL            = 0
        self.trainTimes    = 0
        self.updateTimes   = 0
        self.candidatebest_counts = 0
        self.accumulatedExecuted  = 0
        self.StepsRecordFile = open(self.config.beststeps_record,'a') # TO_Delete

    def Runing(self):
        print('Start Learner!')
        startSignal = True
        balances = ray.get(self.bpm.get_balances.remote())
        self.ExecutePlans(balances)
        candidatebest = None
        while True:
            canididateLength = ray.get(self.bpm.get_currNo.remote())
            if self.CL != canididateLength or self.updateTimes != self.trainTimes:
                self.predictor.LoadModel.remote()
                self.updateTimes = self.trainTimes
                idxlist, inputs, self.CL = ray.get(self.bpm.get_candidatePair.remote())
                if idxlist:
                    if self.trainTimes > 1:
                        to_validate = []
                        prediction = ray.get(self.predictor.GetListPrediction.remote(inputs))
                        for i, pred in enumerate(prediction):
                            if pred != 0:
                                to_validate.append(idxlist[i])
                    else:
                        to_validate = idxlist
                    ray.get(self.bpm.update_validationSet.remote(to_validate))
                candidatebest = ray.get(self.bpm.get_candidatebest.remote())
            self.ExecutePlans(candidatebest)
            candidatebest = None
            if not startSignal:
                self.IsReady,_ = ray.wait([self.trainref], timeout = 0.01)
            if self.candidatebest_counts >= 25:
                if startSignal or self.trainref in self.IsReady:
                    inputs,labels,weights = self.datacollector.get_samples()
                    dataset = MyDataset(inputs,labels,weights)
                    dataset_len = dataset.__len__()
                    print('train data length:',dataset_len)
                    startSignal = False
                    self.trainref = self.remoteAAM.TrainModel.remote(dataset)
                    self.candidatebest_counts = 0       
                    self.IsReady = [] 
                    self.trainTimes += 1
            endSignal = ray.get(self.bpm.get_schedule.remote())
            if endSignal:
                # balances = ray.get(self.bpm.get_balances.remote())
                # self.ExecutePlans(balances)
                print('Stop Learner!')
                return self.accumulatedExecuted
            
    def TrainModel(self):  # Block
        inputs,labels,weights = self.datacollector.get_samples()
        dataset = MyDataset(inputs,labels,weights)
        dataset_len = dataset.__len__()
        print('train data length:',dataset_len)
        ray.get(self.remoteAAM.TrainModel.remote(dataset))

    def CollectSample(self, query_id, optimal_feature,latency,timeout):
        self.datacollector.collect_planVecPool(query_id,optimal_feature,latency,timeout)
        return True
    
    def GetPrediction(self,hint_feature):
        return ray.get(self.remoteAAM.GetPrediction.remote(hint_feature))
    
    def ExecutePlans(self,candidates):
        if isinstance(candidates, pd.DataFrame):
            for idx, samples in candidates.iterrows():
                # if hints:
                sql = samples['sql']
                hint = samples['hint']
                query_id = samples['queryid']
                feature_dict = samples['feature']
                _,bestexec = ray.get(self.bpm.get_AAM_esbest.remote(query_id))
                latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote(hint, sql, query_id,timeout = self.config.timeoutcoeff * self.baseline[query_id]))# self.config.timeoutcoeff * baseline[0]))
                if iscollect:
                    self.candidatebest_counts += 1
                    self.accumulatedExecuted += 1
                    # feature_dict,_,_,cost_plan_json = ray.get(self.planhelper.GetFeature.remote(hint,sql,False,query_id))
                    # feature_dict['steps'] = np.array([steps])
                    self.datacollector.collect_planVecPool(query_id,deepcopy(feature_dict),latency_timeout[0],latency_timeout[1])
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    steps = int(feature_dict['steps'][0] * self.config.maxsteps)
                    print('Query id:{}, CurrBest:{:.3f}, EstiBest:{:.3f}, Advbybest:{:.3f}%, Timeout:{} Steps:{}'.format(query_id,bestexec,latency_timeout[0], Advbybest * 100, latency_timeout[1],steps))
                    self.StepsRecordFile.write(json.dumps([query_id, '|'.join([str(steps),str(round(latency_timeout[0],3))])]) + '\n') # TO_Delete
                    self.StepsRecordFile.flush()
                    if Advbybest >= self.config.splitpoint[-1]:
                        ray.get(self.bpm.update_AAM_esbest.remote(query_id, deepcopy(feature_dict),latency_timeout[0]))
                        bestexec = latency_timeout[0]
    def getBaseline(self,baseline):
        self.baseline = baseline