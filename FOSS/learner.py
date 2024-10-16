import ray
from datacollector import DataColletor
from pairestimator import PairTrainer,MyDataset
import pandas as pd
import torch,os
from copy import deepcopy
@ray.remote(num_gpus = 0.3, num_cpus = 5)
class RemoteEvaluator():
    def __init__(self, config, device = None):
        self.config      = config
        self.pairtrainer = PairTrainer(config, device = device)
        if os.path.exists(self.config.evaluator_path):
            self.pairtrainer.load_model(self.config.evaluator_path)
    def TrainModel(self, dataset):
        self.pairtrainer.retrainmodel()
        self.pairtrainer.fit(dataset, mybatch_size = 128, epochs = 15)
        self.pairtrainer.save_model(self.config.evaluator_path)
        
    def GetPrediction(self,hint_feature):
        return self.pairtrainer.predict_epi_parallel(hint_feature)
    
    def GetListPrediction(self,inputs):
        return self.pairtrainer.predict_list(inputs)
    
    def LoadModel(self, evaluator_path = None):
        if evaluator_path:
            self.pairtrainer.load_model(evaluator_path)
        else:
            self.pairtrainer.load_model(self.config.evaluator_path)

    def SaveModel(self):
        self.pairtrainer.save_model(self.config.evaluator_path)
    
    def get_embed(self, plan_feature):
        batch_size = 16
        keys = plan_feature.iloc[0].keys()
        all_embeddings = []

        for i in range(0, len(plan_feature), batch_size):
            batch_features = {k: torch.cat([torch.tensor(features[k]).unsqueeze(0).to(self.pairtrainer.device)
                                            for features in plan_feature.iloc[i:i+batch_size].values], dim=0) for k in keys}
            embeddings = self.pairtrainer.get_embed(batch_features)
            embeddings = embeddings.cpu().detach()
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).numpy()
        
@ray.remote(num_gpus = 0.3, num_cpus = 5)
class Learner():
    def __init__(self, bpm, planhelper, predictor, genConfig):
        self.bpm           = bpm
        self.planhelper    = planhelper
        self.predictor     = predictor
        self.config        = genConfig
        self.baseline      = None
        self.maxsampleNum  = self.config.maxsamples
        self.uncertainty_sample  = 0
        self.hybrid_global_sample = 0 
        self.hybrid_sample        = 0
        if  self.config.sample_strategy  == 'uncertainty':
            self.uncertainty_sample  = self.config.maxsamples 
        elif self.config.sample_strategy == 'hybrid_global':
            self.hybrid_global_sample = self.config.maxsamples
        elif self.config.sample_strategy == 'hybrid':
            self.hybrid_sample        = self.config.maxsamples
        self.random_samples= self.maxsampleNum - self.uncertainty_sample - self.hybrid_sample - self.hybrid_global_sample
        self.datacollector = DataColletor(self.config)
        self.remote_evaluator     = RemoteEvaluator.remote(genConfig)
        if self.config.update_evaluator:
            self.remote_evaluator.SaveModel.remote()
        self.trainref      = None
        self.IsReady       = []
        self.globalNo      = 0
        self.iterNo        = 0
        self.trainTimes    = 0
        self.updateTimes   = 0
        self.toExecuted_counts = 0
        self.accumulatedExecuted  = 0
        # self.StepsRecordFile = open(self.config.beststeps_record,'a') # TO_Delete

    def Runing(self):
        print('Start Learner!')
        # startSignal = True
        balances = ray.get(self.bpm.get_balances.remote())
        self.ExecutePlans(balances)
        toExecuted = []
        while True:
            globalNo, iterNo = ray.get(self.bpm.get_stateNo.remote())
            # if (self.globalNo - globalNo) >= 15 or self.trainTimes != self.updateTimes:
            if self.trainTimes != self.updateTimes:
                self.predictor.LoadModel.remote()
                self.updateTimes = self.trainTimes
                # tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('heuristic', predictor = self.predictor, sampleNum = self.heuristic_samples))
                # self.globalNo = globalNo
                # if tmpExecuted is not None:
                #     toExecuted.append(tmpExecuted)
                # globalNo, iterNo = ray.get(self.bpm.get_stateNo.remote())
            if iterNo != self.iterNo: # process the rest of the samples
                ray.get(self.bpm.write_iterCandidate.remote())
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('hybrid', sampleNum = self.hybrid_sample, predictor = self.predictor))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('uncertainty', sampleNum = self.uncertainty_sample))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('random', sampleNum = self.random_samples))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                if self.hybrid_global_sample != 0:
                    curpool = self.datacollector.get_featuresPool()
                    tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('hybrid_global', sampleNum = self.hybrid_global_sample, 
                                                                                  predictor = self.predictor, currPool = curpool))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                ray.get(self.bpm.clear_iterCandidate.remote())
                self.iterNo = iterNo
            if len(toExecuted) > 0:
                toExecuted = pd.concat(toExecuted, ignore_index=True)  
                self.ExecutePlans(toExecuted)
                toExecuted = []
            if self.trainref != None:
                self.IsReady,_ = ray.wait([self.trainref], timeout = 0.01)
            if self.toExecuted_counts >= 25:
                if self.trainref == None or self.trainref in self.IsReady:
                    inputs,labels,weights = self.datacollector.get_samples()
                    dataset = MyDataset(inputs,labels,weights)
                    dataset_len = dataset.__len__()
                    print('train data length:',dataset_len)
                    # startSignal = False
                    self.trainref = self.remote_evaluator.TrainModel.remote(dataset)
                    self.toExecuted_counts = 0       
                    self.IsReady = [] 
                    self.trainTimes += 1
            endSignal = ray.get(self.bpm.get_schedule.remote())
            if endSignal:
                print('Stop Learner!')
                return self.accumulatedExecuted
            
    def TrainModel(self):  # Block
        inputs,labels,weights = self.datacollector.get_samples()
        dataset = MyDataset(inputs,labels,weights)
        dataset_len = dataset.__len__()
        print('train data length:',dataset_len)
        ray.get(self.remote_evaluator.TrainModel.remote(dataset))

    def CollectSample(self, query_id, optimal_feature,latency,timeout):
        self.datacollector.collect_planVecPool(query_id,optimal_feature,latency,timeout)
        return True
    
    def GetPrediction(self,hint_feature):
        return ray.get(self.remote_evaluator.GetPrediction.remote(hint_feature))
    
    def ExecutePlans(self,candidates):
        if isinstance(candidates, pd.DataFrame):
            for idx, samples in candidates.iterrows():
                # if hints:
                sql = samples['sql']
                hint = samples['hint']
                query_id = samples['queryid']
                feature_dict = samples['feature']
                _,bestexec = ray.get(self.bpm.get_evaluator_esbest.remote(query_id))
                latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote(hint, sql, query_id,timeout = self.config.timeoutcoeff * self.baseline[query_id]))# self.config.timeoutcoeff * baseline[0]))
                if iscollect:
                    self.toExecuted_counts += 1
                    self.accumulatedExecuted += 1
                    # feature_dict,_,_,cost_plan_json = ray.get(self.planhelper.GetFeature.remote(hint,sql,False,query_id))
                    # feature_dict['steps'] = np.array([steps])
                    self.datacollector.collect_planVecPool(query_id,deepcopy(feature_dict),latency_timeout[0],latency_timeout[1])
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    steps = int(feature_dict['steps'][0] * self.config.maxsteps)
                    print('Query id:{}, CurrBest:{:.3f}, EstiBest:{:.3f}, Advbybest:{:.3f}%, Timeout:{} Steps:{}'.format(query_id,bestexec,latency_timeout[0], Advbybest * 100, latency_timeout[1],steps))
                    # self.StepsRecordFile.write(json.dumps([query_id, '|'.join([str(steps),str(round(latency_timeout[0],3))])]) + '\n') # TO_Delete
                    # self.StepsRecordFile.flush()
                    if Advbybest >= self.config.splitpoint[-1]:
                        ray.get(self.bpm.update_evaluator_esbest.remote(query_id, deepcopy(feature_dict),latency_timeout[0]))
                        bestexec = latency_timeout[0]

    def updateBaseline(self,baseline):
        self.baseline = baseline