import os
import csv 
import math
import pandas as pd
# @ray.remote
class DataColletor:
    def __init__(self, genConfig):
        self.planJsonPool= []
        self.planVecPool = {}
        self.planVecCurr = {}
        self.testPool    = {}
        self.config      = genConfig
    def collect_testPool(self, queryID,feature, label,istimeout):
        if queryID not in self.testPool:
            self.testPool[queryID] = [[feature, label, istimeout]]
        else:
            self.testPool[queryID].append([feature, label, istimeout])

    def get_testData(self):
        testInputs = []
        testLabels = []
        testWeights = []
        for k,v in self.testPool.items():
            inputs,labels,weights = self.getpair_muti(v)
            testInputs.extend(inputs)
            testLabels.extend(labels)
            testWeights.extend(weights)
        return testInputs,testLabels, testWeights
    
    def clear_testPool(self):
        self.testPool.clear()
        
    def collect_planVecPool(self, queryID, feature, label, istimeout):
        if queryID not in self.planVecPool:
            self.planVecPool[queryID] = [[feature, label, istimeout]]
        else:
            self.planVecPool[queryID].append([feature, label, istimeout])

    def collect_planJsonPool(self, queryno, planjson, label, istimeout):
        self.planJsonPool.append([queryno, planjson, label, istimeout])

    def get_featuresPool(self):
        features = []
        for queryId in self.planVecPool:
            queryId_len = len(self.planVecPool[queryId])
            for k in range(queryId_len):
                features.append(self.planVecPool[queryId][k][0])
        features = pd.Series(features)
        return features
    
    def wirte_planJsonPool(self, path):
        if not os.path.exists(path):
            with open(path,mode='w',newline='',encoding='utf8') as cf:
                wf=csv.writer(cf)
                wf.writerow(['queryno','planjson', 'latency','istimeout'])
                for i in self.planJsonPool:
                    wf.writerow(i)
        else:
            with open(path,mode='a',newline='',encoding='utf8') as cfa:
                wf = csv.writer(cfa)
                for i in self.planJsonPool:
                    wf.writerow(i)
        self.planJsonPool.clear()

    def get_samples(self):
        Inputs = []
        Labels = []
        Weights = []
        for k,v in self.planVecPool.items():
            inputs, labels, weights = self.getpair_muti_full(v)
            Inputs.extend(inputs)
            Labels.extend(labels)
            Weights.extend(weights)
        return Inputs, Labels, Weights
        
    def getpair_muti_full(self, old_fea_latency):
        old_length = len(old_fea_latency)
        inputs = []
        labels = []
        weights = []
        if old_length == 1:
            return inputs,labels,weights
        
        to_save = []
        to_delete= []
        for i in range(old_length):
            if old_fea_latency[i][2]:   
                to_delete.append(i)
            else:
                to_save.append(i)
        new_fea_latency = []

        for i in to_save:
            for j in to_delete:
                plan_pair = {'left':old_fea_latency[j][0],'right':old_fea_latency[i][0]}
                latency_pair = [old_fea_latency[j][1], old_fea_latency[i][1]]
                label = 1
                for l, p in enumerate(self.config.splitpoint):
                    if (latency_pair[0] - latency_pair[1]) / latency_pair[0] >= p:
                        label = self.config.classNum - l
                        break
                labels.append(label)
                weights.append(math.log10(1 + (max(latency_pair) - min(latency_pair))))
                inputs.append(plan_pair)
            new_fea_latency.append(old_fea_latency[i])

        new_length = len(new_fea_latency)
        for i in range(new_length):
            for j in range(new_length):
                if i != j:
                    plan_pair = {'left':new_fea_latency[i][0],'right':new_fea_latency[j][0]}
                    latency_pair = [new_fea_latency[i][1], new_fea_latency[j][1]]
                    ratio = (latency_pair[0] - latency_pair[1]) / latency_pair[0]
                    label = 0
                    for l, p in enumerate(self.config.splitpoint):
                        if ratio >= p:
                            label = len(self.config.splitpoint) - l
                            break
                    labels.append(label)
                    inputs.append(plan_pair)
                    weights.append(math.log10(1 + (max(latency_pair) - min(latency_pair))))
        return inputs,labels,weights
    
    def getpair_muti(self,old_fea_latency):
        old_length = len(old_fea_latency)
        inputs = []
        labels = []
        weights = []
        if old_length == 1:
            return inputs,labels
        
        to_save = []
        to_delete= []
        for i in range(old_length):
            if old_fea_latency[i][2]:   # 标记超时的plan
                to_delete.append(i)
            else:
                to_save.append(i)
        new_fea_latency = []

        for i in to_save:
            for j in to_delete:
                plan_pair = {'left':old_fea_latency[j][0],'right':old_fea_latency[i][0]}
                latency_pair = [old_fea_latency[j][1], old_fea_latency[i][1]]
                label = 0
                for l, p in enumerate(self.config.splitpoint):
                    if (latency_pair[0] - latency_pair[1]) / latency_pair[0] >= p:
                        label = self.config.classNum - l
                        break
                if label !=0:
                    labels.append(label)
                    weights.append(math.log10(1 + (max(latency_pair) - min(latency_pair))))
                    inputs.append(plan_pair)
            new_fea_latency.append(old_fea_latency[i])

        new_length = len(new_fea_latency)
        for i in range(new_length - 1):
            for j in range(i + 1, new_length):
                plan_pair = {'left':new_fea_latency[i][0],'right':new_fea_latency[j][0]}
                latency_pair = [new_fea_latency[i][1], new_fea_latency[j][1]]
                ratio = (latency_pair[0] - latency_pair[1]) / latency_pair[0]
                label = 0
                for l, p in enumerate(self.config.splitpoint):
                    if ratio >= p:
                        label = len(self.config.splitpoint) - l
                        break
                labels.append(label)
                inputs.append(plan_pair)
                weights.append(math.log10(1+(max(latency_pair) - min(latency_pair))))
        return inputs,labels,weights
    