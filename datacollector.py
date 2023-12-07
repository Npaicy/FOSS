import os
import csv 
from config import Config
import random
config = Config()
# @ray.remote
class DataColletor:
    def __init__(self):
        self.training_data_offline = []
        self.training_data_online = {}
        self.alpha = config.alpha
        self.cur_data_online = {}
        self.test_data = {}
    # def get_data_offline(self):
    #     return self.training_data_offline
    
    # def get_data_online(self):
    #     return self.training_data_online

    # def collect_data_online(self, query_id,feature, label,istimeout):
    #     if query_id not in self.cur_data_online:
    #         self.cur_data_online[query_id] = [[feature, label,istimeout]]
    #     else:
    #         self.cur_data_online[query_id].append([feature, label,istimeout])
    def collect_test_data(self, query_id,feature, label,istimeout):
        if query_id not in self.test_data:
            self.test_data[query_id] = [[feature, label,istimeout]]
        else:
            self.test_data[query_id].append([feature, label,istimeout])
    def get_test_data(self):
        test_inputs = []
        test_labels = []
        for k,v in self.test_data.items():
            inputs,labels = self.getpair_muti(v)
            test_inputs.extend(inputs)
            test_labels.extend(labels)
        return test_inputs,test_labels
    def clear_test_data(self):
        self.test_data.clear()
    def collect_data_online(self, query_id,feature, label,istimeout):
        if query_id not in self.training_data_online:
            self.training_data_online[query_id] = [[feature, label,istimeout]]
        else:
            self.training_data_online[query_id].append([feature, label,istimeout])
    # def collect_data_online(self, query_id,feature, label,istimeout):
    #     if query_id not in self.training_data_online:
    #         self.training_data_online[query_id] = [[feature, label,istimeout]]
    #     else:
    #         self.training_data_online[query_id].append([feature, label,istimeout])
    def collect_data_offline(self,queryno,planjson,label,istimeout):
        self.training_data_offline.append([queryno,planjson, label,istimeout])
        
    def wirte_offline(self,path):
        if not os.path.exists(path):
            with open(path,mode='w',newline='',encoding='utf8') as cf:
                wf=csv.writer(cf)
                wf.writerow(['queryno','planjson', 'latency','istimeout'])
                for i in self.training_data_offline:
                    wf.writerow(i)
        else:
            with open(path,mode='a',newline='',encoding='utf8') as cfa:
                wf = csv.writer(cfa)
                for i in self.training_data_offline:
                    wf.writerow(i)
        self.training_data_offline = []

    # def process_online(self):
    #     if len(self.training_data_online) < 2:
    #         return None
    #     if len(self.training_data_online) >= config.max_pooling:
    #         self.training_data_online = self.training_data_online[-config.max_pooling:-1]
    #     self.training_data_online.sort(key = lambda x : (x[0]))
    #     samequery = [[self.training_data_online[0][1],self.training_data_online[0][2],self.training_data_online[0][3]]]
    #     total_inputs = []
    #     total_labels = []
    #     for i in range(1,len(self.training_data_online)):
    #         if self.training_data_online[i][0] != self.training_data_online[i-1][0]:
    #             inputs,labels = self.getpair_muti(samequery)
    #             total_inputs.extend(inputs)
    #             total_labels.extend(labels)
    #             samequery.clear()
    #         samequery.append([self.training_data_online[i][1],self.training_data_online[i][2],self.training_data_online[i][3]])
        
    #     return total_inputs, total_labels
    def process_online(self):
        total_inputs = []
        total_labels = []
        for k,v in self.training_data_online.items():
            # random.seed(config.seed)
            # random.shuffle(v)
            inputs,labels = self.getpair_muti_full(v)
            total_inputs.extend(inputs)
            total_labels.extend(labels)
        # self.cur_data_online = {}
        return total_inputs,total_labels
        
    def process_online_noretrain(self):
        total_inputs = []
        total_labels = []
        for k,v in self.cur_data_online.items():
            inputs,labels = self.getpair_muti(v)
            if k not in self.training_data_online:
                self.training_data_online[k] = v
                diff_inputs = []
                diff_labels = []
            else:
                diff_inputs,diff_labels = self.getpair_muti_diff(self.training_data_online[k],self.cur_data_online[k])
                self.training_data_online[k].extend(v)
            total_inputs.extend(inputs)
            total_inputs.extend(diff_inputs)
            total_labels.extend(labels)
            total_labels.extend(diff_labels)
        self.cur_data_online = {}
        return total_inputs,total_labels
        
    def getpair(self,fea_latency):
        length = len(fea_latency)
        inputs = []
        labels = []
        if length == 1:
            return inputs,labels
        for i in range(length):
            for j in range(i+1,length):
                plan_pair = {'left':fea_latency[i][0],'right':fea_latency[j][0]}
                latency_pair = [fea_latency[i][1],fea_latency[j][1]]
                if (latency_pair[0] - latency_pair[1]) / latency_pair[0] <= self.alpha:
                    labels.append(0)
                else:
                    labels.append(1)
                inputs.append(plan_pair)
        return inputs,labels
    def getpair_out(self,old_fea_latency):
    
        inputs = []
        labels = []
        old_length = len(old_fea_latency)
        if old_length == 1:
            return inputs,labels
        to_save = []
        to_delete= []
        for i in range(0,old_length):
            if old_fea_latency[i][2]:   # 标记超时的plan
                to_delete.append(i)
            else:
                to_save.append(i)
        new_fea_latency = []
        for i in to_save:
            for j in to_delete:
                plan_pair = {'left':old_fea_latency[i][0],'right':old_fea_latency[j][0]}
                # print(old_fea_latency[i][1],old_fea_latency[j][1])
                labels.append(0)
                inputs.append(plan_pair)
            new_fea_latency.append(old_fea_latency[i])
        
        for i in range(len(new_fea_latency)):
            for j in range(i+1,len(new_fea_latency)):
                plan_pair = {'left':new_fea_latency[i][0],'right':new_fea_latency[j][0]}
                latency_pair = [new_fea_latency[i][1],new_fea_latency[j][1]]
                if (latency_pair[0] - latency_pair[1])/latency_pair[0] < self.alpha:
                    labels.append(0)
                    inputs.append(plan_pair)
                else:
                    labels.append(1)
                    inputs.append(plan_pair)
        return inputs,labels
    def getpair_muti_diff(self,data1,data2):
        inputs = []
        labels = []
        for d1 in data1:
            for d2 in data2:
                if d1[2] and d2[2]:
                    continue
                elif d1[2] and not d2[2]:
                    inputs.append({'left':d2[0],'right':d1[0]})
                    labels.append(0)
                elif not d1[2] and d2[2]:   
                    inputs.append({'left':d1[0],'right':d2[0]})
                    labels.append(0)
                elif not d1[2] and not d2[2]:
                    ratio = (d1[1] - d2[1]) / max(d1[1],d2[1])  
                    if  ratio >= config.alpha and ratio < config.beta:
                        label = 1
                    elif ratio >= config.beta:
                        label = 2
                    elif ratio < config.alpha:
                        label = 0
                    labels.append(label)
                    inputs.append({'left':d1[0],'right':d2[0]})
        
        return inputs,labels
    def getpair_muti(self,old_fea_latency):
        old_length = len(old_fea_latency)
        inputs = []
        labels = []
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
                plan_pair = {'left':old_fea_latency[i][0],'right':old_fea_latency[j][0]}
                labels.append(0)
                inputs.append(plan_pair)
            new_fea_latency.append(old_fea_latency[i])
        new_length = len(new_fea_latency)
        for i in range(new_length - 1):
            for j in range(i + 1, new_length):
                plan_pair = {'left':new_fea_latency[i][0],'right':new_fea_latency[j][0]}
                latency_pair = [new_fea_latency[i][1], new_fea_latency[j][1]]
                ratio = (latency_pair[0] - latency_pair[1]) / latency_pair[0]
                if  ratio >= config.alpha and ratio < config.beta:
                    label = 1
                elif ratio >= config.beta:
                    label = 2
                elif ratio < config.alpha:
                    label = 0
                labels.append(label)
                inputs.append(plan_pair)
        return inputs,labels
    def getpair_muti_diff_full(self,data1,data2):
        inputs = []
        labels = []
        for d1 in data1:
            for d2 in data2:
                if d1[2] and d2[2]:
                    continue
                elif d1[2] and not d2[2]:
                    if (d1[1] - d2[1]) / d1[1] >= config.beta:
                        inputs.append({'left':d1[0],'right':d2[0]})
                        labels.append(2)
                    elif (d1[1] - d2[1]) / d1[1] >= config.alpha:
                        inputs.append({'left':d1[0],'right':d2[0]})
                        labels.append(1)
                    inputs.append({'left':d2[0],'right':d1[0]})
                    labels.append(0)
                elif not d1[2] and d2[2]:
                    if (d2[1] - d1[1]) / d2[1] >= config.beta:
                        inputs.append({'left':d2[0],'right':d1[0]})
                        labels.append(2)
                    elif (d2[1] - d1[1]) / d2[1] >= config.alpha:
                        inputs.append({'left':d2[0],'right':d1[0]})
                        labels.append(1)
                    inputs.append({'left':d1[0],'right':d2[0]})
                    labels.append(0)
                elif not d1[2] and not d2[2]:
                    ratio = (d1[1] - d2[1]) / max(d1[1],d2[1])  
                    if  ratio >= config.alpha and ratio < config.beta:
                        label = 1
                    elif ratio >= config.beta:
                        label = 2
                    elif ratio < config.alpha:
                        label = 0
                    labels.append(label)
                    inputs.append({'left':d1[0],'right':d2[0]})
                    ratio = (d2[1] - d1[1]) / max(d1[1],d2[1])  
                    if  ratio >= config.alpha and ratio < config.beta:
                        label = 1
                    elif ratio >= config.beta:
                        label = 2
                    elif ratio < config.alpha:
                        label = 0
                    labels.append(label)
                    inputs.append({'left':d1[0],'right':d2[0]})
        return inputs,labels
    def getpair_muti_full(self,old_fea_latency):
        old_length = len(old_fea_latency)
        inputs = []
        labels = []
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
                if (old_fea_latency[j][1] - old_fea_latency[i][1]) / old_fea_latency[j][1] >= config.beta:
                    plan_pair = {'left':old_fea_latency[j][0],'right':old_fea_latency[i][0]}
                    labels.append(2)
                    inputs.append(plan_pair)
                elif (old_fea_latency[j][1] - old_fea_latency[i][1]) / old_fea_latency[j][1] >= config.alpha:
                    plan_pair = {'left':old_fea_latency[j][0],'right':old_fea_latency[i][0]}
                    labels.append(1)
                    inputs.append(plan_pair)
                plan_pair = {'left':old_fea_latency[i][0],'right':old_fea_latency[j][0]}
                labels.append(0)
                inputs.append(plan_pair)
            new_fea_latency.append(old_fea_latency[i])
        new_length = len(new_fea_latency)
        for i in range(new_length):
            for j in range(new_length):
                if i != j:
                    plan_pair = {'left':new_fea_latency[i][0],'right':new_fea_latency[j][0]}
                    latency_pair = [new_fea_latency[i][1], new_fea_latency[j][1]]
                    ratio = (latency_pair[0] - latency_pair[1]) / latency_pair[0] # max(latency_pair)
                    if  ratio >= config.alpha and ratio < config.beta:
                        label = 1
                    elif ratio >= config.beta:
                        label = 2
                    elif ratio < config.alpha:
                        label = 0
                    labels.append(label)
                    inputs.append(plan_pair)
        return inputs,labels