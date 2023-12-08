from model import PlanNetwork
import torch.nn as nn
import torch
from config import Config
from ASL import ASLSingleLabel

config = Config()
class PairCompare(nn.Module):
    def __init__(self,hid_units = 256, device = None):
        nn.Module.__init__(self)
        self.device = config.device if device == None else device
        self.embed = PlanNetwork().to(self.device)
        in_feature = self.embed.hidden_dim
        self.out_mlp1 = nn.Linear(in_feature + 1 , hid_units)
        self.out_mlp2 = nn.Linear(hid_units, hid_units // 2)
        self.out_mlp3 = nn.Linear(hid_units // 2, 3) #modify
        # self.out_dropout = nn.Dropout(0.3)
        self.reLU  =nn.LeakyReLU()
    def forward(self,feature):
        left = self.embed(feature['left'])
        right = self.embed(feature['right'])
        left_steps = feature['left']['steps'].float().to(self.device)
        right_steps = feature['right']['steps'].float().to(self.device)
        leftpos = torch.zeros_like(left_steps,device = self.device)
        rightpos = torch.ones_like(right_steps,device = self.device)
        left = torch.cat([left,leftpos], dim=-1)
        right = torch.cat([right,rightpos], dim=-1)
        hid_left = self.reLU(self.out_mlp1(left))
        hid_right = self.reLU(self.out_mlp1(right))
        out = hid_left - hid_right
        out = self.reLU(self.out_mlp2(out))
        # out = self.out_dropout(out)
        out = self.out_mlp3(out)
        return out



from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# @ray.remote
class PairTrainer:
    def __init__(self,device = None):
        if device == None:
            self.device = config.device
        else:
            self.device = device
        self.seed = config.seed
        torch.manual_seed(self.seed)            # 为CPU设置随机种子
        torch.cuda.manual_seed(self.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(self.seed) 
        self.model = PairCompare(device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = config.pair_lr)
        self.loss_fn = ASLSingleLabel()
        # self.loss_fn = nn.CrossEntropyLoss()
        
    def train_dataset(self,dataset,valdataset = None,testdataset = None,mybatch_size = 32,epochs = 10):
        dataloader = DataLoader(dataset, batch_size=mybatch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            cur_loss = 0
            num_ = 0
            for batch_inputs, batch_labels in dataloader:
                for k1 in batch_inputs:
                    for k2 in batch_inputs[k1]:
                        batch_inputs[k1][k2] = batch_inputs[k1][k2].to(self.device)
                batch_labels = batch_labels.to(self.device)
                prob = self.model(batch_inputs)
                prob = prob.squeeze(dim=1)  
                loss = self.loss_fn(prob, batch_labels)   
                cur_loss += loss.item()
                num_ += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {cur_loss/num_}")
            if valdataset!= None:
                print('Validating......')
                self.test_dataset(valdataset)
            if testdataset != None:
                print('Testing......')
                self.test_dataset(testdataset)
                
    def test_dataset(self, dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model.eval()
        correct = 0
        label0 = 0
        label1 = 0
        label2 = 0
        for batch_inputs, batch_labels in dataloader:
            for k1 in batch_inputs:
                for k2 in batch_inputs[k1]:
                    batch_inputs[k1][k2] = batch_inputs[k1][k2].to(self.device)
            with torch.no_grad():
                prob = self.model(batch_inputs).cpu().detach()

                # batch_labels = batch_labels.unsqueeze(1)
                # print(torch.cat([torch.argmax(prob, dim=1).unsqueeze(1),batch_labels],dim = -1))
                # batch_labels = batch_labels.squeeze(1)
                predict_ = torch.argmax(prob, dim=1)
                label0 += torch.logical_and(predict_ == 0, batch_labels == 0).sum().item()
                label1 += torch.logical_and(predict_ == 1, batch_labels == 1).sum().item()
                label2 += torch.logical_and(predict_ == 2, batch_labels == 2).sum().item()
                correct += (predict_ == batch_labels).sum().item()
                correct += torch.logical_and(predict_ == 1, batch_labels == 2).sum().item()
                correct += torch.logical_and(predict_ == 2 ,batch_labels == 1).sum().item()
                # correct += ((torch.round(prob) == batch_labels).sum().item())
        # import ipdb
        # ipdb.set_trace()
        import numpy as np
        label = np.array(dataset.labels)
        print('Label 0:{}/{} Label 1:{}/{} Label 2:{}/{}'.format(label0,sum(label == 0),
                                                                 label1,sum(label == 1),
                                                                 label2,sum(label == 2)))
        print('AAM Current Accuracy:',correct/len(dataset))
        return correct/len(dataset)

    
    def predict_pair(self, left,right):
        self.model.eval()
        left = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in left.items()}
        right = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in right.items()}
        feature = {'left':left,'right':right}
        with torch.no_grad():
            prob = self.model(feature)
        predicted_class = torch.argmax(prob, dim=1).tolist()
        return predicted_class[0]

    def predict_step(self, curr,optimal,base):
        self.model.eval()
        curr = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in curr.items()}
        optimal = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in optimal.items()}
        base = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in base.items()}
        features = [{'left':optimal,'right':curr},{'left':base,'right':curr}]
        left_batch = {k: torch.cat([feature['left'][k] for feature in features], dim=0) for k in curr.keys()}
        right_batch = {k: torch.cat([feature['right'][k] for feature in features], dim=0) for k in curr.keys()}
        with torch.no_grad():
            prob = self.model({'left': left_batch, 'right': right_batch})
        predicted_class = torch.argmax(prob, dim=1).tolist()
        return predicted_class[0],predicted_class[1]
    def predict_step3(self, esbest, curr, tocmpare):
        self.model.eval()
        curr = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in curr.items()}
        esbest = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in esbest.items()}
        tocmpare = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in tocmpare.items()}
        features = [{'left':esbest,'right':curr},{'left':tocmpare,'right':curr}]
        left_batch = {k: torch.cat([feature['left'][k] for feature in features], dim=0) for k in curr.keys()}
        right_batch = {k: torch.cat([feature['right'][k] for feature in features], dim=0) for k in curr.keys()}
        with torch.no_grad():
            prob = self.model({'left': left_batch, 'right': right_batch})
        predicted_class = torch.argmax(prob, dim=1).tolist()
        return predicted_class[0],predicted_class[1]
    def predict_step4(self, esbest, curr, tocompare, base):
        self.model.eval()
        curr = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in curr.items()}
        esbest = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in esbest.items()}
        tocompare = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in tocompare.items()}
        base = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in base.items()}
        features = [{'left':esbest,'right':curr},{'left':tocompare,'right':curr},{'left':base,'right':curr}]
        left_batch = {k: torch.cat([feature['left'][k] for feature in features], dim=0) for k in curr.keys()}
        right_batch = {k: torch.cat([feature['right'][k] for feature in features], dim=0) for k in curr.keys()}
        with torch.no_grad():
            prob = self.model({'left': left_batch, 'right': right_batch})
        predicted_class = torch.argmax(prob, dim=1).tolist()
        return predicted_class[0],predicted_class[1],predicted_class[2]
    
    # def predict_step4(self, curr, optimal, base, best):
    #     self.model.eval()
    #     curr = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in curr.items()}
    #     optimal = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in optimal.items()}
    #     base = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in base.items()}
    #     best = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in best.items()}
    #     features = [{'left':optimal,'right':curr},{'left':base,'right':curr},{'left':curr,'right':best}]
    #     left_batch = {k: torch.cat([feature['left'][k] for feature in features], dim=0) for k in curr.keys()}
    #     right_batch = {k: torch.cat([feature['right'][k] for feature in features], dim=0) for k in curr.keys()}
    #     with torch.no_grad():
    #         prob = self.model({'left': left_batch, 'right': right_batch})
    #     predicted_class = torch.argmax(prob, dim=1).tolist()
    #     return predicted_class[0],predicted_class[1],predicted_class[2]
    def predict_epi(self,hint_feature):
        self.model.eval()
        hint_norepeat = []
        preinputs = []
        for hf in hint_feature:
            if hf[0] not in hint_norepeat:
                hint_norepeat.append(hf[0])
                tmpdict = {}
                for k, v in hf[1].items():
                    if k != 'action_mask':
                        tmpdict[k] = torch.tensor(v).to(self.device).unsqueeze(0)
                preinputs.append(tmpdict)
                
        inputs = []
        identifier = {}
        counts = 0
        for i in range(len(preinputs) - 1):
            left = preinputs[i]
            for j in range(i + 1,len(preinputs)):
                right = preinputs[j]
                inputs.append({'left':left,'right':right})
                identifier[(i,j)] = counts
                counts += 1
        
        left_batch = {k: torch.cat([input['left'][k] for input in inputs], dim=0) for k in preinputs[0].keys()}
        right_batch = {k: torch.cat([input['right'][k] for input in inputs], dim=0) for k in preinputs[0].keys()}
        with torch.no_grad():
            prob = self.model({'left': left_batch, 'right': right_batch})
        predicted_class = torch.argmax(prob, dim=1).tolist()
        optimal_idx = 0
        for i in range(1,len(preinputs)):
            if predicted_class[identifier[(optimal_idx,i)]] > 0:
                optimal_idx = i
        optimal_hint = hint_norepeat[optimal_idx]
        optimal_feature = {k: v.squeeze(0).cpu().numpy() for k, v in preinputs[optimal_idx].items()}
        return optimal_hint, optimal_feature
    
    def retrainmodel(self):
        del self.model
        
        torch.manual_seed(self.seed)            # 为CPU设置随机种子
        torch.cuda.manual_seed(self.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(self.seed) 
        self.model = PairCompare().to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = config.pair_lr)
        self.loss_fn = ASLSingleLabel()
                    
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    