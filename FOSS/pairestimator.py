from model import PlanNetwork
import torch.nn as nn
import torch
from ASL import ASLSingleLabel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PairCompare(nn.Module):
    def __init__(self,hid_units = 256, output_dim = 3):
        nn.Module.__init__(self)
        self.embed = PlanNetwork()
        in_feature = self.embed.hidden_dim
        self.out_mlp1 = nn.Linear(in_feature + 1 , hid_units)
        self.out_mlp2 = nn.Linear(hid_units, hid_units // 2)
        self.out_mlp3 = nn.Linear(hid_units // 2, output_dim)
        self.reLU  =nn.LeakyReLU()

    def forward(self,feature):
        left = self.embed(feature['left'])
        right = self.embed(feature['right'])
        left_steps = feature['left']['steps'].float().to(left.device)
        right_steps = feature['right']['steps'].float().to(right.device)
        leftpos = torch.zeros_like(left_steps,device = left_steps.device)
        rightpos = torch.ones_like(right_steps,device = right_steps.device)
        left = torch.cat([left,leftpos], dim=-1)
        right = torch.cat([right,rightpos], dim=-1)
        hid_left = self.reLU(self.out_mlp1(left))
        hid_right = self.reLU(self.out_mlp1(right))
        out = hid_left - hid_right
        out = self.reLU(self.out_mlp2(out))
        out = self.out_mlp3(out)
        return out
    
    def get_embed(self, plan_feature):
        self.embed.eval()
        embeddings = self.embed(plan_feature)
        self.embed.train()
        return embeddings

class MyDataset(Dataset):
    def __init__(self, inputs, labels, weights):
        self.inputs = inputs
        self.labels = labels
        self.weights = weights
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx],self.weights[idx]

class PairTrainer:
    def __init__(self,genConfig, device = None):
        self.config = genConfig
        if device == None:
            self.device = self.config.device
        else:
            self.device = device
        self.seed = self.config.seed
        torch.manual_seed(self.seed)         
        torch.cuda.manual_seed(self.seed)       
        torch.cuda.manual_seed_all(self.seed) 
        self._net = PairCompare(output_dim = self.config.classNum + 1).to(self.device)
        self.optimizer = torch.optim.Adam(self._net.parameters(),lr = self.config.pair_lr)
        self.loss_fn = ASLSingleLabel()
        self.softmax = nn.Softmax(dim = 1)
        # self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, dataset, valdataset=None, testdataset=None, mybatch_size = 64, epochs = 10, writer = None):
        dataloader = DataLoader(dataset, batch_size=mybatch_size, shuffle=True)
        self._net.train()
        
        for epoch in range(epochs):
            cur_loss = 0
            num_ = 0
            for batch_inputs, batch_labels, batch_weights in dataloader:
                for k1 in batch_inputs:
                    for k2 in batch_inputs[k1]:
                        batch_inputs[k1][k2] = batch_inputs[k1][k2].to(self.device)
                batch_weights = batch_weights.to(self.device)
                batch_labels = batch_labels.to(self.device)
                prob = self._net(batch_inputs)
                loss = self.loss_fn(prob, batch_labels, batch_weights)
                cur_loss += loss.cpu().detach().item() * len(batch_labels)
                num_ += len(batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if writer is not None:
                writer.add_scalar('Loss', cur_loss / num_, epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {cur_loss / num_}")
            
            if valdataset is not None:
                valacc, valreward = self.test_dataset(valdataset)
                if  writer is not None:
                    writer.add_scalar('Reward/Val', valreward, epoch)
                    writer.add_scalar('Accuracy/Val', valacc, epoch)
            
            if testdataset is not None:
                testacc, testreward = self.test_dataset(testdataset)
                if writer is not None:
                    writer.add_scalar('Reward/Test', testreward, epoch)
                    writer.add_scalar('Accuracy/Test', testacc, epoch)

    def test_dataset(self, dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self._net.eval()
        correct = 0
        label0 = 0
        label1 = 0
        label2 = 0
        total_reward = 0
        for batch_inputs, batch_labels, batch_weights in dataloader:
            for k1 in batch_inputs:
                for k2 in batch_inputs[k1]:
                    batch_inputs[k1][k2] = batch_inputs[k1][k2].to(self.device)
            with torch.no_grad():
                prob = self._net(batch_inputs).cpu().detach()
                prob = self.softmax(prob)
                predict_ = torch.argmax(prob, dim=1)
                equal_1 = torch.logical_and(predict_ == 0, batch_labels == 0)
                equal_2 = torch.logical_and(predict_ != 0, batch_labels != 0)
                conditions = torch.logical_or(equal_1, equal_2)
                correct += conditions.sum().item()
                l0_0 = torch.logical_and(predict_ == 0, batch_labels == 0)
                l1_1 = torch.logical_and(predict_ == 1, batch_labels == 1)
                l2_2 = torch.logical_and(predict_ == 2, batch_labels == 2)
                l1_2 = torch.logical_and(predict_ == 1, batch_labels == 2)
                l2_1 = torch.logical_and(predict_ == 2 ,batch_labels == 1)
                label0 += l0_0.sum().item()
                label1 += l1_1.sum().item()
                label2 += l2_2.sum().item()
                # correct += (predict_ == batch_labels).sum().item()
                # correct += l1_2.sum().item()
                # correct += l2_1.sum().item()
                # conditions = torch.stack((l0_0, l1_1, l2_2, l1_2, l2_1))
                reward_w = torch.where(conditions, torch.tensor(1), torch.tensor(-1))
                reward = reward_w * batch_weights
                total_reward += reward.sum()
                # correct += ((torch.round(prob) == batch_labels).sum().item())
        # import ipdb
        # ipdb.set_trace()
        
        label = np.array(dataset.labels)
        print('Label 0:{}/{} Label 1:{}/{} Label 2:{}/{}'.format(label0,sum(label == 0),
                                                                 label1,sum(label == 1),
                                                                 label2,sum(label == 2)))
        print(f'Evaluator Accuracy:{correct/len(dataset)}  Average Reward:{total_reward / len(dataset)}')

        return correct/len(dataset), total_reward / len(dataset)

    def get_embed(self, plan_feature):
        return self._net.get_embed(plan_feature)
    
    def predict_pair(self, left,right):
        self._net.eval()
        left = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in left.items()}
        right = {k: torch.tensor(v).to(self.device).unsqueeze(0) for k, v in right.items()}
        feature = {'left':left,'right':right}
        with torch.no_grad():
            prob = self._net(feature)
            prob = self.softmax(prob)
        predicted_class = torch.argmax(prob, dim=1).tolist()
        return predicted_class[0], prob.detach().numpy().tolist()[0]
    
    def predict_list(self, inputs):
        self._net.eval()
        batch_size = 64
        num_batches = (len(inputs) + batch_size - 1) // batch_size
        predicted_classes = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_inputs = inputs[start_idx:end_idx]

            left_batch = {k: torch.cat([torch.tensor(input['left'][k]).to(self.device).unsqueeze(0) for input in batch_inputs], dim=0) for k in inputs[0]['left'].keys()}
            right_batch = {k: torch.cat([torch.tensor(input['right'][k]).to(self.device).unsqueeze(0) for input in batch_inputs], dim=0) for k in inputs[0]['right'].keys()}

            with torch.no_grad():
                prob = self._net({'left': left_batch, 'right': right_batch})

            predicted_class = torch.argmax(prob, dim=1).cpu().detach().numpy()
            predicted_classes.extend(predicted_class.tolist())
        return predicted_classes
    
    def predict_epi(self, hint_feature):
        self._net.eval()
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
        best_hint = hint_norepeat[0]
        best_feature = preinputs[0]
        for k in range(1, len(hint_norepeat)):
            feature_compared = preinputs[k]
            feature = {'left':best_feature,'right':feature_compared}
            with torch.no_grad():
                prob = self._net(feature)
                prob = self.softmax(prob)
            predicted_class = torch.argmax(prob, dim=1).tolist()
            if predicted_class[0] > 0:
                best_hint = hint_norepeat[k]
                best_feature = preinputs[k]
        best_feature = {k: v.squeeze(0).numpy() for k, v in best_feature.items()}
        return best_hint, best_feature

    def predict_epi_parallel(self,hint_feature):
        self._net.eval()
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
            prob = self._net({'left': left_batch, 'right': right_batch})
        predicted_class = torch.argmax(prob, dim=1).tolist()
        optimal_idx = 0
        for i in range(1,len(preinputs)):
            if predicted_class[identifier[(optimal_idx,i)]] > 0:
                optimal_idx = i
        optimal_hint = hint_norepeat[optimal_idx]
        optimal_feature = {k: v.squeeze(0).cpu().numpy() for k, v in preinputs[optimal_idx].items()}
        return optimal_hint, optimal_feature
    
    def retrainmodel(self):
        del self._net
        # torch.manual_seed(self.seed)          
        # torch.cuda.manual_seed(self.seed)      
        # torch.cuda.manual_seed_all(self.seed) 
        self._net = PairCompare(output_dim = self.config.classNum + 1).to(self.device)
        self.optimizer = torch.optim.Adam(self._net.parameters(), lr = self.config.pair_lr)

    def save_model(self, model_path):
        torch.save(self._net.state_dict(), model_path)
        
    def load_model(self, model_path):
        self._net.load_state_dict(torch.load(model_path, map_location=self.device))
    