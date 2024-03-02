import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
# 3096 2048
class Config:
    def __init__(self):

        # ======   PG Config   =======
        # tpcds timeout 2e5 AAM 5e-4 RL 1e-4
        self.max_time_out = 3e5# 3e5
        self.mode = 'JOBRand'
        self.expname = ''
        self.database = 'imdb'
        self.user = ''
        self.password = ''
        self.ip = '127.0.0.1'
        self.port  = 5431
        self.operator_pg2hint = {'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}
        self.OperatorDict = {'NESTLOOP':1,'HASHJOIN':2,'MERGEJOIN':3}
        self.Operatortype = ['NESTLOOP','HASHJOIN','MERGEJOIN']
      
        # ====== Embed Config ========
        self.types = 20     
        self.columns = 100
        self.heightsize = 64
        self.maxnode = 50
        self.maxjoins = 10
        self.tablenum = 22
        self.opsnum = 13
        # ======     Model    ========
        self.emb_size = 16
        self.ffn_dim = 16  
        self.head_size = 10
        self.num_layers = 10
        self.pair_lr = 3e-4 # 3e-4  
        self.hidden_dim = self.emb_size * 7 + 4 * (self.emb_size // 8) + self.emb_size // 2 
        # ====== General ========
        self.device = torch.device("cuda")
        self.seed = 3407
        self.splitpoint = [0.5, 0.05]
        self.classNum = len(self.splitpoint)
        self.maxsteps = 3
        self.maxsamples = 250 #300
        self.num_agents = 1
        self.num_policies = 1
        self.maxbounty = 12
        self.penalty_coeff = 2
        self.timeoutcoeff = 1.01 + self.splitpoint[0] # 1.50
    def ConfirmPath(self):
        self.total_latency_buffer = '../latencybuffer/{}.json'.format(self.database)
        self.train_workload_path = '../experiment/{}/train/'.format(self.mode)
        self.test_workload_path = '../experiment/{}/test/'.format(self.mode)
        self.encoding_path = '../savedmodel/encoding_{}.json'.format(self.mode)
        self.model_path = '../savedmodel/AAM_{}.pt'.format(self.mode)
        self.offline_data_path = '../data/latencyds_pool_{}_{}.csv'.format(self.mode,self.expname)
        self.agent_checkpoint = '../agentcheckpoint/{}/'.format(self.mode)
        self.latency_buffer_path = '../latencybuffer/{}_buffer_{}.json'.format(self.mode,self.expname)
        self.eval_output_path = '../timely_result/{}.json'.format(self.mode)
        self.outfile_path = '../result/{}_{}.json'.format(self.mode,self.expname)
        self.ExperiencePool = '../experiencepool/{}_{}.json'.format(self.mode,self.expname)
        self.beststeps_record = '../result/{}_{}_maxsteps.json'.format(self.mode, self.maxsteps)# TO_Delete
        self.pg_latency = '../pg_latency/{}.json'.format(self.mode)
        self.runstate = '../runstate/{}_{}'.format(self.mode,self.expname)