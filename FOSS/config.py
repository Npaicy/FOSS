import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
class Config:
    def __init__(self):

        # ======   PG Config   =======
        self.max_time_out = 3e5
        self.mode = ''
        self.expname = ''
        self.database = ''
        self.user = ''
        self.password = ''
        self.ip = ''
        self.port  = 5432
        self.operator_pg2hint = {'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}
        self.OperatorDict = {'NESTLOOP':1,'HASHJOIN':2,'MERGEJOIN':3}
        self.Operatortype = ['NESTLOOP','HASHJOIN','MERGEJOIN']
      
        # ====== Embed Config ========
        self.types = 20     
        self.columns = 60 # JOBRand :60 TPCDS:80  STACK:50 JOBEXT:80
        self.heightsize = 30 #30
        self.maxnode = 50 #50
        self.maxjoins = 10
        self.tablenum = 23
        self.opsnum = 18
        self.num_node_feature = 22 + self.maxjoins
        # ======     Model    ========
        self.emb_size = 16
        self.ffn_dim = 16  
        self.head_size = 10
        self.num_layers = 10
        self.pair_lr = 3e-4 # 3e-4  
        self.hidden_dim = self.emb_size * 7 + 8 * (self.emb_size // 8) + self.emb_size // 2 + 1
        # ====== General ========
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        self.seed = 3407
        self.splitpoint = [0.5, 0.05]
        self.classNum = len(self.splitpoint)
        self.maxsteps = 3
        self.num_agents = 1
        self.num_policies = 1
        self.maxbounty = 12
        self.penalty_coeff = 2
        self.timeoutcoeff = 1.01 + self.splitpoint[0]
        self.update_evaluator = True
        self.left_deep_restriction = True
    def ConfirmPath(self):
        self.total_latency_buffer = './latencybuffer/{}.json'.format(self.database)
        self.train_workload_path = './experiment/{}/train/'.format(self.mode)
        self.test_workload_path = './experiment/{}/test/'.format(self.mode)
        self.encoding_path = './model/encoding_{}.json'.format(self.mode)
        self.evaluator_path = './model/evaluator_{}.pt'.format(self.mode)
        self.agent_checkpoint = './model/planner/{}/'.format(self.mode)
        self.latency_buffer_path = './latencybuffer/{}_buffer_{}.json'.format(self.mode,self.expname)
        self.eval_output_path = './timely_result/{}.json'.format(self.mode)
        self.outfile_path = './result/{}_{}.json'.format(self.mode,self.expname)
        self.TestExperiencePool = './experiencepool/Test_{}_{}.json'.format(self.mode,self.expname)
        self.ExperiencePool = './experiencepool/Train_{}_{}.csv'.format(self.mode,self.expname)
        # self.beststeps_record = './result/{}_{}_maxsteps.json'.format(self.mode, self.maxsteps)# TO_Delete
        self.pg_latency = './pg_latency/{}.json'.format(self.mode)
        self.runstate = './runstate/{}_{}'.format(self.mode,self.expname)
