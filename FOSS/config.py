import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
# 3096 2048
class Config:
    def __init__(self):

        # ======   PG Config   =======
        self.max_time_out = 2e5
        self.mode = 'JOBRand'
        self.expname = '1234'
        self.database = 'imdb'
        self.user = 'postgres'
        self.password = ''
        self.ip = '127.0.0.1'
        self.port  = 5432
        # self.geqo_threshold = 1
        self.operator_pg2hint = {'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}
        self.OperatorDict = {'NESTLOOP':1,'HASHJOIN':2,'MERGEJOIN':3}
        self.Operatortype = ['NESTLOOP','HASHJOIN','MERGEJOIN']
      
        # ====== Embed Config ========
        # self.joins = 200#100    #最大连接数
        self.types = 20     #类型数
        self.columns = 100
        self.heightsize = 64#64     #树的最大高度
        self.maxnode = 50#50
        self.maxjoins = 10
        self.expands = self.columns // 50
        # ======     Model    ========
        self.emb_size = 16#32
        self.ffn_dim = 16  #32    # 前馈神经网络宽度
        self.head_size = 10
        self.num_layers = 10
        self.pair_lr = 3e-4  
        self.hidden_dim = self.emb_size * 7 + 2 * (self.emb_size // 8) + 1 + self.emb_size // 2
        # ====== General ========
        self.device = torch.device("cuda")
        self.seed = 1234
        self.alpha = 0.05 # pair分割点，重要参数
        self.beta = 0.75
        # self.max_pooling = 5000
        # self.n_copys = 1 
        self.maxsteps = 3
        self.num_agents = 1
        self.num_policies = 1
        self.querynum_perepoch = 10
        self.startiter = 1
        self.maxbounty = 12
        self.timeoutcoeff = 1.5
        # self.hist_file = '../hist&sample/{}_hist_file_200.pkl'.format(self.database)
        # self.table_sample = '../hist&sample/{}_sample'.format(self.mode)
        self.total_latency_buffer = '../latencybuffer/{}.json'.format(self.database)
        self.train_workload_path = '../experiment/{}/train/'.format(self.mode)
        self.test_workload_path = '../experiment/{}/test/'.format(self.mode)
        self.encoding_path = '../savedmodel/encoding_{}.json'.format(self.mode)
        self.model_path = '../savedmodel/pairtrainer_{}.pt'.format(self.mode)
        self.offline_data_path = '../data/latencyds_pool_{}_{}.csv'.format(self.mode,self.expname)
        self.agent_checkpoint = '../agentcheckpoint/{}/'.format(self.mode)
        self.latency_buffer_path = '../latencybuffer/{}_buffer_{}.json'.format(self.mode,self.expname)
        self.eval_output_path = '../timely_result/{}.json'.format(self.mode)
        self.outfile_path = '../result/{}_{}.json'.format(self.mode,self.expname)
        self.ExperiencePool = '../experiencepool/{}_{}.json'.format(self.mode,self.expname)
        self.pg_latency = '../pg_latency/{}.json'.format(self.mode)
        self.runstate = '../runstate/{}_{}'.format(self.mode,self.expname)