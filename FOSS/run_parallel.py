
import ray
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np
import time
import argparse
import random
import os,shutil
from FOSSEnv import FOSSEnvTrain, FOSSEnvTest
from model import CustomModel
from manager import QueryManager, BestPlanManager, ResultManager
from config import Config
from planhelper import PlanHelperRemote
from learner import Learner,RemoteAAM


ModelCatalog.register_custom_model("gen_model", CustomModel)
def gen_policy(i):
    if i % 2 == 0:
        gamma_ = 0.99
        lr = 5e-5
    else:
        gamma_ = 0.95
        lr = 1e-4
    config = PPOConfig.overrides(
        model={
            "custom_model": 'gen_model',
            "vf_share_layers": True,
            "fcnet_hiddens": [384, 384, 384],
            "fcnet_activation": "tanh",
        },
        entropy_coeff = 0.2,
        kl_coeff = 1.5,
        lambda_ = 0.9,
        gamma = gamma_, # 0.9
        lr = lr)
    return PolicySpec(config=config)
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = "policy_{}".format(agent_id)
    return pol_id

class FOSS():
    def __init__(self,config):
        self.genConfig    = config
        self.planhelper   = PlanHelperRemote.remote(self.genConfig)
        self.bpm          = BestPlanManager.remote(self.genConfig)
        self.predictor    = RemoteAAM.remote(self.genConfig)
        self.learner      = Learner.remote(self.bpm, self.planhelper, self.predictor, self.genConfig)
        self.planner      = None
        self.writer       = SummaryWriter(log_dir = self.genConfig.runstate)
        self.queryManager = QueryManager(self.genConfig, planhelper = self.planhelper)
        self.evalEnv      = FOSSEnvTest({'planhelper':self.planhelper,'genConfig':self.genConfig})
        self.anaManger    = ResultManager(self.genConfig, self.writer)
    def RunBaseline(self):
        self.baselineTrain = 0
        self.baselineTest  = 0
        self.baseline      = {}
        while True:
            sql, query_id, oneloop = self.queryManager.get2validate()
            feature_dict,hint,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('', sql, True, query_id = query_id))
            feature_dict['steps'] = np.array([0])
            latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            key_out = '_'.join(['base',query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(cost_plan_json['Planning Time'] * 1000)])
            self.anaManger.recordRuning(key_out, value_out)
            self.anaManger.recordeval(query_id, latency_timeout[0])
            self.anaManger.recordQuery(query_id, False)
            print('\t Train Set Query:{} Latency:{:.4f}'.format(query_id,latency_timeout[0]))
            self.baseline[query_id] = latency_timeout[0]
            self.bpm.update_AAM_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
            self.bpm.update_RL_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
            self.learner.CollectSample.remote(query_id,feature_dict,latency_timeout[0],latency_timeout[1])
            if left_deep:
                self.bpm.updateMask.remote(toTrain = [query_id])
            else:
                self.bpm.updateMask.remote(toMask = [query_id])
            self.baselineTrain += latency_timeout[0]
            if oneloop:
                break
        print('Baseline_train:{:.4f}'.format(self.baselineTrain))

        while True:
            sql,query_id,oneloop = self.queryManager.get2eval()
            _, _,_, plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,False,query_id = query_id))
            latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            key_out = '_'.join(['base',query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(plan_json['Planning Time'] * 1000)])
            self.anaManger.recordRuning(key_out,value_out)
            self.anaManger.recordeval(query_id,latency_timeout[0])
            self.anaManger.recordQuery(query_id,True)
            print('\t Test Set Query:{} Latency:{:.4f}'.format(query_id,latency_timeout[0]))
            self.baselineTest += latency_timeout[0]
            self.baseline[query_id] = latency_timeout[0]
            if oneloop:
                break
        self.anaManger.recordMetric(0)
        print('Baseline_test:{:.4f}'.format(self.baselineTest))
        self.learner.getBaseline.remote(self.baseline)
        latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
        ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
        if not os.path.exists(self.genConfig.pg_latency):
            shutil.copy(self.genConfig.latency_buffer_path, self.genConfig.pg_latency)

    def BuildPlanner(self, train_batch_size, workersNum, planner_path = None):
        policies = {"policy_{}".format(i): gen_policy(i) for i in range(self.genConfig.num_policies)}
        planner_config = (
            PPOConfig()
            .environment(FOSSEnvTrain, env_config = {'bestplanmanager':self.bpm,'genConfig':self.genConfig}, disable_env_checking = True)
            .training(train_batch_size = train_batch_size)
            .multi_agent(policies = policies, policy_mapping_fn = policy_mapping_fn)
            .resources(num_gpus = 1,num_gpus_per_learner_worker = 1)\
            .rollouts(num_rollout_workers = workersNum, num_envs_per_worker = 1)#,sample_async=True)
        )
        planner_config['seed'] = self.genConfig.seed
        self.planner           = planner_config.build(logger_creator = None)
        if planner_path != None:
            self.planner.from_checkpoint(planner_path)

    def InitAAM(self, initialPro = 0.25):
        estotal,_ = self.Validate(0, Explore = True, initialPro = initialPro)
        self.writer.add_scalar('Accumulated Excecuted Plans Num', 0, 0)
        self.writer.add_scalar('Train Set Best Plan WRL', estotal / self.baselineTrain, 0)
        runner_ref = self.learner.Runing.remote()
        time.sleep(5)
        ray.get(self.bpm.update_schedule.remote(True))
        ray.get(runner_ref)

    def Validate(self, valIter, loopPro = 0.2, initialPro = 0.25, IsTest = False, Explore = False):
        actions = {}
        balanceKeys = None
        if not IsTest:
            sortedQueryID = ray.get(self.planhelper.GetSortedQueryID.remote())
            balanceKeys = sortedQueryID[:int(loopPro * len(sortedQueryID))]
        while True:
            if IsTest:
                sql, query_id, oneloop = self.queryManager.get2eval()
            else:
                sql, query_id, oneloop = self.queryManager.get2validate()
            sqlinfo = {'sql':sql,'query_id':query_id}
            planstart = time.time()
            obs, info = self.evalEnv.reset(options = sqlinfo)
            hint_feature = [(info['hint'], deepcopy(obs[0]))]
            steps = 1
            if info['useFOSS']:
                done = False
                add_bpmCandidate = False
                while not done:
                    actions.clear()
                    for i in range(self.genConfig.num_policies):
                        actions[i] = self.planner.compute_single_action(obs[i], policy_id = 'policy_{}'.format(i), explore = Explore)
                    obs, reward, terminated, _, info_all = self.evalEnv.step(actions)
                    for k in info_all:
                        self.anaManger.recordExp(query_id, info_all[k]['hint'], k, steps)
                        hint_feature.append((info_all[k]['hint'],deepcopy(obs[k])))
                    steps += 1
                    if terminated['__all__'] == True:
                        done = True
                if not Explore:
                    # optimal_hint,optimal_feature = ray.get(learner.GetPrediction.remote(hint_feature)) # 这里会有一些阻塞，因为跟learner用的同一个RM
                    optimal_hint,optimal_feature = ray.get(self.predictor.GetPrediction.remote(hint_feature))
                    plantime = time.time() - planstart 
                    if IsTest:
                        latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, query_id))
                    else:
                        latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, query_id, timeout = self.genConfig.timeoutcoeff * self.baseline[query_id]))
                        if query_id in balanceKeys:
                            add_bpmCandidate = True
                        if iscollect:
                            ray.get(self.learner.CollectSample.remote(query_id,optimal_feature,latency_timeout[0],latency_timeout[1]))
                else:
                    if random.random() < initialPro:
                        add_bpmCandidate = True
                if add_bpmCandidate:
                    for hint,feature in hint_feature[1:]:
                        del feature['action_mask']
                        self.bpm.add_balances.remote(query_id, hint, feature, sql)
            else:
                plantime =  time.time() - planstart 
                latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            if not Explore:
                if not IsTest:
                    bestfeature, bestexec = ray.get(self.bpm.get_AAM_esbest.remote(query_id))
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    self.anaManger.recordBestPlanSteps(query_id,int(self.genConfig.maxsteps * optimal_feature['steps'][0]), latency_timeout[0])# TO_Delete
                    if Advbybest >= self.genConfig.splitpoint[-1]:
                        self.bpm.update_AAM_esbest.remote(query_id, deepcopy(optimal_feature),latency_timeout[0])
                    self.bpm.update_RL_esbest.remote(query_id,deepcopy(optimal_feature),latency_timeout[0])
                self.anaManger.recordeval(query_id,latency_timeout[0])
                if IsTest:
                    key_out = '_'.join(['test', str(valIter), query_id])
                else:
                    key_out = '_'.join(['train',str(valIter),query_id])
                value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(plantime * 1000)])
                self.anaManger.recordRuning(key_out,value_out)
            if oneloop:
                break 
        estotal, best_steps = ray.get(self.bpm.get_AAM_best.remote())
        return estotal, best_steps
    
    def Run(self, totalIter = 300, valFreq = 5):
        ray.get(self.bpm.update_schedule.remote(False))
        runner_ref = self.learner.Runing.remote()
        for trainIter in range(totalIter):
            latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
            ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
            result = self.planner.train()
            print(f'TrainIter {trainIter}')
            if trainIter <= 3 or (trainIter + 1) % valFreq == 0:
                ray.get(self.bpm.update_schedule.remote(True))
                accumulatedPlans = ray.get(runner_ref)
                self.writer.add_scalar('Accumulated Excecuted Plans Num', accumulatedPlans, trainIter + 1)
                estotal, best_steps = self.Validate(trainIter + 1)
                self.writer.add_scalar('Train Set Best Plan WRL', estotal / self.baselineTrain, trainIter + 1)
                self.Validate(trainIter + 1, IsTest = True)
                self.anaManger.recordMetric(trainIter + 1)
                self.anaManger.writeout()
                # checkpoint = self.planner.save(checkpoint_dir = self.genConfig.agent_checkpoint)
                ray.get(self.bpm.update_schedule.remote(False))
                runner_ref = self.learner.Runing.remote()

    def Close(self):
        self.anaManger.close()  
        ray.shutdown()
    

if __name__ == "__main__":
    SupportWorkload = ['JOBRand', 'JOB', 'TPCDS', 'STACK']

    parser = argparse.ArgumentParser("FOSS Controller")
    parser.add_argument("--Workload",choices = SupportWorkload,
                        help="Choose the Workload from [JOBRand, TPCDS, STACK]")
    parser.add_argument("--ExpName",
                        help="The experiment name must be unique")
    parser.add_argument("--Database",
                        help="The Database Name")
    parser.add_argument("--Seed", type=int, default = 3407,
                        help="Random Seed")
    parser.add_argument("--Maxsteps", type=int, default = 3,
                        help="Max steps of agent")
    parser.add_argument("--Maxsamples", type=int, default = 250,
                        help="Maximum number of samples in a single iteration (in best plan manager)")
    parser.add_argument("--TotalIter", type=int, default = 300,
                        help="The total number of iterations for which the planner is trained")
    parser.add_argument("--ValidateFreq", type=int, default = 5,
                        help="The frequency of validation")
    parser.add_argument("--Agents", type=int, default = 1,
                        help="The Num of Agents.")
    parser.add_argument("--PenaltyCoeff", type=int, default = 2,
                        help="The coefficient of penalty.")
    
    args = parser.parse_args()

    config = Config()
    if args.Workload:
        config.mode = args.Workload
    if args.Seed:
        config.seed = args.Seed
    if args.ExpName:
        config.expname = args.ExpName
    else:
        config.expname = str(config.seed)+'_' +str(args.Maxsteps)
    if args.Database:
        config.database = args.Database
    if args.Maxsteps:
        config.maxsteps = args.Maxsteps
    if args.Maxsamples:
        config.maxsamples = args.Maxsamples
    if args.Agents:
        config.num_agents = args.Agents
        config.num_policies = args.Agents
    if args.PenaltyCoeff:
        config.penalty_coeff = args.PenaltyCoeff
    config.ConfirmPath()
    foss = FOSS(config)
    print("Running Baseline......")
    foss.RunBaseline()
    print("Building Planner......")
    foss.BuildPlanner(train_batch_size = 3000, workersNum = 5)
    initialPro = 0.25
    print("Initialing AAM........")
    foss.InitAAM(initialPro = initialPro)
    time.sleep(10)  # wait for AAM Training
    print("Training FOSS.........")
    foss.Run(totalIter = args.TotalIter, valFreq = args.ValidateFreq)
    foss.Close()
