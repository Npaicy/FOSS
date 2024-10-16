from config import Config
import ray
from ray.rllib.algorithms.ppo import PPOConfig
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
from planhelper import RemotePlanHelper
from learner import Learner, RemoteEvaluator


ModelCatalog.register_custom_model("gen_model", CustomModel)
def gen_policy(i):
    if i % 2 == 0:
        gamma_ = 0.99
        lr = 5e-5 # JOBRand:5e-5 TPCDS:1e-4
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
        entropy_coeff = 0.05, #JOBRand:0.05 TPCDS:0.01
        kl_coeff = 1.5,
        lambda_ = 0.9,
        gamma = gamma_,
        lr = lr)
    return PolicySpec(config=config)
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = "policy_{}".format(agent_id)
    return pol_id

class FOSS():
    def __init__(self, config):
        self.genConfig    = config
        self.planhelper   = RemotePlanHelper.remote(self.genConfig)
        self.bpm          = BestPlanManager.remote(self.genConfig)
        self.predictor    = RemoteEvaluator.remote(self.genConfig)
        if self.genConfig.update_evaluator:
            self.learner      = Learner.remote(self.bpm, self.planhelper, self.predictor, self.genConfig)
        else:
            self.learner  = None
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
            feature_dict,hint,left_deep,plan_json = ray.get(self.planhelper.GetFeature.remote('', sql, True, query_id = query_id))
            feature_dict['steps'] = np.array([0])
            latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            planning_time = plan_json['Planning Time'] * 1000
            key_out = '_'.join(['base',query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
            self.anaManger.recordRuning(key_out, value_out)
            self.anaManger.recordeval(query_id, latency_timeout[0], planning_time)
            self.anaManger.recordQuery(query_id, False)
            print('\t Train Set Query:{} Execution Time:{:.4f}'.format(query_id,latency_timeout[0]))
            self.baseline[query_id] = latency_timeout[0]
            self.bpm.update_evaluator_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
            self.bpm.update_RL_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
            if self.learner:
                self.learner.CollectSample.remote(query_id, feature_dict, latency_timeout[0], latency_timeout[1])
            # if left_deep:
            self.bpm.updateMask.remote(toTrain = [query_id])
            # else:
            #     self.bpm.updateMask.remote(toMask = [query_id])
            self.baselineTrain = self.baselineTrain + latency_timeout[0] + planning_time
            if oneloop:
                break
        print('Baseline_train:{:.4f}'.format(self.baselineTrain / 1000))

        while True:
            sql, query_id, oneloop = self.queryManager.get2eval()
            _, _,_, plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,False,query_id = query_id))
            latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            planning_time = plan_json['Planning Time'] * 1000
            key_out = '_'.join(['base',query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
            self.anaManger.recordRuning(key_out,value_out)
            self.anaManger.recordeval(query_id, latency_timeout[0], planning_time)
            self.anaManger.recordQuery(query_id,True)
            print('\t Test Set Query:{} Execution Time:{:.4f}'.format(query_id,latency_timeout[0]))
            self.baselineTest = self.baselineTest + latency_timeout[0] + planning_time
            self.baseline[query_id] = latency_timeout[0]
            if oneloop:
                break
        self.anaManger.recordMetric(0)
        print('Baseline_test:{:.4f}'.format(self.baselineTest / 1000))
        if self.learner:
            self.learner.updateBaseline.remote(self.baseline)
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

    def InitEvaluator(self, initialPro = 0.25):
        estotal,_ = self.Validate(0, Explore = True, initialPro = initialPro)
        self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', 0, 0)
        self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, 0)
        runner_ref = self.learner.Runing.remote()
        time.sleep(5)
        ray.get(self.bpm.update_schedule.remote(True))
        ray.get(runner_ref)

    def Validate(self, valIter, loopPro = 0, initialPro = 0.25, isTest = False, Explore = False):
        actions = {}
        balanceKeys = None
        if not isTest:
            sortedQueryID = ray.get(self.planhelper.GetSortedQueryID.remote())
            balanceKeys = sortedQueryID[:int(loopPro * len(sortedQueryID))]
        while True:
            if isTest:
                sql, query_id, oneloop = self.queryManager.get2eval()
            else:
                sql, query_id, oneloop = self.queryManager.get2validate()
            sqlinfo = {'sql':sql, 'query_id':query_id}
            planning_start = time.time()
            obs, info = self.evalEnv.reset(options = sqlinfo)
            hint_feature = [(info['hint'], deepcopy(obs[0]))]
            steps = 1
            if info['useFOSS']:
                done = False
                add_bpmCandidate = False
                while not done:
                    actions.clear()
                    for i in range(self.genConfig.num_policies):
                        actions[i] = self.planner.compute_single_action(obs[i], policy_id = 'policy_{}'.format(i), explore = True) #  explore = True or False? 
                    obs, reward, terminated, _, info_all = self.evalEnv.step(actions)
                    for k in info_all:
                        if isTest:
                            self.anaManger.recordExp(query_id, info_all[k]['hint'], k, steps)
                        hint_feature.append((info_all[k]['hint'],deepcopy(obs[k])))
                    steps += 1
                    if terminated['__all__'] == True:
                        done = True
                if not Explore:
                    # optimal_hint,optimal_feature = ray.get(learner.GetPrediction.remote(hint_feature)) # 这里会有一些阻塞，因为跟learner用的同一个Evaluator
                    optimal_hint,optimal_feature = ray.get(self.predictor.GetPrediction.remote(hint_feature))
                    planning_time = time.time() - planning_start
                    if isTest:
                        latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, query_id))
                    else:
                        latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, query_id, timeout = self.genConfig.timeoutcoeff * self.baseline[query_id]))
                        if query_id in balanceKeys:
                            add_bpmCandidate = True  # if not balance
                        if iscollect and self.learner:
                            ray.get(self.learner.CollectSample.remote(query_id,optimal_feature,latency_timeout[0],latency_timeout[1]))
                else:
                    if random.random() < initialPro:
                        add_bpmCandidate = True
                if add_bpmCandidate:
                    for hint,feature in hint_feature[1:]:
                        del feature['action_mask']
                        self.bpm.add_balances.remote(query_id, hint, feature, sql)
            else:
                planning_time =  time.time() - planning_start 
                latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote('',sql,query_id))
            if not Explore:
                if not isTest:
                    bestfeature, bestexec = ray.get(self.bpm.get_evaluator_esbest.remote(query_id))
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    # self.anaManger.recordBestPlanSteps(query_id,int(self.genConfig.maxsteps * optimal_feature['steps'][0]), latency_timeout[0])# TO_Delete
                    if Advbybest >= self.genConfig.splitpoint[-1]:
                        self.bpm.update_evaluator_esbest.remote(query_id, deepcopy(optimal_feature),latency_timeout[0])
                    self.bpm.update_RL_esbest.remote(query_id,deepcopy(optimal_feature), latency_timeout[0])
                planning_time = planning_time * 1000
                self.anaManger.recordeval(query_id,latency_timeout[0], planning_time)
                if isTest:
                    key_out = '_'.join(['test', str(valIter), query_id])
                else:
                    key_out = '_'.join(['train',str(valIter),query_id])
                value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
                self.anaManger.recordRuning(key_out,value_out)
            if oneloop:
                break 
        estotal, best_steps = ray.get(self.bpm.get_evaluator_best.remote())
        if isTest:
            self.anaManger.recordTime(f'{valIter}_iter_time')
        return estotal, best_steps
    
    def Run(self, totalIter = 300, valFreq = 5):
        ray.get(self.bpm.update_schedule.remote(False))
        runner_ref = self.learner.Runing.remote()
        startVal = 10
        for trainIter in range(totalIter):
            latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
            ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
            result = self.planner.train()
            print(f'TrainIter {trainIter}')
            if trainIter <= 4 or (trainIter + 1) % valFreq == 0:
                ray.get(self.bpm.update_schedule.remote(True))
                accumulatedPlans = ray.get(runner_ref)
                self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', accumulatedPlans, trainIter + 1)
                if (trainIter + 1) % valFreq == 0 and trainIter >= startVal:
                    estotal, best_steps = self.Validate(trainIter + 1)
                    self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, trainIter + 1)
                    self.Validate(trainIter + 1, isTest = True)
                    self.anaManger.recordMetric(trainIter + 1)
                    self.anaManger.writeout()
                # checkpoint = self.planner.save(checkpoint_dir = self.genConfig.agent_checkpoint)
                ray.get(self.bpm.update_schedule.remote(False))
                runner_ref = self.learner.Runing.remote()
    def Sim(self, totalIter = 300, valFreq = 10):
        startVal = 10
        for trainIter in range(totalIter):
            result = self.planner.train()
            print(f'TrainIter {trainIter}')
            if (trainIter + 1) % valFreq == 0 and trainIter >= startVal:
                estotal, best_steps = self.Validate(trainIter + 1)
                self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, trainIter + 1)
                self.Validate(trainIter + 1, isTest = True)
                self.anaManger.recordMetric(trainIter + 1)
                self.anaManger.writeout()

    def Close(self):
        self.anaManger.close()  
        ray.shutdown()
    

if __name__ == "__main__":
    SupportWorkload = ['JOBRand', 'TPCDS', 'STACK','JOBEXT']

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
    parser.add_argument("--Maxsamples", type=int, default = 5, # JOBRand 5 TPCDS&STACK 25 or more,
                        help="Maximum number of samples in a single iteration (in best plan manager)")
    parser.add_argument("--TotalIter", type=int, default = 300,
                        help="The total number of iterations for which the planner is trained")
    parser.add_argument("--ValidateFreq", type=int, default = 5,
                        help="The frequency of validation")
    parser.add_argument("--Agents", type=int, default = 1,
                        help="The Num of Agents.")
    parser.add_argument("--PenaltyCoeff", type=int, default = 2,
                        help="The coefficient of penalty.")
    parser.add_argument("--SampleStrategy", type=str, default = 'hybrid',
                        help="The strategy of sampling.")
    parser.add_argument("--NotUpdateEvaluator", action='store_true',
                        help="Whether update evaluator or not.")
    parser.add_argument("--OffLeftDeep", action='store_true',
                        help="Remove the left-deep restriction, but it may not work on the version of pg_hint_plan from https://github.com/yxfish13/PostgreSQL12.1_hint.")
    parser.add_argument("--EvaluatorPath", type=str, default = None,
                        help="The model path of Evaluator.")
    args = parser.parse_args()

    config = Config()

    arg_to_config = {
        'Workload': 'mode',
        'Seed': 'seed',
        'ExpName': 'expname',
        'Database': 'database',
        'Maxsteps': 'maxsteps',
        'Maxsamples': 'maxsamples',
        'Agents': ('num_agents', 'num_policies'),
        'PenaltyCoeff': 'penalty_coeff',
        'SampleStrategy': 'sample_strategy'
    }

    for arg, config_attr in arg_to_config.items():
        arg_value = getattr(args, arg, None)
        # print(arg,arg_value)
        if arg_value is not None:
            if isinstance(config_attr, tuple):
                for attr in config_attr:
                    setattr(config, attr, arg_value)
            else:
                setattr(config, config_attr, arg_value)
    if args.NotUpdateEvaluator:
        config.update_evaluator = False
    if args.OffLeftDeep:
        config.left_deep_restriction = False
    if not args.ExpName:
        config.expname = f"{config.seed}_{args.Maxsteps}"
    config.ConfirmPath()
    if args.EvaluatorPath:
        config.evaluator_path = args.EvaluatorPath
    foss = FOSS(config)
    print("Running Baseline......")
    foss.RunBaseline()
    print("Initialing Planner......")
    foss.BuildPlanner(train_batch_size = 3000, workersNum = 5)
    if config.update_evaluator:
        initialPro = 0.25
        print("Initialing Evaluator........")
        foss.InitEvaluator(initialPro = initialPro)
        time.sleep(10)  # wait for evaluator Training
        print("Training FOSS.........")
        foss.Run(totalIter = args.TotalIter, valFreq = args.ValidateFreq)
    else:
        foss.Sim(totalIter = args.TotalIter, valFreq = args.ValidateFreq)
    foss.Close()
