
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from FOSSEnvTrain import FOSSEnvTrain
from FOSSEnvExplore import FOSSEnvExplore
from FOSSEnvTest import FOSSEnvTest
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from model import CustomModel
from querymanage import QueryManager
from config import Config
from datacollector import DataColletor
from pairestimator import PairTrainer,MyDataset
from torch.utils.tensorboard import SummaryWriter
from planhelper import PlanHelper
from bestplanmanage import BestPlanManager
# from encoding import Encoding
from copy import deepcopy
import numpy as np
import time
import json
import pandas as pd
import random
# import torch
# import os
config = Config()
# JOBEXT entropy_coeff:0.2 lr = 5e-5 
# JOB  
# JOBRand inner 121
# TPCDS Convergence is too fast 
# def set_seed(seed=3407):  # torch.manual_seed(3407) is all u need
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
        
def gen_policy(i):
    if i % 2 == 0:
        gamma_ = 0.99
        lr = 5e-5
    else:
        gamma_ = 0.9
        lr = 5e-5
    config = PPOConfig.overrides(
        model={
            "custom_model": 'gen_model',
            "vf_share_layers": True,
            "fcnet_hiddens": [384, 384, 384],
            "fcnet_activation": "tanh",
        },
        # clip_param = 0.2,
        entropy_coeff = 0.2,
        # num_sgd_iter = 20,
        # sgd_minibatch_size = 256,
        # vf_loss_coeff = 0.5,
        # vf_clip_param = 15,
        kl_coeff = 1.5,
        lambda_ = 0.9,
        gamma = gamma_, # 0.9
        # shuffle_sequences = True,
        lr = lr)
    return PolicySpec(config=config)
# def gen_policy(i):
#     if i % 2 == 0:
#         lambda_ = 0.7
#         gamma_ = 0.7
#     else:
#         lambda_ = 0.9
#         gamma_ = 0.9
#     config = PPOConfig.overrides(
#         model={
#             "custom_model": 'gen_model',
#             "vf_share_layers": True,
#             "fcnet_hiddens": [256, 256, 256],
#             "fcnet_activation": "tanh",
#         },
#         # clip_param = 0.2,
#         entropy_coeff_schedule = [[0, 0.15], [5e4, 0.1],[1e5, 0.05],[2e5, 0.01]],
#         # vf_loss_coeff = 0.5,
#         sgd_minibatch_size = 256,
#         num_sgd_iter = 30,
#         lambda_ = lambda_,
#         gamma = gamma_, # 0.9
#         shuffle_sequences = True,
#         lr_schedule = [[0, 5e-5], [5e4, 1e-4],[1e5, 2e-4],[2e5, 3e-4]])
#     return PolicySpec(config=config)
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = "policy_{}".format(agent_id)
    return pol_id
if __name__ == "__main__":
    # set_seed()
    ray.init()
    
    writer = SummaryWriter(log_dir=f'/home/zhongkai/zk_pywork/FOSS/runstate/{config.mode}_{config.loop}')
    ModelCatalog.register_custom_model("gen_model", CustomModel)
    
    out_file = open(config.outfile_path,'w')
    experienceopool = open(config.ExperiencePool,'w')

    # *********Run*********
    datacollector = DataColletor()
    planhelper = PlanHelper()
    querymanager = QueryManager(writer = writer)
    pairtrainer = PairTrainer()
    bpm = BestPlanManager.remote()
    pairtrainer.save_model(config.model_path)
    # =====Run Baseline========
    Baseline_train = 0
    Baseline_test = 0
    Baseline = {}
    print("RUNNING BASELINE......")
    
    while True:
        sql, queryno, query_id, oneloop = querymanager.get2validate()
        feature_dict,hint,_,cost_plan_json = planhelper.get_feature('',sql,False,query_id = query_id)
        feature_dict['steps'] = np.array([0])
        cost_plan_json['steps'] = 0
        latency_timeout,iscollect,_ = planhelper.getLatency('',sql,query_id)
        key_out = '_'.join(['base',query_id])
        value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(cost_plan_json['Planning Time'] * 1000)])
        out_file.write(json.dumps([key_out,value_out])+"\n")
        out_file.flush()
        querymanager.recordeval(query_id,latency_timeout[0])
        bpm.update_AAM_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
        bpm.update_RL_esbest.remote(query_id,deepcopy(feature_dict),latency_timeout[0])
        datacollector.collect_data_online(query_id,feature_dict,latency_timeout[0],latency_timeout[1])
        # datacollector.collect_data_offline(query_id,cost_plan_json,latency_timeout[0],latency_timeout[1])
        print('\t Train Set Query:{} Latency:{:.4f}'.format(query_id,latency_timeout[0]))
        Baseline[query_id] = latency_timeout[0]
        Baseline_train += latency_timeout[0]
        if oneloop:
            break
    print('Baseline_train:{:.4f}'.format(Baseline_train))
    
    
    while True:
        sql,_,query_id,oneloop = querymanager.get2eval()
        _, _,_, plan_json = planhelper.get_feature('',sql,False,query_id = query_id)
        latency_timeout,iscollect,_ = planhelper.getLatency('',sql,query_id)
        key_out = '_'.join(['base',query_id])
        value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(plan_json['Planning Time'] * 1000)])
        out_file.write(json.dumps([key_out,value_out])+"\n")
        out_file.flush()
        print('\t Test Set Query:{} Latency:{:.4f}'.format(query_id,latency_timeout[0]))
        querymanager.recordeval(query_id,latency_timeout[0])
        Baseline_test += latency_timeout[0]
        Baseline[query_id] = latency_timeout[0]
        if oneloop:
            break
    print('Baseline_test:{:.4f}'.format(Baseline_test))
    writer.add_scalar('Train Set Best Plan WRL',1.0 , 0)
    # RUN NOW

       #  =========Train config===========
    train_base_config={'bestplanmanager':bpm}
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(config.num_policies)}
    train_config = (
        PPOConfig()
        .environment(FOSSEnvTrain,env_config=train_base_config,disable_env_checking=True)
        .training(train_batch_size = 2700) #2100
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus = 1,num_gpus_per_learner_worker = 1)\
        .rollouts(num_rollout_workers = 10,num_envs_per_worker = 1)#,sample_async=True)
    )
    train_config['seed'] = config.seed
    train_algo = train_config.build()
        # =========Env Construct===========
    env_config = {'planhelper':planhelper}
    explore_env = FOSSEnvExplore(env_config)
    eval_env = FOSSEnvTest(env_config)
    
    # =======Run Param===========
    querynum_perepoch = config.querynum_perepoch
    iterations = 35 #math.floor(totalnum * 1.0 / querynum_perepoch) * 6#4
    test_iter = 0
    cv_iter = 0
    interesult = pd.DataFrame(columns=['sql','query_id','baselinetime','hint','policy_id','steps','exectime','feature_dict','plan_json','timeout','min_step'])
    total_time = 0
    
    # decay = 0.05
    for iter in range(iterations):
    #===================Explore=====================
        start_time = time.time()
        totalfindingtime = 0
        actions = {}
        if iter >= config.startiter:
            querynum_perepoch = 10
        for k in range(querynum_perepoch):
            sql, queryno, query_id, base_train_feature, oneloop = querymanager.get2explore()
            sqlinfo = {'sql':sql,'feature':base_train_feature,'query_id':query_id}
            feature_dict, _, cost_plan_json = base_train_feature
            obs, info = explore_env.reset(options=sqlinfo)
            baselinetime = Baseline[query_id]
            datacollector.collect_test_data(query_id,feature_dict,baselinetime,False)
            done = False
            steps = 1
            while not done:
                actions.clear()
                for i in range(config.num_policies):
                    actions[i] = train_algo.compute_single_action(obs[i],policy_id = 'policy_{}'.format(i),explore = False)
                obs, reward, terminated, _, info_all = explore_env.step(actions)
                for i in range(config.num_agents):
                    feature_dict = deepcopy(obs[i])
                    del feature_dict['action_mask']
                    interesult.loc[len(interesult.index)] = [sql,query_id,baselinetime,info_all['exechint'][i],
                                                            i,steps,config.max_time_out,feature_dict,
                                                            info_all['cost_plan_json'][i],False,info_all['min_step'][i]]
                steps += 1
                if terminated['__all__'] == True:
                    done = True
            # queryImportance[query_id] = 1
            if oneloop:
                break
        # if iter < config.startiter:
        #     bpm.updatequeryImportance.remote(queryImportance,False)
        # else:
        #     bpm.updatequeryImportance.remote(queryImportance,True)
        randomidx = list(interesult.index)
        random.shuffle(randomidx)
        for i in randomidx:
            sql = interesult.loc[i,'sql']
            hint = interesult.loc[i,'hint']
            query_id = interesult.loc[i,'query_id']
            baselinetime = interesult.loc[i,'baselinetime']
            timeout = config.timeoutcoeff * baselinetime
            latency_timeout,iscollect,findingtime = planhelper.getLatency(hint, sql, query_id,timeout = timeout)
            if findingtime:
                totalfindingtime += (latency_timeout[0] / 1000 - findingtime)
            if not latency_timeout[1] and latency_timeout[0] > timeout - 1e-6: # 消除验证集数据影响
                latency_timeout[0] = timeout
                latency_timeout[1] = True
            interesult.loc[i,'exectime'] = latency_timeout[0]
            interesult.loc[i,'timeout'] = latency_timeout[1]
            if iscollect:
                datacollector.collect_data_online(query_id,deepcopy(interesult.loc[i,'feature_dict']),latency_timeout[0],latency_timeout[1])
                # datacollector.collect_data_offline(query_id,interesult.loc[i,'plan_json'],latency_timeout[0],latency_timeout[1])
                # Adv = (baselinetime - latency_timeout[0]) / baselinetime * 100   
                # if Adv >= 90:
                #     datacollector.collect_data_online(query_id,interesult.loc[i,'feature_dict'],latency_timeout[0],latency_timeout[1])
                #     datacollector.collect_data_offline(query_id,interesult.loc[i,'plan_json'],latency_timeout[0],latency_timeout[1])
        normalidx = list(interesult.index)
        for i in normalidx:
            query_id = interesult.loc[i,'query_id']
            hint = interesult.loc[i,'hint']
            policyid = interesult.loc[i,'policy_id']
            steps = interesult.loc[i,'steps']
            exectime = interesult.loc[i,'exectime']
            baselinetime = interesult.loc[i,'baselinetime']
            timeout = interesult.loc[i,'timeout']
            Adv = (baselinetime - exectime) / baselinetime * 100   
            print('Query_id:{} Steps:{} Adv:{:.4f}% \nHintDict:{}'.format(query_id,steps,Adv,hint))
            datacollector.collect_test_data(query_id,deepcopy(interesult.loc[i,'feature_dict']),exectime,timeout)
            experienceopool.write(json.dumps([query_id,'|'.join([hint,str(policyid),str(steps),'{:.4f}'.format(exectime)])])+"\n")
            experienceopool.flush()
            
        min_exectime_idx = interesult.groupby('query_id')['exectime'].idxmin()
        min_result = interesult.loc[min_exectime_idx, ['query_id', 'exectime', 'baselinetime', 'min_step']]
        for idx in list(min_exectime_idx):
            if (min_result.loc[idx,'baselinetime'] - min_result.loc[idx,'exectime']) / min_result.loc[idx,'baselinetime'] < config.alpha:
                min_result.loc[idx,'min_step'] = 0
                min_result.loc[idx,'exectime'] = min_result.loc[idx, 'baselinetime']
            query_id = min_result.loc[idx,'query_id']
            exectime = min_result.loc[idx,'exectime']
            #steps    = min_result.loc[idx,'steps']
            min_step = min_result.loc[idx,'min_step']
            bestfeature, bestexec = ray.get(bpm.get_AAM_esbest.remote(query_id))
            if (bestexec - exectime) / bestexec >= config.alpha:
                bpm.update_AAM_esbest.remote(query_id,deepcopy(interesult.loc[idx,'feature_dict']),exectime)
            key_out = '{}_beststep_{}'.format(iter,query_id)
            value_out = '{}|{:.4f}'.format(min_step, exectime)
            out_file.write(json.dumps([key_out,value_out])+"\n")
            out_file.flush()
        interesult.drop(interesult.index, inplace=True)
        estotal = ray.get(bpm.get_AAM_best_totaltime.remote())
        writer.add_scalar('Train Set Best Plan WRL', estotal / Baseline_train, iter + 1)
        explore_time = time.time()
        total_time += (explore_time - start_time)
        key_out = '{}_exploretime'.format(iter)
        value_out = '{:.4f}'.format(explore_time - start_time)
        out_file.write(json.dumps([key_out,value_out])+"\n")
        out_file.flush()

        #===================Asmetric Advantage Time=====================
        # Process Data 
        inputs,labels = datacollector.process_online()
        test_inputs,test_labels = datacollector.get_test_data()
        traindataset = MyDataset(inputs,labels)
        testdataset = MyDataset(test_inputs,test_labels)
        if testdataset.__len__() > 0:
            print('Before Train')
            pairtrainer.test_dataset(testdataset)
        datacollector.clear_test_data()
        # datacollector.wirte_offline(config.offline_data_path)
        #=========Evaluate Current Accuracy========
        dataset_len = traindataset.__len__()
        print('train data length:',dataset_len)
        pairtrainer.retrainmodel()
        pairtrainer.train_dataset(traindataset,mybatch_size = 128, epochs = 10)
        pairtrainer.save_model(config.model_path)
        if testdataset.__len__() > 0:
            print('After Train')
            pairtrainer.test_dataset(testdataset)
        
        AAMtime = time.time()
        total_time += (AAMtime - explore_time)
        key_out = '{}_AAMtime'.format(iter)
        value_out = '{:.4f}'.format(AAMtime - explore_time)
        out_file.write(json.dumps([key_out,value_out])+"\n")
        out_file.flush()
        
        #=================Train================
         # min(5 + iter , 10) #15
        # print('outer_iter:%d  total_inneriters:%d'%(iter,inner_iterations))
        train_time = 0
        candidatebest_counts = 0
        inner_iterations = min(5 + iter , 10) #15
        for inner_iter in range(inner_iterations):
            latencybuffer = planhelper.getPGLatencyBuffer()
            bpm.update_latencyBuffer.remote(latencybuffer)
            train_iter = time.time()
            result = train_algo.train()
            plan2exploreNO = 0
            if iter >= config.startiter: 
                candidatebest = ray.get(bpm.get_candidatebest.remote())
                for query_id,hints in candidatebest.items():
                    if hints:
                        sql = querymanager.getsqlbyqueryid(query_id)
                        bestfeature,bestexec = ray.get(bpm.get_AAM_esbest.remote(query_id))
                        for hint,steps in hints:
                            latency_timeout,iscollect,findingtime = planhelper.getLatency(hint, sql, query_id,timeout = config.timeoutcoeff * Baseline[query_id])
                            if iscollect:
                                if findingtime:
                                    totalfindingtime += (latency_timeout[0] / 1000 - findingtime)
                                plan2exploreNO += 1
                                feature_dict,_,_,cost_plan_json = planhelper.get_feature(hint,sql,False,query_id)
                                feature_dict['steps'] = np.array([steps])
                                if latency_timeout[0] >= config.timeoutcoeff * Baseline[query_id]:  #消除验证集影响
                                    latency_timeout[0] = config.timeoutcoeff * Baseline[query_id]
                                    latency_timeout[1] = True
                                # cost_plan_json['steps'] = steps
                                datacollector.collect_data_online(query_id,deepcopy(feature_dict),latency_timeout[0],latency_timeout[1])
                                # datacollector.collect_data_offline(query_id,cost_plan_json,latency_timeout[0],latency_timeout[1])
                            Advbybest = (bestexec - latency_timeout[0]) / bestexec
                            print(f'Query id:{query_id}, CurrBest:{bestexec}, EstiBest:{latency_timeout[0]}, Advbybest:{Advbybest * 100}%')
                            if Advbybest >= config.alpha:
                                bpm.update_AAM_esbest.remote(query_id, deepcopy(feature_dict),latency_timeout[0]) 
                                bestexec = latency_timeout[0]
                                key_out = '{}_beststep_{}'.format(iter,query_id)
                                value_out = '{}|{:.4f}'.format(steps, latency_timeout[0])
                                out_file.write(json.dumps([key_out,value_out])+"\n")
                                out_file.flush()
                print(f'Num of Candidate Best Plan:{plan2exploreNO}')
                writer.add_scalar('Num of Candidate Best Plan', plan2exploreNO, cv_iter)     
                cv_iter += 1       
                candidatebest_counts += plan2exploreNO
                if candidatebest_counts >= 20:
                    inputs,labels = datacollector.process_online()
                    dataset = MyDataset(inputs,labels)
                    #  datacollector.wirte_offline(config.offline_data_path)
                    dataset_len = dataset.__len__()
                    print('train data length:',dataset_len)
                    pairtrainer.retrainmodel()
                    pairtrainer.train_dataset(dataset,mybatch_size = 128, epochs = 10)
                    pairtrainer.save_model(config.model_path)
                    candidatebest_counts = 0
            bpm.clear_candidatebest.remote()
            
            train_iter = time.time() - train_iter + totalfindingtime
            total_time += train_iter
            train_time += train_iter
            #===============Validate and Test==================
            if iter >= config.startiter and test_iter % 5 == 0:
                while True:
                    sql, queryno, query_id, oneloop = querymanager.get2validate()
                    sqlinfo = {'sql':sql,'queryno':queryno,'query_id':query_id}
                    planstart = time.time()
                    obs, info = eval_env.reset(options=sqlinfo)
                    hint_feature = [(info['hint'],deepcopy(obs[0]))]
                    steps = 1
                    if info['useFOSS']:
                        done = False
                        while not done:
                            actions.clear()
                            for i in range(config.num_policies):
                                actions[i] = train_algo.compute_single_action(obs[i],policy_id = 'policy_{}'.format(i), explore = False)
                            obs, reward, terminated, _, info_all = eval_env.step(actions)
                            for k in info_all:
                                experienceopool.write(json.dumps([query_id,'|'.join([info_all[k]['hint'],str(k),str(steps)])])+"\n")
                                experienceopool.flush()
                                hint_feature.append((info_all[k]['hint'],deepcopy(obs[k])))
                            steps += 1
                            if terminated['__all__'] == True:
                                done = True
                        optimal_hint,optimal_feature = pairtrainer.predict_epi(hint_feature)
                        plantime = time.time() - planstart 
                        latency_timeout,iscollect,_ = planhelper.getLatency(optimal_hint, sql, query_id,timeout = Baseline[query_id])
                        # if latency_timeout[1] and latency_timeout[0] + 1 < config.max_time_out:
                        #     latency_timeout,_ = planhelper.getLatencyNoCache(optimal_hint, sql, query_id)
                    else:
                        plantime =  time.time() - planstart 
                        latency_timeout,_,_ = planhelper.getLatency('',sql,query_id)
                    bestfeature,bestexec = ray.get(bpm.get_AAM_esbest.remote(query_id))
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    if Advbybest >= config.alpha:
                        bpm.update_AAM_esbest.remote(query_id, deepcopy(optimal_feature),latency_timeout[0])
                    bpm.update_RL_esbest.remote(query_id,deepcopy(optimal_feature),latency_timeout[0])
                    querymanager.recordeval(query_id,latency_timeout[0])
                    key_out = '_'.join(['train',str(test_iter),query_id])
                    value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(plantime * 1000)])
                    out_file.write(json.dumps([key_out,value_out])+"\n")
                    out_file.flush()
                    if oneloop:
                        break 
                while True:
                    sql, queryno, query_id, oneloop = querymanager.get2eval()
                    sqlinfo = {'sql':sql,'queryno':queryno,'query_id':query_id}
                    planstart = time.time()
                    obs, info = eval_env.reset(options = sqlinfo)
                    hint_feature = [(info['hint'],deepcopy(obs[0]))]
                    steps = 1
                    if info['useFOSS']:
                        done = False
                        while not done:
                            actions.clear()
                            for i in range(config.num_policies):
                                actions[i] = train_algo.compute_single_action(obs[i],policy_id = 'policy_{}'.format(i), explore = False)
                            obs, reward, terminated, _, info_all = eval_env.step(actions)
                            for k in info_all:
                                experienceopool.write(json.dumps([query_id,'|'.join([info_all[k]['hint'],str(k),str(steps)])])+"\n")
                                experienceopool.flush()
                                hint_feature.append((info_all[k]['hint'],deepcopy(obs[k])))
                            steps += 1
                            if terminated['__all__'] == True:
                                done = True
                        optimal_hint,_ = pairtrainer.predict_epi(hint_feature)
                        plantime = time.time() - planstart 
                        latency_timeout,_,_ = planhelper.getLatency(optimal_hint, sql, query_id)
                        if latency_timeout[1] and latency_timeout[0] + 1 < config.max_time_out:
                            latency_timeout,_ = planhelper.getLatencyNoCache(optimal_hint, sql, query_id)
                    else:
                        plantime =  time.time() - planstart 
                        latency_timeout,_,_ = planhelper.getLatency('',sql,query_id)
                        
                    querymanager.recordeval(query_id,latency_timeout[0])
                    key_out = '_'.join(['test',str(test_iter),query_id])
                    value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(plantime * 1000)])
                    out_file.write(json.dumps([key_out,value_out])+"\n")
                    out_file.flush()
                    if oneloop:
                        break
                querymanager.recordwrl()
                querymanager.recordgmrl()
                querymanager.writeout()
            test_iter += 1
        key_out = '{}_traintime'.format(iter)
        value_out = '{:.4f}'.format(train_time)
        out_file.write(json.dumps([key_out,value_out])+"\n")
        out_file.flush()
        checkpoint = train_algo.save(checkpoint_dir = config.agent_checkpoint)
    key_out = 'totaltime'
    value_out = '{:.4f}'.format(total_time)
    out_file.write(json.dumps([key_out,value_out])+"\n")
    out_file.flush()
    out_file.close()
    experienceopool.close()   
    ray.shutdown()