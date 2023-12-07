from ray.rllib.env.multi_agent_env import MultiAgentEnv
from FOSSEnvTrainBase import FOSSEnvTrainBase
import os
from config import Config
from pairestimator import PairTrainer
from querymanage import QueryManager
from planhelper import PlanHelper
import ray
config = Config()
class FOSSEnvTrain(MultiAgentEnv):

    def __init__(self,out_config):
        super().__init__()
        self.pairtrainer = PairTrainer(device='cpu')
        self.planhelper = PlanHelper()
        self.querymanger = QueryManager()
        self.bpm = out_config['bestplanmanager']
        if os.path.exists(config.model_path):
            self.pairtrainer.load_model(config.model_path)
            AAM_esbest_feature = ray.get(self.bpm.get_AAM_esbest.remote())
            # queryImportance = ray.get(self.bpm.get_weightsByRLesbest.remote())
            # self.querymanger.updateBuffer(queryImportance)
            latencyBuffer = ray.get(self.bpm.get_latencyBuffer.remote())
            self.planhelper.updatePGLatencyBuffer(latencyBuffer)
            self.querymanger.update_AAM_esbest(AAM_esbest_feature)
            RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
            self.querymanger.update_RL_esbest(RL_esbest_feature)
            self.AAMversion = os.path.getmtime(config.model_path)
        else:
            # to raise error
            self.AAMversion = None
            # raise Exception('Model path is None!')
        tableNum = self.planhelper.gettablenum()
        env_config = {'pairtrainer':self.pairtrainer,'querymanger':self.querymanger,
                      'bestplanmanager':self.bpm,'planhelper':self.planhelper,'tableNum':tableNum}
        self.agents = [FOSSEnvTrainBase(env_config) for _ in range(config.num_agents)]
        self._agent_ids = set(range(config.num_agents))
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.update_times = 0
        self.resetted = False
    def reset(self, *, seed=None, options=None):
        if not self.resetted:
            self.resetted = True
            return {},{}
        super().reset(seed=seed)
        # if self.epi % 100 == 0:
        #     self.planhelper.encoding.save_to_file(config.encoding_path)
        # bonusPlus = 0
        if self.AAMversion:
            now_version = os.path.getmtime(config.model_path)
            if now_version != self.AAMversion:
                self.update_times += 1
                if self.update_times > config.startiter:
                # if ray.get(self.bpm.get_startEval.remote()):
                    # bonusPlus = min(self.update_times // 3, 6)
                    queryImportance = ray.get(self.bpm.update_weightsByRLesbest.remote())
                    # queryImportance = ray.get(self.bpm.getqueryImportance.remote())
                    self.querymanger.updateBuffer(queryImportance)
                latencyBuffer = ray.get(self.bpm.get_latencyBuffer.remote())
                self.planhelper.updatePGLatencyBuffer(latencyBuffer)
                AAM_esbest_feature = ray.get(self.bpm.get_AAM_esbest.remote())
                self.querymanger.update_AAM_esbest(AAM_esbest_feature)
                RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
                self.querymanger.update_RL_esbest(RL_esbest_feature)
                self.pairtrainer.load_model(config.model_path)
                print('Load New AAM|Now VersionTime:{}'.format(now_version))
                self.AAMversion = now_version
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        reset_results = [a.reset() for a in self.agents]
        # self.epi += 1
        return (
            {i: oi[0] for i, oi in enumerate(reset_results)},
            {i: oi[1] for i, oi in enumerate(reset_results)},
        )

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info