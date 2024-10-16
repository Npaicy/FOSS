import torch
import numpy as np
from collections import deque,defaultdict
from database_util import *
from pghelper import PGHelper
from encoding import Encoding
import os, re
import ray
import copy
from util import swap_dict_items
class PlanHelper:

    def __init__(self, globalConfig):
        self.config = globalConfig
        self.pgrunner = PGHelper(self.config)
        self.encoding = Encoding(self.config)
        if os.path.exists(self.config.encoding_path):
            self.encoding.load_from_file(self.config.encoding_path)
        else:
            print(' Init Encoding......')
            column_data_properties = self.pgrunner.get_column_data_properties()
            self.encoding.loadcdp(column_data_properties)
        self.alias2table = {}
    def getPGLatencyBuffer(self):
        return self.pgrunner.latencyBuffer
    
    def updatePGLatencyBuffer(self,latencyBuffer):
        self.pgrunner.latencyBuffer = latencyBuffer

    def get_table_num(self):
        return self.pgrunner.get_table_num()
    
    def getLatency(self, hint, sql, query_id, timeout = None, hintstyle = 'FOSS'):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.pgrunner.getLatency(hint, sql, query_id, timeout, hintstyle)
    
    def getMinLatency(self):
        return self.pgrunner.get_minLatency()
    
    def tryGetLatency(self, hint, query_id):
        return self.pgrunner.tryGetLatency(hint, query_id)
    
    def getLatencyNoCache(self,hint,sql,query_id,timeout = None):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.pgrunner.getLatencyNoCache(hint, sql, query_id, timeout)
    
    def get_feature(self, exechint, sql, toextract,query_id = None, plan_json = None, hintstyle = 'FOSS'):
        sql = sql.lower()
        if plan_json == None:
            plan_json = self.pgrunner.getCostPlanJson(exechint, sql, hintstyle, query_id)
        self.extractAliasFromSql(sql)
        hintdict = {'scan table':{},'join operator':{}}
        ori_dict = self.traversePlan(plan_json['Plan'], hintdict)
        plan_feature = self.pre_collate(ori_dict, max_node=self.config.maxnode)
        left_deep = None
        if toextract:
            hintdict,left_deep = self.processhint(hintdict)
        return plan_feature, hintdict, left_deep, plan_json
    
    def extractAliasFromSql(self, sql):
        try:
            fromclause = re.split(r'from[\n \t]', sql, flags=re.IGNORECASE)[1]
            fromclause = re.split(r'where[\n \t]', fromclause, flags=re.IGNORECASE)[0]
            fromclause = [oneclause.strip('\n ') for oneclause in fromclause.split(',')]
        except:
            print(sql)
            raise ValueError('SQL!')
        for fc in fromclause:
            fc = fc.replace('\t','')
            fc = fc.replace('\n','')
            fc = fc.strip(' ')
            if ' as ' in fc:
                fcs = fc.split(' as ')
                self.alias2table[fcs[1]] = fcs[0]
            else:
                fcs = fc.split(' ')
                if len(fcs) == 2:
                    fcs[0] = fcs[0].strip(' ')
                    fcs[1] = fcs[1].strip(' ')
                    self.alias2table[fcs[1]] = fcs[0]
                else:
                    fcs[0] = fcs[0].strip(' ')
                    self.alias2table[fcs[0]] = fcs[0]

    def get_hintNum(self):
        hintNum = {}
        for queryid in self.pgrunner.latencyBuffer:
            hintNum[queryid] = len(self.pgrunner.latencyBuffer[queryid])
        return hintNum
    
    def processhint(self, hint):
        left_deep = True
        ICP = {'join order':[],'join operator':[],'structure':[]}
        if len(hint['join operator']) <= 1:
            return ICP, None
        hint['join operator'] = dict(reversed(list(hint['join operator'].items())))
        encodOfJoin = list(hint['join operator'].keys())
        for k in range(0,len(encodOfJoin) - 1):
            if len(encodOfJoin[k]) == len(encodOfJoin[k + 1]) and encodOfJoin[k][-1] > encodOfJoin[k + 1][-1]:
                hint['join operator'] = swap_dict_items(hint['join operator'], encodOfJoin[k], encodOfJoin[k + 1])
        padLen = max([len(k) for k in hint['scan table'].keys()])
        sortbyencod = []
        for k in hint['scan table'].keys():
            sum_e = 0
            for i_e, e in enumerate(k[-1::-1]):
                sum_e += eval(e) * (2 ** (padLen - len(k) + i_e))
            sortbyencod.append((sum_e, k, hint['scan table'][k]))
        sortbyencod.sort(key = lambda x: x[0])
        encod = [x[1] for x in sortbyencod]
        jointable = [x[2] for x in sortbyencod]
        ICP['join order'] = [table[0] for _, _, table in sortbyencod]
        for joinEncod in hint['join operator']:
            prefixLen = len(joinEncod)
            JoinE = []
            JoinI = []
            for i_, scanEncod in enumerate(encod):
                if scanEncod[0:prefixLen] == joinEncod:
                    JoinI.append(i_)
                    JoinE.append(scanEncod)
            if len(JoinI) != 2 or (JoinI[1] - JoinI[0]) != 1:
                raise KeyError('Parse Error')
            if JoinI[0] != 0:
                left_deep = False
            encod[JoinI[0]] = joinEncod
            del  encod[JoinI[1]]
            jointable[JoinI[0]] = jointable[JoinI[0]] + jointable[JoinI[1]]
            del jointable[JoinI[1]]
            ICP['structure'].append(JoinI[0])
            ICP['join operator'].append(self.config.operator_pg2hint[hint['join operator'][joinEncod]])
        return ICP,left_deep

    def pre_collate(self, the_dict, max_node, rel_pos_max=20, alpha=0):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        pc_dict = the_dict['pc_dict']
        assert len(pc_dict) != 0
        distance_matrix = bfs(N ,pc_dict, rel_pos_max)
        attn_bias[1:, 1:] = torch.from_numpy(distance_matrix).float() * alpha + (1 - torch.from_numpy(distance_matrix).float())
        attn_bias[0, :] = 1
        attn_bias[:, 0] = alpha
        attn_bias[0, 0] = 1
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1, alpha)
        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        return {
            'x': x.numpy(),
            'attn_bias': attn_bias.numpy(),
            'heights': heights.numpy(),
        }
    
    def traverseNode(self, plannode,  pos = None, parentAlias = None):
        nodeType = plannode['Node Type']
        typeId = self.encoding.encode_type(nodeType)
        table = 'NA'
        table_id = 0
        alias = None
        if nodeType in SCANTYPE:
            try:
                table = plannode['Relation Name']
                table_id = self.encoding.encode_table(plannode['Relation Name'])
            except:
                raise ValueError('Relation Name Parse Error')
            try:
                if 'Alias' in plannode:
                    alias = plannode['Alias']
                    if alias not in self.alias2table.keys():
                        self.alias2table[plannode['Alias']] = table
            except:
                raise ValueError('Alias Parse Error')
        elif nodeType == 'Bitmap Index Scan' and alias == None:
            alias = parentAlias
        join, filters, db_est = processCond(plannode, alias, self.alias2table)
        joinids = self.encoding.encode_join(join)
        filters_encoded = self.encoding.encode_filters(filters, alias, self.alias2table)
        node = TreeNode(nodeType, table, table_id, typeId, filters, joinids, filters_encoded, db_est, pos)
        node.pos = pos
        node.feature = node2feature(node)
        return node
    
    def traversePlan(self, plan, hint): 
        # pos:{3:'root', 0:'left', 1:'right', 2:'internal-no-brother'}
        adj_list = []  
        features = []
        heights = []
        root = self.traverseNode(plan, pos = 3)
        NodeList = deque()
        NodeList.append((root, plan, '0', 0))
        next_id = 1
        while NodeList:
            parentNode, parentPlan, parentEncod, idx = NodeList.popleft()
            features.append(parentNode.feature)
            heights.append(len(parentEncod))
            if parentPlan['Node Type'] in JOINTYPE:
                hint['join operator'][parentEncod] = parentPlan['Node Type']
            elif parentPlan['Node Type'] in SCANTYPE:
                hint['scan table'][parentEncod] = [parentPlan['Alias']]
                parentNode.alias = parentPlan['Alias']  # Bitmap Index Scan
            if 'Plans' in parentPlan:
                subPlanNum = len(parentPlan['Plans'])
                if subPlanNum == 1:
                    subplan = parentPlan['Plans'][0]
                    node = self.traverseNode(subplan, pos = 2, parentAlias = parentNode.alias)
                    subEncod = parentEncod + '0'
                    NodeList.append((node, subplan, subEncod, next_id))
                    parentNode.addChild(node)
                    adj_list.append((idx, next_id))
                    next_id += 1
                else:
                    for child_idx in range(subPlanNum - 1, -1 , -1):
                        subplan = parentPlan['Plans'][child_idx]
                        node = self.traverseNode(subplan, pos = child_idx, parentAlias = parentNode.alias)
                        subEncod = parentEncod + str(child_idx)
                        NodeList.append((node, subplan, subEncod, next_id))
                        parentNode.addChild(node)
                        adj_list.append((idx, next_id))
                        next_id += 1
        pc_dict = defaultdict(list)
        for parent, child in adj_list:
            pc_dict[parent].append(child)
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights':  torch.LongTensor(heights),
            'pc_dict':  pc_dict}

    def to_exechint(self, hintdict):
        if self.config.left_deep_restriction:
            hintdict['structure'] = [0] * len(hintdict['structure']) # bad pg_hint_plan!
        join_hint = []
        order_hint = copy.deepcopy(hintdict['join order'])
        join_hint_help = copy.deepcopy(hintdict['join order'])
        leading_hint = 'LEADING'
        for i in range(len(hintdict['join operator'])):
            i_struct = hintdict['structure'][i]
            join_hint_tmp = ' '.join(join_hint_help[i_struct:i_struct + 2])
            join_hint_help = join_hint_help[:i_struct] + [join_hint_tmp] + join_hint_help[i_struct + 2:]
            join_hint.append(hintdict['join operator'][i] + '('+ join_hint_tmp + ')')

            order_hint_tmp = '(' + ' '.join(order_hint[i_struct:i_struct + 2]) + ')'
            order_hint = order_hint[:i_struct] + [order_hint_tmp] + order_hint[i_struct + 2:]
        join_hint.reverse()
        leading_hint = leading_hint + '(' + order_hint[0] + ')'
        exechint = '/*+' + leading_hint + '\n' + '\n'.join(join_hint) + '*/\n'
        return exechint

@ray.remote
class RemotePlanHelper():
    def __init__(self,globalConfig):
        self.config = globalConfig
        self.planhelper = PlanHelper(globalConfig)
    def GetFeature(self,hint,sql,toextract,query_id = None):
        return self.planhelper.get_feature(hint,sql, toextract,query_id = query_id) 
    def GetLatency(self,hint,sql, query_id, timeout = None):
        if timeout == None:
            timeout = self.config.max_time_out
        return self.planhelper.getLatency(hint,sql, query_id, timeout)
    def SaveEncoding(self,path):
        self.planhelper.encoding.save_to_file(path)
    def GetPGLatencyBuffer(self):
        return self.planhelper.getPGLatencyBuffer()
    def GetTableNum(self):
        return self.planhelper.get_table_num()
    def GetExechint(self,hintdict):
        return self.planhelper.to_exechint(hintdict)
    def GetSortedQueryID(self):
        hintNum = self.planhelper.get_hintNum()
        sorted_keys = sorted(hintNum, key=hintNum.get)
        return sorted_keys