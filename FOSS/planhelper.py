import torch
import numpy as np
from collections import deque,defaultdict
from database_util import *
from pghelper import PGHelper
from encoding import Encoding
import os
import ray
#@ray.remote
class PlanHelper:

    def __init__(self, globalConfig):
        self.config = globalConfig
        self.pgrunner = PGHelper(globalConfig, globalConfig.database,globalConfig.user,globalConfig.password,globalConfig.ip,globalConfig.port)
        self.encoding = Encoding(self.config)
        if os.path.exists(globalConfig.encoding_path):
            self.encoding.load_from_file(globalConfig.encoding_path)
        else:
            print(' Init Encoding......')
            column_data_properties = self.pgrunner.get_column_data_properties()
            self.encoding.loadcdp(column_data_properties)
            # self.encoding.save_to_file(globalConfig.encoding_path)
        self.alias2full = {}
        self.treeNodes = [] 
        self.hist_file = None
        self.table_sample = None
    
    def getPGLatencyBuffer(self):
        return self.pgrunner.latencyBuffer
    
    def updatePGLatencyBuffer(self,latencyBuffer):
        self.pgrunner.latencyBuffer = latencyBuffer
    def gettablenum(self):
        return self.pgrunner.tablenum
    def getLatency(self,hint,sql,query_id,timeout = None):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.pgrunner.getLatency(hint,sql,query_id,timeout = timeout)
    def getMinLatency(self):
        return self.pgrunner.get_minLatency()
    def tryGetLatency(self,hint,query_id):
        return self.pgrunner.tryGetLatency(hint,query_id)
    def getLatencyNoCache(self,hint,sql,query_id,timeout = None):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.pgrunner.getLatencyNoCache(hint,sql,query_id,timeout = timeout)
    def get_feature(self,hint,sql,toextract,query_id = None,plan_json = None):
        if plan_json == None:
            plan_json = self.pgrunner.getCostPlanJson(hint, sql)
        collated_dict, hint,left_deep =self.js_node2dict(plan_json['Plan'], toextract,query_id = query_id)
        return collated_dict, hint,left_deep, plan_json
    def get_hintNum(self):
        hintNum = {}
        for queryid in self.pgrunner.latencyBuffer:
            hintNum[queryid] = len(self.pgrunner.latencyBuffer[queryid])
        return hintNum

    def js_node2dict(self, node,toextract,query_id = None):
        if toextract:
            hint = {'join order':[],'join operator':[]}
        else:
            hint = None
        # print(len(self.encoding.join2idx))
        try:
            treeNode = self.traversePlan(node,hint,toextract,query_id = query_id)
        except:
            print(query_id)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict, max_node=self.config.maxnode)
        self.treeNodes.clear()
        del self.treeNodes[:]
        left_deep = None
        if toextract:
            hint,left_deep = self.processhint(hint)
        return collated_dict, hint,left_deep
    
    def processhint(self,hint):
        left_deep = True
        newhint = {'join order':[],'join operator':[]}
        if len(hint['join operator']) <= 1:
            return newhint,False
        newhint['join order'].extend(hint['join order'][0])
        newhint['join operator'].append(self.config.operator_pg2hint[hint['join operator'][0]])
        for i in range(1,len(hint['join order'])):
            if hint['join order'][i][0] not in newhint['join order'] and hint['join order'][i][1] in newhint['join order']:
                newhint['join order'].append(hint['join order'][i][0])
                newhint['join operator'].append(self.config.operator_pg2hint[hint['join operator'][i]])
            elif hint['join order'][i][0] in newhint['join order'] and hint['join order'][i][1] not in newhint['join order']:
                newhint['join order'].append(hint['join order'][i][1])
                newhint['join operator'].append(self.config.operator_pg2hint[hint['join operator'][i]])
            elif  hint['join order'][i][0] not in newhint['join order'] and hint['join order'][i][1] not in newhint['join order']:
                newhint['join order'].extend(hint['join order'][i])
                newhint['join operator'].append(self.config.operator_pg2hint[hint['join operator'][i]])
                left_deep = False
        for i in range(len(newhint['join order'])):
            newhint['join order'][i] = newhint['join order'][i].strip('()')
        return newhint,left_deep

    # pre-process first half of old collator
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
            'heights': heights.numpy()
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))
        pc_dict = defaultdict(list)
        for parent, child in adj_list:
            pc_dict[parent].append(child)
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'pc_dict': pc_dict,
        }

    def topo_sort(self, root_node):
        #        nodes = []
        adj_list = []  #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        return adj_list, num_child, features

    def traversePlan(self,
                     plan,
                     #encoding,
                     hint,
                     toextract,
                     query_id = None,
                     pos=None):  # bfs accumulate plan
        # pos:{3:'root', 0:'left', 1:'right', 2:'internal-no-brother'}
        nodeType = plan['Node Type']
        table = 'NA'
        table_id = 0
        if 'Relation Name' in plan:
            table = plan['Relation Name']
            # table_id = ray.get(encoding.encode_table.remote(plan['Relation Name']))
            table_id = self.encoding.encode_table(plan['Relation Name'])
            if 'Alias' in plan and plan['Alias'] not in self.alias2full.keys():
                self.alias2full[plan['Alias']] = table
        
        # typeId = ray.get(encoding.encode_type.remote(nodeType))
        typeId = self.encoding.encode_type(nodeType)
        card = None  #plan['Actual Rows']
        join,filters,db_est,alias = processCond(plan)

        if len(join) > 1:
            join_ = ' and '.join(sorted(join))
        elif len(join) == 1:
            join_ = join[0]
        else: join_ = None
        joinids = self.encoding.encode_join(join_)

        # left_col,right_col = joinids[0], joinids[1]
        if alias != None:
            filters_encoded = self.encoding.encode_filters(filters, self.alias2full[alias])
        else:
            filters_encoded = self.encoding.encode_filters(filters)
        root = TreeNode(nodeType,table, table_id,typeId, filters, card, joinids, join_,
                        filters_encoded,db_est, pos)
        
        self.treeNodes.append(root)
        
        if pos == None:
            root.pos = 3
        else:
            root.pos = pos

        if 'Plans' in plan:
            if len(plan['Plans']) == 1:
                subplan = plan['Plans'][0]
                subplan['parent'] = plan
                node = self.traversePlan(subplan,hint,toextract, pos = 2,query_id=query_id)
                node.parent = root
                root.addChild(node)
            else:
                for child_idx, subplan in enumerate(plan['Plans']):
                    subplan['parent'] = plan
                    node = self.traversePlan(subplan, hint,toextract,pos = child_idx,query_id=query_id)
                    node.parent = root
                    root.addChild(node)
        root.query_id = query_id
        if toextract:
            gethint(plan,hint,join)
            # extracthint(plan,hint,alias)
        root.feature = node2feature(root)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order
    def to_exechint(self,hintdict):
        join_hint = []
        leading_hint = 'LEADING'
        temp = '('+' '.join(hintdict['join order'][0:2])+')'
        if(len(hintdict['join order']) > 2):
            for i in range(2,len(hintdict['join order'])):
                temp = '(' + temp + hintdict['join order'][i] + ')'
        leading_hint = leading_hint + '(' +temp + ')'
        for i in range(len(hintdict['join operator'])-1,-1,-1):
            join_hint.append(hintdict['join operator'][i]+'('+' '.join(hintdict['join order'][:i+2])+')')
        exechint = '/*+' + leading_hint + '\n' + '\n'.join(join_hint) + '*/\n'
        # print(exechint)
        return exechint

@ray.remote
class PlanHelperRemote():
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
        return self.planhelper.gettablenum()
    def GetExechint(self,hintdict):
        return self.planhelper.to_exechint(hintdict)
    def GetSortedQueryID(self):
        hintNum = self.planhelper.get_hintNum()
        sorted_keys = sorted(hintNum, key=hintNum.get)
        return sorted_keys