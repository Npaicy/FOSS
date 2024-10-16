import numpy as np
from collections import deque
import math
JOINTYPE = ["Nested Loop", "Hash Join", "Merge Join"]
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Filter','Recheck Cond']
SCANTYPE = ['Index Only Scan', 'Seq Scan', 'Index Scan', 'Bitmap Heap Scan','Tid Scan']
BINOP = [' >= ',' <= ',' = ',' > ',' < ']
def bfs(N, pc_dict, rel_pos_max):
    distance_matrix = np.full((N, N), True)
    for start_node in range(N):
        queue = deque([(start_node, 0)])  # node, distance
        while queue:
            node, distance = queue.popleft()
            for end_node in pc_dict[node]:
                if distance + 1 < rel_pos_max:
                    distance_matrix[start_node][end_node] = False
                queue.append((end_node, distance + 1))
        distance_matrix[start_node][start_node] = False
    return distance_matrix

def node2feature(node):
    filtmaxnum = 3
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((4, filtmaxnum - num_filter))
    filts = np.array(list(node.filterDict.values())) + 1  #cols, ops, vals, dtype
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(filtmaxnum)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId] + node.join) + 1
    table = np.array([node.table_id]) + 1
    pos = np.array([node.pos]) + 1
    db_est = np.array(node.db_est) + 1
    return np.concatenate((type_join, filts, mask, pos, table, db_est))

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    # x = x + 1 # pad id = 0
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) # + 1
        new_x[:xlen, :] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen, alpha):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(alpha)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = alpha
        x = new_x
    return x

def processCond(json_node, alias, alias2table):
    join = []
    filters = set()
    for condtype in CONDTYPE:
        if condtype in json_node:
            condition = json_node[condtype]
            if ' AND ' in condition:
                condition = condition[1:-1]
            cond_list = condition.split(' AND ')
            for cond in cond_list:
                cond = cond[1:-1]
                if condtype == 'Filter' or '::text' in cond or cond[-1].isnumeric():
                    filters.add((cond))
                else:
                    for op in BINOP:
                        if op in cond:
                            twoCol = [col.split(' ')[0].strip('() ') for col in cond.split(op)]
                            onejoin = [op]
                            for col in twoCol:
                                col_split = col.split('.')
                                if len(col_split) == 1:
                                    onejoin.append(alias2table[alias] + '.' + col_split[0])
                                else:
                                    onejoin.append(alias2table[col_split[0]] + '.' + col_split[1])
                            join.append(onejoin)
                            break
    planrows    = math.log10(1 + int(json_node['Plan Rows']))
    totalcost   = math.log10(1 + int(json_node['Total Cost']))
    planwidth   = math.log10(1 + int(json_node["Plan Width"]))
    startupcost = math.log10(1 + int(json_node['Startup Cost']))
    db_est = [planrows, totalcost, planwidth, startupcost]
    return join, list(filters), db_est

class TreeNode:

    def __init__(self, nodeType,table,table_id ,typeId, filt, join,
                 filterDict, db_est, pos):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = table
        self.table_id = table_id
        self.query_id = None 
        self.join = join
        self.children = []
        self.rounds = 0

        self.filterDict = filterDict
        self.db_est = db_est
        self.pos = pos
        self.alias = None
        self.parent = None

        self.feature = None

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter,
                                                    self.join_str,
                                                    len(self.children))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print('--' * indent + '{} with {} and {}, {} childs'.format(
            node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
