import numpy as np
from collections import deque
OPERATORTYPE = ["Nested Loop", "Hash Join", "Merge Join"]
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Filter','Recheck Cond']
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
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 60

    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M
# def node2feature(node):
#     # type, join, filter123, mask123,table,  pos
#     # 1, 1, 3x3 (9), 3, 1
#     num_filter = len(node.filterDict['colId'])
#     pad = np.zeros((3, 3 - num_filter))
#     filts = np.array(list(node.filterDict.values()))  #cols, ops, vals
#     ## 3x3 -> 9, get back with reshape 3,3
#     filts = np.concatenate((filts, pad), axis=1).flatten()
#     mask = np.zeros(3)
#     mask[:num_filter] = 1
#     type_join = np.array([node.typeId] + node.join)
#     # table
#     table = np.array([node.table_id])
#     pos = np.array([node.pos])
#     return np.concatenate((type_join, filts, mask, table, pos))
def node2feature(node, encoding, hist_file, table_sample = None):
    # type, join, filter123, mask123,table,  pos
    # 1, 1, 3x3 (9), 3, 1
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))  #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId] + node.join)
    # table
    # hists = filterDict2Hist(hist_file, node.filterDict, encoding)
    table = np.array([node.table_id])
    pos = np.array([node.pos])
    # if node.table_id == 0 or node.table not in table_sample[node.query_id]:
    #     sample = np.zeros(1000)
    # else:
    #     sample = table_sample[node.query_id][node.table]
    return np.concatenate((type_join, filts, mask, pos,table))
    # # else:

    # return np.concatenate((type_join, filts, mask, pos,hists,table,sample))
def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets-1))
    for i in range(len(filterDict['colId'])):
        colId = filterDict['colId'][i]
        
        # print(encoding.idx2col)
        col = encoding.idx2col[str(colId)]
        if col == 'NA':
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file['table_column']==col,'bins'].item()
        
        opId = filterDict['opId'][0]
        op = encoding.idx2op[str(opId)]
        
        val = filterDict['val'][0]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val_unnorm:
                left = j
            if bins[j]>val_unnorm:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op == '<' or op =='<=':
            res[:left] = 1
        elif op == '>' or op == '>=':
            res[right:] = 1
        ress[i] = res
    
    ress = ress.flatten()
    return ress    

# def node2feature(node):
#     # type, join, filter123, mask123
#     # 1, 1, 3x3 (9), 3
#     # TODO: add sample (or so-called table)


#     num_filter = len(node.filterDict['colId'])
#     pad = np.zeros((3, 3 - num_filter))
#     filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
#     ## 3x3 -> 9, get back with reshape 3,3
#     filts = np.concatenate((filts, pad), axis=1).flatten()  #将该节点的filter变成3*3矩阵后压缩为一维
#     mask = np.zeros(3)
#     mask[:num_filter] = 1
#     type_join = np.array([node.typeId, node.join])


#     table = np.array([node.table_id])

#     # return np.concatenate((type_join,filts,mask))
#     #sample是采样的行
#     return np.concatenate((type_join, filts, mask, table))

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
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen, alpha):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(alpha)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = alpha
        x = new_x
    return x#.unsqueeze(0)
# def pad_attn_bias_unsqueeze(x, padlen):
#     xlen = x.size(0)
#     if xlen < padlen:
#         new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
#         new_x[:xlen, :xlen] = x
#         new_x[xlen:, :xlen] = 0
#         x = new_x
#     return x

def processCond(json_node):
    alias = None
    if 'Alias' in json_node:
        alias = json_node['Alias']
    else:
        pl = json_node
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    join = []
    filters = set()
    for condtype in CONDTYPE:
        if condtype in json_node:
            cond = json_node[condtype]
            if ' AND ' in cond:
                cond = cond[1:-1]  #去除括号
            cond_list = cond.split(' AND ')
            for sc in cond_list:
                sc = sc[1:-1]
                for op in BINOP:
                    if op in sc:
                        if '\'' not in sc and not sc[-1].isnumeric():
                            twoCol = sc.split(op)
                            twoCol = [alias + '.' + col 
                                        if len(col.split('.')) == 1 else col for col in twoCol ]
                            join.append(op.join(twoCol))
                            break
                        elif sc[-1].isnumeric():
                            filters.add(sc)
                            break
    return join,list(filters),alias
                
            
def formatJoin(json_node,alias):
   
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    elif 'Merge Cond' in json_node and not json_node['Merge Cond'][-2].isnumeric():
        join = json_node['Merge Cond']
    
    ## sometimes no alias, say t.id 
    ## remove repeat (both way are the same)
    if join is not None:
        twoCol = join[1:-1].split(' = ')
        try:
            twoCol = [alias + '.' + col 
                    if len(col.split('.')) == 1 else col for col in twoCol ] 
        except:
            print(alias,twoCol,json_node)
        join = ' = '.join(sorted(twoCol))
    
    return join


def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break

    filters = []
    # recheck cond
    if 'Filter' in plan and plan['Filter'][-2].isnumeric():
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan and plan['Recheck Cond'][-2].isnumeric():
        filters.append(plan['Recheck Cond'])
  
    return filters, alias

def gethint(json_node,hint,join):
    if len(join) != 0:
        temp_node = json_node
        count_join = 0
        for tmpjoin in join:
            for o in BINOP:
                if o in tmpjoin:
                    twoCol = tmpjoin.split(o)
                    break
            tmp = []
            for col in twoCol:
                tmpsplit = col.split('.')
                if len(tmpsplit) == 1:
                    print('alias error')
                else:
                    tmp.append(tmpsplit[0])
            if tmp not in hint['join order']:
                hint['join order'].append(tmp)
                count_join += 1
        for i in range(count_join):
            while temp_node['Node Type'] not in OPERATORTYPE:
                if 'parent' in temp_node.keys():
                    temp_node = temp_node['parent']
                else:
                    print('extracthint Error!')
                    break
            hint['join operator'].append(temp_node['Node Type'])
        

def extracthint(json_node,hint,alias):

    join = None
    if 'Hash Cond' in json_node:
        hint['join operator'].append('Hash Join')
        join = [json_node['Hash Cond']]
 
    elif 'Join Filter' in json_node:
        hint['join operator'].append('Nested Loop')
        join = [json_node['Join Filter']]
    elif 'Merge Cond' in json_node:
        hint['join operator'].append('Merge Join')
        join = [json_node['Merge Cond']]
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        if ' AND ' in json_node['Index Cond']:
            join = json_node['Index Cond'].split(' AND ')
        else:
            join = [json_node['Index Cond']]
        temp_node = json_node
        for i in range(len(join)):
            while temp_node['Node Type'] not in OPERATORTYPE:
                if 'parent' in temp_node.keys():
                    temp_node = temp_node['parent']
                else:
                    #print(temp_node)
                    print('extracthint Error!')
                    break
            hint['join operator'].append(temp_node['Node Type'])
    if join is not None:
        for tmpjoin in join:
            twoCol = tmpjoin[1:-1].split(' = ')
            tmp = []
            # print(twoCol)
            for col in twoCol:
                tmpsplit = col.split('.')
                if len(tmpsplit) == 1:
                    tmp.append(alias)
                else:
                    tmp.append(tmpsplit[0])
            hint['join order'].append(tmp)

class TreeNode:

    def __init__(self, nodeType,table,table_id ,typeId, filt, card, join, join_str,
                 filterDict, pos):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = table
        self.table_id = table_id
        self.query_id = None  ## so that sample bitmap can recognise

        self.join = join
        self.join_str = join_str
        self.card = card  #'Actual Rows'
        self.children = []
        self.rounds = 0

        self.filterDict = filterDict
        self.pos = pos

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
