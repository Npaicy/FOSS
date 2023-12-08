
import json
#@ray.remote
from config import Config
config = Config()
class Encoding:
    def __init__(self,column_min_max_vals = None):
        self.column_min_max_vals = column_min_max_vals
        self.op2idx ={'>': 0,'=': 1,'<': 2,'>=':3,'<=':4,'NA':5}
        self.idx2op = {0: '>', 1: '=', 2: '<',3: '>=',4: '<=', 5: 'NA'}
        self.col2idx = {'NA':0}
        self.idx2col = {0:'NA'}
        self.type2idx = {
            "Aggregate": 0,
            "Nested Loop": 1,
            "Seq Scan": 2,
            "Index Scan": 3,
            "Hash Join": 4,
            "Hash": 5,
            "Merge Join": 6,
            "Sort": 7,
            "Gather": 8,
            "Materialize": 9,
            "Index Only Scan": 10,
            "Bitmap Heap Scan": 11,
            "Bitmap Index Scan": 12,
            "Gather Merge": 13
        }
        self.idx2type = {
            "0": "Aggregate",
            "1": "Nested Loop",
            "2": "Seq Scan",
            "3": "Index Scan",
            "4": "Hash Join",
            "5": "Hash",
            "6": "Merge Join",
            "7": "Sort",
            "8": "Gather",
            "9": "Materialize",
            "10": "Index Only Scan",
            "11": "Bitmap Heap Scan",
            "12": "Bitmap Index Scan",
            "13": "Gather Merge"
        }
        # self.join2idx = {}
        # self.idx2join = {}
        self.table2idx = {'NA': 0}
        self.idx2table = {0: 'NA'}
        self.maxjoinlen = 0
    def loadcdp(self, column_min_max_vals):
        self.column_min_max_vals = column_min_max_vals
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]

        val_norm = 0.0
        if maxi > mini:
            val_norm = (val - mini) / (maxi - mini)
        return val_norm

    def encode_filters(self,filters=[], name=None):
        ## filters: list of dict

        if len(filters) == 0:
            return {
                'colId': [self.col2idx['NA']],
                'opId': [self.op2idx['NA']],
                'val': [0.0]
            }
        res = {'colId': [], 'opId': [], 'val': []}
        for filt in filters:
            # TODO:
            if ("::text" in filt):
                continue
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
                #           print(filters)
                #col, op, num = 'a','<','1'
                #try:
                col, op, num = f.split(' ')
                column = name + '.' + col
                if column not in self.col2idx:
                    self.col2idx[column] = len(self.col2idx)
                    self.idx2col[self.col2idx[column]] = column
                res['colId'].append(self.col2idx[column])
                res['opId'].append(self.op2idx[op])
                res['val'].append(self.normalize_val(column, float(num)))

            if (len(res['colId']) == 0):
                return {
                    'colId': [self.col2idx['NA']],
                    'opId': [self.op2idx['NA']],
                    'val': [0.0]
                }
        return res

    # def encode_join(self, join):
    #     if join not in self.join2idx:
    #         self.join2idx[join] = len(self.join2idx)
    #         self.idx2join[self.join2idx[join]] = join
    #     return self.join2idx[join]
    def encode_join(self, join):
        if join == None:
            return [self.col2idx['NA']] * config.maxjoins
        if ' and ' in join:
            join_lst = join.split(' and ')
        else:
            join_lst = [join]
        joinid = []
        for _join in join_lst:
            for tc in _join.split(' = '):
                if tc not in self.col2idx:
                    self.col2idx[tc] = len(self.col2idx)
                    self.idx2col[self.col2idx[tc]] = tc
                joinid.append(self.col2idx[tc])
        if len(joinid) > self.maxjoinlen:
            self.maxjoinlen = len(joinid)
            print('Now Join Length:',self.maxjoinlen)
        if len(joinid) < config.maxjoins:
            joinid.extend([self.col2idx['NA']] * (config.maxjoins - len(joinid)))
        elif len(joinid) > config.maxjoins:
            print(join)
            raise Exception('Too many joins! Please increase the value of maxjoins in config.py!')
        return joinid

    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]
    def save_to_file(self, filename):
        data = {
            "column_min_max_vals": self.column_min_max_vals,
            "op2idx": self.op2idx,
            "idx2op": self.idx2op,
            "col2idx": self.col2idx,
            "idx2col": self.idx2col,
            "type2idx": self.type2idx,
            "idx2type": self.idx2type,
            "table2idx": self.table2idx,
            "idx2table": self.idx2table
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_from_file(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
            self.column_min_max_vals = data["column_min_max_vals"]
            self.op2idx = data["op2idx"]
            self.idx2op = data["idx2op"]
            self.col2idx = data["col2idx"]
            self.idx2col = data["idx2col"]
            self.type2idx = data["type2idx"]
            self.idx2type = data["idx2type"]
            # self.join2idx = data["join2idx"]
            # self.idx2join = data["idx2join"]
            self.table2idx = data["table2idx"]
            self.idx2table = data["idx2table"]