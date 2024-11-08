import json
BINOP = [' >= ',' <= ',' = ',' > ',' < ']
class Encoding:
    def __init__(self,genConfig, column_min_max_vals = None):
        self.config = genConfig
        self.column_min_max_vals = column_min_max_vals
        self.op2idx ={'= ANY': 0,'>=':1,'<=':2,'>': 3,'=': 4,'<': 5,'NA':6,'IS NULL':7,'IS NOT NULL':8, '<>':9,'~~':10,'!~~':11, '~~*': 12}
        self.idx2op = {}
        for k,v in self.op2idx.items():
            self.idx2op[v] = k
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
            "Gather Merge": 13,
            "Limit": 14
        }
        self.idx2type = {
            0: "Aggregate",
            1: "Nested Loop",
            2: "Seq Scan",
            3: "Index Scan",
            4: "Hash Join",
            5: "Hash",
            6: "Merge Join",
            7: "Sort",
            8: "Gather",
            9: "Materialize",
            10: "Index Only Scan",
            11: "Bitmap Heap Scan",
            12: "Bitmap Index Scan",
            13: "Gather Merge",
            14: 'Limit'
        }
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

    def encode_filters(self, filters, alias, aliastable):
        res = {'colId': [], 'opId': [], 'val':[],'dtype':[]}
        if len(filters) == 0:
            return res  # 0:number  1:text 2:NULL
        for filt in filters:
            if "::" in filt:
                filt = filt.replace('::text','')
                filt = filt.replace('::bpchar','')
                filt = filt.replace('::date','')
                filt = filt.replace('::timestamp','')
                filt = filt.replace('::integer[]','')
                filt = filt.replace('::numeric','')
                filt_split = filt.split(' ')
                col = filt_split[0].strip("'()")
                if ' = ANY ' in filt:
                    op = self.op2idx['= ANY']
                else:
                    if filt_split[1] not in self.op2idx:
                        self.op2idx[filt_split[1]] = len(self.op2idx)
                    op = self.op2idx[filt_split[1]]
                val = 0.0
                dtype = 1
            elif 'IS NOT NULL' in filt:
                filt_split = filt.split(' ')
                col = filt_split[0].strip('()')
                op = self.op2idx['IS NOT NULL']
                val = 0.0
                dtype = 2
            elif 'IS NULL' in filt:
                filt_split = filt.split(' ')
                col = filt_split[0].strip('()')
                op = self.op2idx['IS NULL']
                val = 0.0
                dtype = 2
            else:
                filt = ''.join(c for c in filt if c not in '()')
                if ' OR ' in filt:
                    fs = filt.split(' OR ')
                elif ' AND ' in filt:
                    fs = filt.split(' AND ')
                else:
                    fs = [filt]
                for f in fs:
                    for tmpop in self.op2idx:
                        if tmpop in f:
                            try:
                                op = self.op2idx[tmpop]
                                col = f.split(tmpop)[0].strip()
                                val = self.normalize_val(aliastable[alias] + '.' + col, float(f.split(tmpop)[1]))
                                dtype = 0
                                break
                            except:
                                op = self.op2idx[tmpop]
                                col = f.split(tmpop)[0].strip()
                                val = 0.0
                                dtype = 1
                                # print(filters)
            column = aliastable[alias] + '.' + col
            if column not in self.col2idx:
                self.col2idx[column] = len(self.col2idx)
                self.idx2col[self.col2idx[column]] = column
            if self.col2idx[column] not in res['colId']:
                res['colId'].append(self.col2idx[column])
                res['opId'].append(op)
                res['val'].append(val)
                res['dtype'].append(dtype)

        return res

    def encode_join(self, join):
        joinNum = len(join) * 2
        if joinNum == 0:
            return [self.col2idx['NA']] * self.config.maxjoins
        joinid = []
        for onejoin in join:
            for tc in onejoin[1:]:
                if tc not in self.col2idx:
                    self.col2idx[tc] = len(self.col2idx)
                    self.idx2col[self.col2idx[tc]] = tc
                joinid.append(self.col2idx[tc])
        if joinNum > self.maxjoinlen:
            self.maxjoinlen = joinNum
            # print('Now Join Length:',self.maxjoinlen)
        if joinNum <= self.config.maxjoins:
            joinid.extend([self.col2idx['NA']] * (self.config.maxjoins - joinNum))
        else:
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
            self.table2idx = data["table2idx"]
            self.idx2table = data["idx2table"]