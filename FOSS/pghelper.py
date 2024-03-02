import psycopg2
# from config import Config
import os, shutil
import json
import time

class PGHelper:
    def __init__(self,globalConfig ,dbname = '',user = '',password = '',host = '',port = 5432):
        self.con = psycopg2.connect(database=dbname, user=user,password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.config = globalConfig
        self.cur.execute("load 'pg_hint_plan';")
        self.PG_DataType = ['smallint','integer','bigint','decimal','numeric','real',
                'double precision','smallserial','serial','bigserial']
        self.latencyBuffer = {}
        self.latencyTotalBuffer = {}
        if os.path.exists(self.config.pg_latency):
            shutil.copy(self.config.pg_latency, self.config.latency_buffer_path)
            tmp_buffer_file = open(self.config.latency_buffer_path,"r")
            lines = tmp_buffer_file.readlines()
            tmp_buffer_file.close()
            for line in lines:
                data = json.loads(line)
                if data[0] not in self.latencyBuffer:
                    self.latencyBuffer[data[0]] = {}
                self.latencyBuffer[data[0]][data[1]] = data[2]
            self.buffer_file = open(self.config.latency_buffer_path,"a")
        else:
            self.buffer_file = open(self.config.latency_buffer_path,"w")
        self.cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        self.table_names = [name[0] for name in self.cur.fetchall()]
        self.tablenum = len(self.table_names)

    def getLatency(self, hint, sql, queryid, timeout):
        # return [123.12,False], True,None
        if queryid in self.latencyBuffer:
            if hint in self.latencyBuffer[queryid]:
                return self.latencyBuffer[queryid][hint],False,None
        if queryid in self.latencyTotalBuffer:
            if hint in self.latencyTotalBuffer[queryid]:
                start = time.time()
                out = self.latencyTotalBuffer[queryid][hint]
                if out[0] >= timeout:
                    out[0] = timeout
                    out[1] = True
                if queryid not in self.latencyBuffer:
                    self.latencyBuffer[queryid] = {}
                self.latencyBuffer[queryid][hint] = out
                end = time.time()
                finding_time = end - start
                return out,True, finding_time
        try:
            self.cur.execute("SET statement_timeout = " + str(timeout) + ";")
            self.cur.execute(hint + "explain (COSTS, FORMAT JSON,ANALYZE) "+sql)
            rows = self.cur.fetchall()
            plan_json = rows[0][0][0]
            plan_json['timeout'] = False
        except KeyboardInterrupt:
            raise
        except:
            plan_json = {}
            plan_json['Plan'] = {'Actual Total Time':timeout}
            plan_json['timeout'] = True
            self.con.commit()
        if queryid not in self.latencyBuffer:
            self.latencyBuffer[queryid] = {}
        out = [plan_json['Plan']['Actual Total Time'],plan_json['timeout']]
        self.latencyBuffer[queryid][hint] = out
        self.buffer_file.write(json.dumps([queryid, hint, out])+"\n")
        self.buffer_file.flush()
        return out, True, None
    
    def tryGetLatency(self,hint,query_id):
        try:
            lat_timeout = self.latencyBuffer[query_id][hint]
            if lat_timeout[1]:
                return None
            else:
                return lat_timeout[0]
        except:
            return None
    def getLatencyNoCache(self,hint,sql,queryid,timeout):
        try:
            # self.cur.execute("SET geqo TO off;")
            self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
            self.cur.execute(hint + "explain (COSTS, FORMAT JSON,ANALYZE) "+sql)
            rows = self.cur.fetchall()
            plan_json = rows[0][0][0]
            plan_json['timeout'] = False
        except KeyboardInterrupt:
            raise
        except:
            plan_json = {}
            plan_json['Plan'] = {'Actual Total Time':timeout}
            plan_json['timeout'] = True
            self.con.commit()
        if queryid not in self.latencyBuffer:
            self.latencyBuffer[queryid] = {}
        out = [plan_json['Plan']['Actual Total Time'],plan_json['timeout']]
        self.latencyBuffer[queryid][hint] = out
        self.buffer_file.write(json.dumps([queryid, hint, out])+"\n")
        self.buffer_file.flush()
        return out, True
    def getCostPlanJson(self,hint,sql):
        import time
        startTime = time.time()
        self.cur.execute("SET statement_timeout = " + str(self.config.max_time_out) + ";")
        self.cur.execute(hint + "explain (COSTS, FORMAT JSON) " + sql)
        rows = self.cur.fetchall()
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time() - startTime
        return plan_json
    def get_minLatency(self):
        minLatency = {}
        for queryid in self.latencyBuffer:
            minlat = self.config.max_time_out
            hint2send = ''
            for hint in self.latencyBuffer[queryid]:
                if self.latencyBuffer[queryid][hint][0] < minlat:
                    minlat = self.latencyBuffer[queryid][hint][0]
                    hint2send  = hint
            # minLatency = round(minLatency,3)
            minLatency[queryid] = [minlat,hint2send]
        return minLatency

    
    def gettablenum(self):
        return self.tablenum
    
    def get_min_max_values(self,table_name, column_name):
        self.cur.execute(f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name};")
        min_val, max_val = self.cur.fetchone()
        if min_val != None and max_val != None:
            max_val = float(max_val)
            min_val = float(min_val)
        return min_val,max_val
    
    def get_column_data_properties(self):
        column_data_properties = {}
        for table_name in self.table_names:
            self.cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
            for column_name, data_type in self.cur.fetchall():
                if data_type in self.PG_DataType and column_name != 'cc_closed_date_sk':
                    min_val, max_val = self.get_min_max_values(table_name, column_name)
                    column_data_properties[table_name+'.'+column_name] = (min_val, max_val)
        return column_data_properties