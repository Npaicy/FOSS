from operator import index
import psycopg2
# from config import Config
import os, shutil
import json
import pandas as pd
import time
PGDATATYPE = ['smallint','integer','bigint','decimal','numeric','real',
                'double precision','smallserial','serial','bigserial']
class PGHelper:
    def __init__(self, globalConfig):
        self.con = psycopg2.connect(database=globalConfig.database, user=globalConfig.user,
                                    password=globalConfig.password, host=globalConfig.ip,
                                    port=globalConfig.port)
        self.cur = self.con.cursor()
        self.config = globalConfig
        self.cur.execute("SET geqo=off;")
        self.cur.execute("load 'pg_hint_plan';")
        self.latencyBuffer = {}
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
        self.tablenum    = len(self.table_names)
    def getLatency(self, hint, sql, queryid, timeout, hintstyle):
        exectime = time.time()
        if queryid in self.latencyBuffer:
            if hint in self.latencyBuffer[queryid]:
                return self.latencyBuffer[queryid][hint], False, None
        try:
            if hint != '':
                self.cur.execute("SET statement_timeout = " + str(timeout) + ";")
            self.cur.execute(hint + sql)
            timeout = False
        except KeyboardInterrupt:
            raise
        except:
            timeout = True
            self.con.commit()
        if queryid not in self.latencyBuffer:
            self.latencyBuffer[queryid] = {}
        exectime = round((time.time() - exectime) * 1000, 3)
        latency_timeout = [exectime, timeout]
        self.latencyBuffer[queryid][hint] = latency_timeout
        self.buffer_file.write(json.dumps([queryid, hint, latency_timeout])+"\n")
        self.buffer_file.flush()
        return latency_timeout, True, None
    
    def tryGetLatency(self,hint,query_id):
        try:
            lat_timeout = self.latencyBuffer[query_id][hint]
            if lat_timeout[1]:
                return None
            else:
                return lat_timeout[0]
        except:
            return None
    def getCostPlanJson(self, hint, sql, hintstyle, query_id = None):
        import time
        startTime = time.time()
        try:
            self.cur.execute("SET statement_timeout = " + str(self.config.max_time_out) + ";")
        
            self.cur.execute(hint + "explain (COSTS, FORMAT JSON) " + sql) # Bao Test
            rows = self.cur.fetchall()
        except:
            print(hint + "explain (COSTS, FORMAT JSON) " + sql)
            raise
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
            minLatency[queryid] = [minlat,hint2send]
        return minLatency

    
    def get_table_num(self):
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
                if data_type in PGDATATYPE and column_name != 'cc_closed_date_sk':
                    min_val, max_val = self.get_min_max_values(table_name, column_name)
                    column_data_properties[table_name + '.' + column_name] = (min_val, max_val)
        return column_data_properties
    