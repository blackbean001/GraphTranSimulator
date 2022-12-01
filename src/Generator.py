# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:32:21 2022

@author: Administrator
"""

import networkx as nx
import os
import json
import random
import csv
import copy
import logging
import numpy as np
import itertools
from utils import *
from src import *
from collections import Counter, defaultdict

#root = "/Users/MrBlackBean/Dropbox/大湾区研究院工作文件夹/算法测试/graphtranssimulator"
#root = "C://Users//Administrator.DESKTOP-9T4A4SV//Dropbox//大湾区研究院工作文件夹//算法测试//graphtranssimulator"
root = "/home/lisong/algorithms/graphtranssimulator"

#logging.basicConfig(filename=os.path.join(root,"test.log"),level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphGenerator:
    def __init__(self, conf, sim_name=None, root=None):
        """ Generate graph from parameter files and conf files 
        param conf: JSON configuration file
        param sim_name: if not None, overrides the simulation_name in "conf_file"
        """
        # load conf
        if isinstance(conf,str):
            with open(conf,'r',encoding='utf8') as fr:
                self.conf = json.load(fr)
            logger.info("load conf: " + str(conf))
        if isinstance(conf,dict):
            self.conf = conf
        logger.info(json.dumps(self.conf,indent=2))
        if not root:
            self.project_root = self.conf["project_root"]
        else:
            self.project_root = root
        self.general_params = self.conf["general"]
        self.input_params = self.conf["input_files"]
        
        if sim_name == None:
            self.sim_name = conf["simulation_name"]
        else:
            self.sim_name = sim_name
        
        self.degree_file = os.path.join(self.project_root,self.input_params["directory"],self.input_params["degree"])
        self.acct_file = os.path.join(self.project_root,self.input_params["directory"],self.input_params["accounts"])
        self.normal_file = os.path.join(self.project_root,self.input_params["directory"],self.input_params["normal_patterns"])
        self.alert_file = os.path.join(self.project_root,self.input_params["directory"],self.input_params["alert_patterns"])
        
        # set random seed
        seed = self.general_params.get("random_seed")
        env_seed = os.getenv("RANDOM_SEED")
        if env_seed is not None:
            seed = env_seed  # Overwrite random seed if specified as an environment variable
        self.seed = seed if seed is None else int(seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        logger.info("Random seed: " + str(self.seed))
        
        self.num_nodes = -1
        self.edge_id = 0 # edge id (transaction id)

    def check_account_exist(self, aid):
        """Validate the existence of a specified account. If absent, it raises KeyError.
        :param aid: Account ID
        """
        if not self.g.has_node(aid):
            raise KeyError("Account %s does not exist" % str(aid))

    def set_num_accounts(self):
        with open(self.acct_file, "r") as rf:
            reader = csv.reader(rf)
            # Parse header
            header = next(reader)

            count = 0
            for row in reader:
                if row[0].startswith("#"):
                    continue
                num = int(row[header.index('count')])
                count += num

        self.num_accounts = count

    def add_graph_attribute(self):
        self.g.graph["sim_name"] = self.sim_name
       
    def generate_graph(self):
        '''
        generate directed random graph from degree.csv. degree.csv columns： count,in-degree,out-degree
        '''
        _in_deg, _out_deg = self.load_degrees(self.num_accounts)
        self.generate_graph_from_degrees(_in_deg, _out_deg, self.seed)
        self.add_graph_attribute()
        return self.g
        
    def add_edge_indexes(self):
        logger.info("Add %d base transactions" % self.g.number_of_edges())
        edge_count_dict = {}
        for (src, dst) in self.g.edges():
            self.check_account_exist(src)  # Ensure the originator and beneficiary accounts exist
            self.check_account_exist(dst)
            if (src,dst) not in edge_count_dict:
                edge_count_dict[(src,dst)] = 0
            else:
                edge_count_dict[(src,dst)] += 1
            if src == dst:
                raise ValueError("Self loop from/to %s is not allowed for transaction networks" % str(src))
            self.g.edges[src,dst,edge_count_dict[(src,dst)]]['edge_id'] = self.edge_id
            self.edge_id += 1             

    def generate_graph_from_degrees(self, _in_deg, _out_deg, seed=20, remove_selfloop=True):

        if not sum(_in_deg) == sum(_out_deg):
            raise nx.NetworkError('Invalid degree sequences. Sequences must have equal sums.')
        if not len(_in_deg) == len(_out_deg):
            raise nx.NetworkError('in-degree sequences should have the same length as out-degree sequences')
        if not len(_in_deg) == self.num_accounts:
            raise ValueError('in-degree sequences should have the same length as total number of accounts')
            
        self.num_nodes = len(_in_deg)
        
        _g = nx.empty_graph(self.num_nodes, nx.MultiDiGraph()) # allow self_loops and multiedges
        if self.num_nodes==0 or max(_in_deg)==0 or max(_out_deg)==0:
            return _g  # No edges exist
        
        self.g =nx.generators.degree_seq.directed_configuration_model(_in_deg, _out_deg, seed=self.seed)
        if remove_selfloop:
            for (f, t) in list(nx.selfloop_edges(self.g)):
                while self.g.has_edge(f,t):
                    self.g.remove_edge(f,t)
        
        self.add_edge_indexes()
        
        
        return self.g

    def load_degrees(self, num_v, special_token="#", auto_adjust=False):
        with open(self.degree_file,'r') as rf:
            reader = csv.reader(rf)
            header = next(reader)
            
            _in_deg = list()
            _out_deg = list()
            
            row_id = 0
            for row in reader:
                if row[0].startswith(special_token):
                    continue  # 保留函数
                try:
                    count, in_degree, out_degree = [int(item) for item in row]
                except:
                    logging.error("degree.csv: can not load row {}".format(row_id))
                    continue
                _in_deg += [in_degree]*count
                _out_deg += [out_degree]*count
                row_id += 1       
            
            in_len, out_len = len(_in_deg), len(_out_deg)
            if in_len != out_len:
                raise ValueError("The length of in-degree (%d) and out-degree (%d) sequences must be same."
                                 % (in_len, out_len))
            
            if not auto_adjust:
                total_in_deg, total_out_deg = sum(_in_deg), sum(_out_deg)
                if total_in_deg != total_out_deg:
                    raise ValueError("The sum of in-degree (%d) and out-degree (%d) must be same."
                                     % (total_in_deg, total_out_deg))
                if num_v % in_len != 0:
                    raise ValueError("The number of total accounts (%d) "
                                     "must be a multiple of the degree sequence length (%d)."
                                     % (num_v, in_len))
                repeats = num_v // in_len
                _in_deg = _in_deg * repeats
                _out_deg = _out_deg * repeats
            
            else:
                pass # to-do
        return _in_deg, _out_deg   

class TransactionGenerator():
    def __init__(self, G_generator):
        self.g = G_generator.g
        self.conf = G_generator.conf
        self.edge_id = G_generator.edge_id
        self.input_params = G_generator.input_params        
        self.project_root = G_generator.project_root
        self.general_params = G_generator.general_params
        self.acct_file = G_generator.acct_file
        self.normal_file = G_generator.normal_file
        self.alert_file = G_generator.alert_file
        
        self.attr_names = list()
        self.bank_to_accts_normal = defaultdict(set)
        self.bank_to_accts_alert = defaultdict(set)
        self.acct_to_bank_normal = defaultdict(str)
        self.acct_to_bank_alert = defaultdict(str)       
        
        self.graph_generator_params = self.conf["graph_generator"]
        self.degree_threshold = self.graph_generator_params["hub_degree_threshold"]
        
        self.is_aggregated = self.input_params["is_aggregated_accounts"]
        with open(os.path.join(self.project_root,self.input_params["directory"],self.input_params["region_map"]),'r') as fr:
            self.region_map = json.load(fr)
        
        self.default_params = self.conf["default"]
        self.default_bank_id = self.default_params["bank_id"]
        self.default_model = self.default_params["default_model"]
        self.default_start_day = self.default_params["start_day"]
        self.default_end_day = self.default_params["end_day"]
        self.default_start_range = self.default_params["start_range"]
        self.default_end_range = self.default_params["end_range"]
        self.default_count = self.default_params["count"]
        self.default_region = self.default_params["region"]
        self.default_business = self.default_params["business"]
        self.default_is_sar = self.default_params["is_sar"]
        self.default_count = self.default_params["count"]
        self.default_min_balance = self.default_params["min_balance"]
        self.default_max_balance = self.default_params["max_balance"]   
        self.default_min_amount = self.default_params["min_amount"]
        self.default_max_amount = self.default_params["max_amount"]
        
        self.output_params = self.conf["output_files"]
        self.output_dir = os.path.join(self.project_root, self.output_params["directory"])
        self.out_tx_file = self.output_params["trans_file"]
        self.out_account_file = self.output_params["acc_file"]
        self.out_alert_member_file = self.output_params["alert_members"]
        self.out_normal_models_file = self.output_params["normal_models"]
        
        self.normal_types = self.general_params["normal_types"]
        self.alert_types = self.general_params["alert_types"]
        self.simulation_steps = int(self.general_params["simulation_steps"])
        self.failure_thre = self.general_params["failure_thre"]
        self.margin_ratio = self.general_params["margin_ratio"]
        
        self.hubs_normal = self.get_hub_nodes()
        self.hubs_alert = copy.deepcopy(self.hubs_normal)
        self.num_accounts = self.get_num_accounts()
        self.normal_groups = dict()
        self.normal_id = 0
        self.alert_groups = dict()
        self.alert_id = 0
        
        
    def get_hub_nodes(self, check_hub_exists=True):
        """
        choose hub accounts with larger degree than the specified threshold
        """
        nodes = [n for n in self.g.nodes()  # Hub vertices (with large in/out degrees)
                 if self.degree_threshold <= self.g.in_degree(n)
                 or self.degree_threshold <= self.g.out_degree(n)]  
        if check_hub_exists:
            if len(nodes)==0:
                raise ValueError("No hub accounts found, please try again with smaller \"degree_threshold\" in conf.json")
        return set(nodes)

    def get_num_accounts(self, special_token="#"):
        if self.is_aggregated:      
            with open(self.acct_file,'r') as rf:
                reader = csv.reader(rf)
                header = next(reader)
                
                count = 0
                for row in reader:
                    if row[0].startswith(special_token):
                        continue
                    count += int(row[header.index("count")])
            self.num_accounts = count
        else:
            with open(self.acct_file,'r') as rf:
                reader = csv.reader(rf)
                header = next(reader)
                
                count = 0
                for row in reader:
                    if row[0].startswith(special_token):
                        continue
                    count += 1
        return count            

    def check_col_names(self, must_include_header, header):
        absent = []
        for name in must_include_header:
            if name not in header:
                absent.append(name)
        return absent

    def get_other_attributes():
        pass

    def assign_accounts_to_nodes(self):        
        with open(self.acct_file,'r') as fr:
            reader = csv.reader(fr)
        
            if self.is_aggregated:
                self.assign_aggregated_accounts(reader)
            else:
                self.assign_detailed_accounts(reader)

    def add_account(self, acct_id, **attr):
        """Add an account vertex
        :param acct_id: Account ID
        :param init_balance: Initial amount
        :param start: The day when the account opened
        :param end: The day when the account closed
        :param country: Country name
        :param business: Business type
        :param bank_id: Bank ID
        :param attr: Optional attributes-
        :return:
        """
        
        if attr['bank_id'] is None:
            attr['bank_id'] = self.default_bank_id
                
        self.g.nodes[acct_id].update(attr)

        self.bank_to_accts_normal[attr['bank_id']].add(acct_id)
        self.acct_to_bank_normal[acct_id] = attr['bank_id']  

        self.bank_to_accts_alert[attr['bank_id']].add(acct_id)
        self.acct_to_bank_alert[acct_id] = attr['bank_id'] 
          
    def assign_aggregated_accounts(self,reader,special_token="#",use_default=False):
        """
        1. Load and add account attributes from a csv file with aggregated parameters;
        2. csv columns must include: [count,min_balance,max_balance,start_day,start_range,end_day,end_range,region,business,bank_id]
        3. Attributes must include: [init_balance,start,end,region,business,bank_id,is_sar]
        4. Other attributes may include: [first_name,last_name,street_addr,city,state,country,gender,phone_number,birth_date], 
           set include_all=False and change self. get_other_attributes()

        """
        if self.default_min_balance is None:
            raise KeyError("Option 'default_min_balance' is required to load raw account list")
        min_balance = self.default_min_balance

        if self.default_max_balance is None:
            raise KeyError("Option 'default_max_balance' is required to load raw account list")
        max_balance = self.default_max_balance
        
        # check whether accounts.csv includes all must-included columns
        header = next(reader)
        must_include_header = ["count","min_balance","max_balance","start_day","start_range", \
                        "end_day","end_range","region","business","bank_id"]
        
        absent = self.check_col_names(must_include_header, header)
        if not use_default:
            if len(absent)>0:
                raise KeyError("{} must be included in accounts.csv".format(" ".join(absent)))    

        # add must_included attributes to self.attr_names
        must_included_attrs = ['init_balance','start','end','region','business','bank_id']
        
        self.attr_names.extend(must_included_attrs)
        
        acct_id = 0
        for row in reader:
            if row[0].startswith("#"):
                  continue       
            count = self.default_count if "count" in absent or row[header.index("count")]=='' else int(row[header.index("count")])
            min_balance = self.default_min_balance if "min_balance" in absent or row[header.index("min_balance")]=='' else float(row[header.index("min_balance")])
            max_balance = self.default_max_balance if "max_balance" in absent or row[header.index("max_balance")]=='' else float(row[header.index("max_balance")])
            start_day = self.default_start_day if "start_day" in absent or row[header.index("start_day")]=='' else int(row[header.index("start_day")])
            end_day = self.default_end_day if "end_day" in absent or row[header.index("end_day")]=='' else int(row[header.index("end_day")])
            start_range = self.default_start_range if "start_range" in absent or row[header.index("start_range")]=='' else int(row[header.index("start_range")])
            end_range = self.default_end_range if "end_range" in absent or row[header.index("end_range")]=='' else int(row[header.index("end_range")])
            region = self.default_region if "region" in absent or row[header.index("region")]=='' else row[header.index("region")]
            business = self.default_business if "business" in absent or row[header.index("business")]=='' else row[header.index("business")]
            bank_id = self.default_bank_id if "bank_id" in absent or row[header.index("bank_id")]=='' else row[header.index("bank_id")]
            
            for i in range(count):
                start,end = day_sampling(start_day,start_range,end_day,end_range,"uniform")
                init_balance = UniformSampling(min_balance, max_balance).get()  # Generate amount
                attr = {"init_balance":init_balance,"start":start,"end":end,"region":region,
                        "business":business,"bank_id":bank_id}
                                
                self.add_account(acct_id, **attr)
                acct_id += 1

        logger.info("Generated {} accounts.".format(self.num_accounts))            
    
    def assign_detailed_accounts(self,reader,special_token="#",use_default=False):
        """
        Load and add account attributes from a csv file with detailed account parameters;
        csv_columns must include: [uuid,init_balance,start_day,end_day,region,business,bank_id];
        attributes must include: [init_balance,start,end,region,business,bank_id,is_sar]
        Other attributes may include: [first_name,last_name,street_addr,city,state,country,gender,phone_number,birth_date]
        """
        if self.default_min_balance is None:
            raise KeyError("Option 'default_min_balance' is required to load raw account list")
        min_balance = self.default_min_balance

        if self.default_max_balance is None:
            raise KeyError("Option 'default_max_balance' is required to load raw account list")
        max_balance = self.default_max_balance
        
        # check whether accounts.csv includes all must-included columns
        header = next(reader)
        must_include_header = ["uuid","init_balance","start_day",
                        "end_day","region","business","bank_id"]
        
        if 'uuid' not in header:
            raise KeyError("{} must be included in accounts.csv".format("uuid"))

        absent = self.check_col_names(must_include_header, header)
        
        if not use_default:
            if len(absent)>0:
                raise KeyError("{} must be included in accounts.csv".format(" ".join(absent)))    
            
        # add must_included attributes to self.attr_names
        must_included_attrs = ['init_balance','start','end','region','business','bank_id','is_sar']

        float_attr = ["min_balance","max_balance"]
        int_attr = ["uuid","start_day","end_day","start_range","end_range"]
        for row in reader:
            if row[0].startswith("#"):
                  continue

            acc_id = 0 if "acc_id" in absent or row[header.index("acc_id")]==None else int(row[header.index("acc_id")])
            min_balance = self.default_min_balance if "min_balance" in absent or row[header.index("min_balance")]=='' else float(row[header.index("min_balance")])
            max_balance = self.default_max_balance if "max_balance" in absent or row[header.index("max_balance")]=='' else float(row[header.index("max_balance")])
            start_day = self.default_start_day if "start_day" in absent or row[header.index("start_day")]=='' else int(row[header.index("start_day")])
            end_day = self.default_end_day if "end_day" in absent or row[header.index("end_day")]=='' else int(row[header.index("end_day")])
            start_range = self.default_start_range if "start_range" in absent or row[header.index("start_range")]=='' else int(row[header.index("start_range")])
            end_range = self.default_end_range if "end_range" in absent or row[header.index("end_range")]=='' else int(row[header.index("end_range")])
            region = self.default_region if "region" in absent or row[header.index("region")]=='' else row[header.index("region")]
            business = self.default_business if "business" in absent or row[header.index("business")]=='' else row[header.index("business")]
            bank_id = self.default_bank_id if "bank_id" in absent or row[header.index("bank_id")]=='' else row[header.index("bank_id")]
            
            
            start, end = self.sample_day(start_day,start_range,end_day,end_range,"uniform")
            init_balance = UniformSampling(min_balance, max_balance).get()  # Generate amount
            attr = {"init_balance":init_balance,"start":start,"end":end,"region":region,
                    "business":business,"bank_id":bank_id}

            self.add_account(acc_id, **attr)
        
    def generate_normal_transactions(self):
        """
        generate normal transactions from normalModels.csv and alertPatterns.csv
        """
        # generator normal transactions
        self.nominator = Nominator(self.g, self.degree_threshold, logger)
        self.assign_normal_transactions()


    def assign_normal_transactions(self, special_token="#"):
        """
        1. assign normal transaction patterns to the entire graph (generated by degrees.csv)
        2. normalPatterns.csv columns: count, type,  min_account, max_account, min_period, max_period, bank_id
        3. normalPattern types: fan_in, fan_out, forward, single, mutual, periodical;
        """
        with open(self.normal_file,'r') as rf:
            reader = csv.reader(rf)
            self.nominator.initialize_type_counts(reader, self.hubs_normal, self.acct_to_bank_normal, self.bank_to_accts_normal, special_token="#")

        with open(self.normal_file,'r') as rf:
            reader = csv.reader(rf)
            header = next(reader)
            
            must_included = ['count', 'type',  'min_accounts', 'max_accounts', 'min_amount', 'max_amount', 'min_period', 'max_period', 'bank_id']
            absent = self.check_col_names(must_included, header)
            if len(absent)>0:
                raise KeyError("{} must be included in accounts.csv".format(" ".join(absent)))    
            
            for name in must_included:
                idx_count = header.index("count")
                idx_type = header.index("type")
                idx_min_accounts = header.index("min_accounts")
                idx_max_accounts = header.index("max_accounts")
                idx_min_amount = header.index("min_amount")
                idx_max_amount = header.index("max_amount")
                idx_min_period = header.index("min_period")
                idx_max_period = header.index("max_period")
                idx_bank_id = header.index("bank_id")
                
            for row in reader:
                logger.info("processing normal transaction: " + " ".join(row))
                if len(row)==0 or row[0].startswith(special_token):
                      continue
                count = int(row[idx_count])
                type = row[idx_type]
                min_accts = int(row[idx_min_accounts])
                max_accts = int(row[idx_max_accounts])
                min_amount = parse_float(row[idx_min_amount])
                max_amount = parse_float(row[idx_max_amount])
                min_period = parse_int(row[idx_min_period])
                max_period = parse_int(row[idx_max_period])
                bank_id = row[idx_bank_id] if row[idx_bank_id] is not None else ""  # If empty, it has inter-bank transactions
                
                if type not in self.normal_types:
                    logger.warning("- Pattern type name (%s) must be one of %s" % (type, str(self.normal_types.keys())))        
                
                # 当前逻辑是，当某一个 type 连续多次失败，则将 self.nominator.remaining_count_dict[type] 设置为 0；
#                while self.nominator.remaining_count_dict[type] > 0 and \
#                         self.nominator.continuous_failure_counts[type] < self.failure_thre:
                while self.nominator.remaining_count_dict[type] > 0:

                    num_accts = random.randrange(min_accts, max_accts+1)
                    period = random.randrange(min_period, max_period+1)
                    
                    self.build_normal_models(type, num_accts, min_amount, max_amount, period, bank_id, 
                                             is_normal=True)
                
                if self.nominator.remaining_count_dict[type] == 0:
                    logger.info("finished doing jobs for type:{}".format(type))
                #if self.nominator.continuous_failure_counts[type] == self.failure_thre:
                #    logger.warning("- type:{} meets continuous failure threshold {}, quit doing jobs...".format(type, self.failure_thre))
            
            # 剩余没有分配的 edges 设置为 single, 参数为 conf 中的 default 值
            sub_g_list = self.nominator.assign_remaining_single(self.default_min_amount,self.default_max_amount,
                                                                self.default_start_day,self.default_end_day,
                                                                self.normal_types)
            for sub_g in sub_g_list:
                self.normal_groups[self.normal_id] = sub_g
                self.normal_id += 1
                
            self.g = self.nominator.g
   
    # AMLSim 的代码完全没有用到 num_accts, min_amount 等参数
    def build_normal_models(self, type, num_accounts, min_amount, max_amount, period, bank_id, is_normal,
                            special_token="#"): 
        """Add an normal typology transaction set
        :param type: Name of pattern type
            (fan_in, fan_out, forward, single, mutual, periodical)
        :param num_accounts: Number of transaction members (accounts) 
        :param min_amount: Minimum amount of the transaction
        :param max_amount: Maximum amount of the transaction
        :param period: Period (number of days) for all transactions
        :param bank_id: Bank ID which it chooses members from. If empty, it chooses members from all banks
        """  
        if len(self.nominator.acct_to_bank[type].keys())==0:
            logger.warning("- no appropriate candidates left for type: {}".format(type))
            return

        start_date = random.randrange(0, self.simulation_steps-period+1)
        end_date = start_date + period - 1

        type_id = self.normal_types[type]
        sub_g = nx.DiGraph(type_id=type_id, reason=type, start=start_date, end=end_date)
        self.nominator.init_sub_g(sub_g)
        if len(self.nominator.bank_to_accts[type][bank_id])==0:
            logger.warning("- No {} candidates found for bank_id {}".format(type, bank_id))
            return
        # 从 self.hub 中随机选，先从bank_id中选，如果没有则随机选
        _main_acct, _main_bank_id = self.nominator.get_main_acct(num_accounts, bank_id, type)
        
        if type == "fan_in":
            sub_g = self.nominator.get_fan_in_subgraph(_main_acct, num_accounts, _main_bank_id, 
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1

        if type == "fan_out":
            sub_g = self.nominator.get_fan_out_subgraph(_main_acct, num_accounts, _main_bank_id, 
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1
        
        if type == "forward":
            sub_g = self.nominator.get_forward_subgraph(_main_acct, num_accounts, _main_bank_id, 
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1

        if type == "mutual":
            sub_g = self.nominator.get_mutual_subgraph(_main_acct, num_accounts, _main_bank_id,  \
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1

        if type == "periodical":
            sub_g = self.nominator.get_periodical_subgraph(_main_acct, num_accounts, _main_bank_id, 
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1

        if type == "single":
            sub_g = self.nominator.get_single_subgraph(_main_acct, num_accounts, _main_bank_id, 
                                                           start_date, end_date, min_amount, max_amount)
            if sub_g == None:
                logger.warning("- could not generate subgraph for type {}, is trying again with a new main_node...".format(type))
                self.nominator.continuous_failure_counts[type] += 1
                return
            self.nominator.continuous_failure_counts[type] == 0
            self.nominator.remaining_count_dict[type] -= 1
            self.nominator.done_count_dict[type] += 1
        
        sub_g.graph[MAIN_ACCT_KEY] = _main_acct
        sub_g.graph[TYPE_KEY] = type
        sub_g.graph[IS_NORMAL_KEY] = True
        self.normal_groups[self.normal_id] = sub_g
        self.normal_id += 1
        logger.info("successfully generate normal subgraph for type: {}".format(type))

    def remove_alert_candidate(self, acct, type):
        """Remove an account vertex from AML typology member candidates
        :param acct: Account ID
        """
        self.hubs_alert[type].discard(acct)
        bank_id = self.acct_to_bank_alert[type][acct]
        del self.acct_to_bank_alert[type][acct]
        self.bank_to_accts_alert[type][bank_id].discard(acct)

    def add_alert_candidate(self, acct, type):
        self.hubs_alert[type].add(acct)
        bank_id = self.acct_to_bank_alert[type][acct]
        self.acct_to_bank_alert[type][acct] = bank_id
        self.bank_to_accts_alert[type][bank_id].add(acct) 
        
    def add_type(self, bank_to_accts, acct_to_bankid, hubs_alert, types):
        new_acct_to_bankid = dict()
        new_bank_to_accts = dict()
        new_hubs_alert = dict()
        for type in types:
            new_acct_to_bankid[type] = copy.deepcopy(acct_to_bankid)
            new_bank_to_accts[type] = copy.deepcopy(bank_to_accts)
            new_hubs_alert[type] = copy.deepcopy(hubs_alert)
        return new_bank_to_accts, new_acct_to_bankid, new_hubs_alert
        
    def generate_alert_transactions(self, special_token="#"):
        logger.info("start generating alert transactions......")
        """
        1. alertPatterns.csv columns: count, type, min_account, max_account, min_period, max_period, bank_id, is_sar
        2. alertPattern types: (fan_in, fan_out, cycle, bipartite, stack, random, scatter_gather, gather_scatter);
        """
        with open(self.alert_file,"r") as rf:
            reader = csv.reader(rf)
            header = next(reader)
            must_included = ["count", "type", "min_accounts", "max_accounts", "min_period", "max_period", "bank_id", "is_sar"]
            absent = self.check_col_names(must_included, header)
            if len(absent)>0:
                logger.error("{} must be included in accounts.csv".format(" ".join(absent)))
                raise KeyError("{} must be included in accounts.csv".format(" ".join(absent)))    
            
            for name in must_included:
                idx_count = header.index("count")
                idx_type = header.index("type")
                idx_min_accounts = header.index("min_accounts")
                idx_max_accounts = header.index("max_accounts")
                idx_min_amount = header.index("min_amount")
                idx_max_amount = header.index("max_amount")
                idx_min_period = header.index("min_period")
                idx_max_period = header.index("max_period")
                idx_bank_id = header.index("bank_id")
                idx_is_sar = header.index("is_sar")
            
            # add type to bank_to_accts_alert / acct_to_bank_alert / hubs_alert
            types = []
            for row in reader:
                type = row[idx_type]
                types.append(type)
            self.bank_to_accts_alert, self.acct_to_bank_alert, self.hubs_alert = self.add_type(self.bank_to_accts_alert, \
                                                                                 self.acct_to_bank_alert, self.hubs_alert, types)            

        with open(self.alert_file,"r") as rf:
            reader = csv.reader(rf)
            header = next(reader)
            for row in reader:
                logger.info("processing alert transaction: " + " ".join(row))
                if len(row)==0 or row[0].startswith(special_token):
                      continue
                count = int(row[idx_count])
                type = row[idx_type]
                min_accts = int(row[idx_min_accounts])
                max_accts = int(row[idx_max_accounts])
                min_amount = parse_float(row[idx_min_amount])
                max_amount = parse_float(row[idx_max_amount])
                min_period = parse_int(row[idx_min_period])
                max_period = parse_int(row[idx_max_period])
                bank_id = row[idx_bank_id] if row[idx_bank_id] is not None else ""  # If empty, it has inter-bank transactions
                is_sar = row[idx_is_sar]
                
                if type not in self.alert_types:
                    logger.error("Pattern type name (%s) must be one of %s" % (type, str(self.alert_types.keys())))        
                
                # to-do: 与 normal一样，找不到 main_node 时制定一些停止措施
                # 由于 alert model 是增加 edge，因此不需要和 normal一样在出错return时考虑 add back main_node 的情况
                for i in range(count):
                    num_accts = random.randrange(min_accts, max_accts+1)
                    period = random.randrange(min_period, max_period+1)

                    self.add_alert_typology(type, num_accts, min_amount, max_amount, period, bank_id, is_sar=is_sar, cur_count = i)
    
    def add_alert_typology(self, type, num_accounts, min_amount, max_amount, period, bank_id="", is_sar=False, cur_count = 0):
        """Add an alert typology transaction set
        :param type_name: Name of pattern type
            (fan_in, fan_out, cycle, bipartite, stack, random, scatter_gather, gather_scatter)
        :param num_accounts: Number of transaction members (accounts)
        :param min_amount: Minimum amount of the transaction
        :param max_amount: Maximum amount of the transaction
        :param period: Period (number of days) for all transactions
        :param bank_id: Bank ID which it chooses members from. If empty, it chooses members from all banks
        :param is_sar: whether the account is sar account
        """                    
        def add_node_attr(_acct):
            attr_dict = self.g.nodes[_acct]
            attr_dict[IS_SAR_KEY] = is_sar
            sub_g.add_node(_acct, **attr_dict)
        
        def add_edge(_orig, _bene, _amount, _date, type):
            """
            Add transaction edge to the normal topology subgraph as well as the whole transaction graph
            (thus the generated whole graph's degrees does not follow exactly as the degree.csv)
            """
            self.g.add_edge(_orig, _bene, amount=_amount, date=_date, type=type, edge_id = self.edge_id, IS_SAR_KEY = is_sar, IS_ALERT_KEY = True)

            sub_g.add_edge(_orig, _bene, amount=_amount, date=_date, type=type, edge_id = self.edge_id, IS_SAR_KEY = is_sar, IS_ALERT_KEY = True)
            self.edge_id += 1

        def get_main_acct():
            """Create a main account ID and a bank ID from hub accounts
            :return: main account ID and bank ID
            """
            if [node for node in self.hubs_alert[type] if self.acct_to_bank_alert[type][node]==bank_id] == []:
                logger.error("not enough main accounts found for type {}, current number {}".format(type, cur_count))
                raise ValueError("not enough main accounts found for type {}".format(type))
            _main_acct = random.sample([node for node in self.hubs_alert[type] if \
                                        self.acct_to_bank_alert[type][node]==bank_id], 1)[0]
            _main_bank_id = bank_id
            self.remove_alert_candidate(_main_acct,type)
            add_node_attr(_main_acct)
            return _main_acct, _main_bank_id
                
        # 判断是否 external
        def judge_external(type):
            is_part_external, is_total_external, is_internal = None, None, None
            if bank_id != "" and bank_id not in self.bank_to_accts_alert[type]:  # Invalid bank ID
                raise KeyError("No such bank ID: %s" % bank_id)
            if len(self.acct_to_bank_alert[type]) < num_accounts:
                logger.error("the number of all remaining accounts is less than the number required for type: {}, current number: {}".format(type, cur_count))
                raise ValueError("the number of all remaining accounts is less than the number required for type: {}, current number: {}".format(type, cur_count))           
            if len(self.bank_to_accts_alert[type]) >= 2:
                if len(self.acct_to_bank_alert[type]) >= num_accounts:
                    if bank_id == "":
                        is_total_external = True
                    else:
                        if len(self.bank_to_accts_alert[type][bank_id]) < num_accounts-1:
                            is_part_external = True
                        else:
                            is_internal = True
            else:
                if len(self.acct_to_bank_alert[type]) >= num_accounts:
                    is_internal = True
            return is_part_external, is_total_external, is_internal
        
        logger.info("generate alert subgraph for type: {}".format(type))
        
        start_date = random.randrange(0, self.simulation_steps-period+1)
        end_date = start_date + period - 1

        num_neighbors = num_accounts - 1
        type_id = self.alert_types[type]
        sub_g = nx.DiGraph(type_id=type_id, reason=type, start=start_date, end=end_date)
        
        if type=="fan_in":
            is_part_external, is_total_external, is_internal = judge_external(type)
            main_acct, main_bank_id = get_main_acct()

            if is_total_external:
                sub_bank_candidates = [b for b,nbs in self.bank_to_accts_alert[type].items()  \
                                       if b != main_bank_id and len(nbs) >= num_neighbors]
                sub_bank_id = random.choice(sub_bank_candidates)
            if is_internal:
                sub_bank_id = main_bank_id
                
            if is_total_external:
                sub_accts = random.sample(self.bank_to_accts_alert[type][sub_bank_id], num_neighbors)
            if is_part_external:
                sub_accts_candidate = [acct for acct in self.acct_to_bank_alert[type].keys()  \
                                         if self.acct_to_bank_alert[type][acct]!=main_bank_id]
                sub_accts = random.sample(sub_accts_candidate,  \
                                          num_neighbors-len(self.bank_to_accts_alert[type][main_bank_id])+1)  \
                                          + list(set(self.bank_to_accts_alert[type][main_bank_id])-{main_acct})

            if is_internal:
                sub_accts = random.sample(list(set(self.bank_to_accts_alert[type][main_bank_id])-{main_acct}), num_neighbors)
            
            for n in sub_accts:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)

            for orig in sub_accts:
                amount = RoundedSampling(min_amount, max_amount).get()
                date = random.randrange(start_date, end_date + 1)
                add_edge(orig, main_acct, amount, date, type)
            
            logger.info("successfully generate alert subgraph: {}".format(type))
        
        elif type=="fan_out":
            is_part_external, is_total_external, is_internal = judge_external(type)

            main_acct, main_bank_id = get_main_acct()
            if is_total_external:
                sub_bank_candidates = [b for b, nbs in self.bank_to_accts_alert[type].items()
                                       if b != main_bank_id and len(nbs) >= num_neighbors]
                if not sub_bank_candidates:
                    #self.add_alert_candidates(main_acct)
                    logger.warning("- No banks with appropriate number of neighboring accounts found.")
                    return
                sub_bank_id = random.choice(sub_bank_candidates)
            if is_internal:
                sub_bank_id = main_bank_id
            
            if is_total_external:
                sub_accts = random.sample(self.bank_to_accts_alert[type][sub_bank_id], num_neighbors)
            if is_part_external:
                sub_accts_candidate = [acct for acct in self.acct_to_bank_alert[type].keys()  \
                                         if self.acct_to_bank_alert[type][acct]!=main_bank_id]
                sub_accts = random.sample(sub_accts_candidate,  \
                                          num_neighbors-len(self.bank_to_accts_alert[type][main_bank_id])+1)  \
                                          + list(set(self.bank_to_accts_alert[type][main_bank_id])-{main_acct})
            if is_internal:
                sub_accts = random.sample(list(set(self.bank_to_accts_alert[type][main_bank_id])-{main_acct}), num_neighbors)

            for n in sub_accts:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)

            for bene in sub_accts:
                amount = RoundedSampling(min_amount, max_amount).get()
                date = random.randrange(start_date, end_date + 1)
                add_edge(main_acct, bene, amount, date, type)
            
            logger.info("successfully generate alert subgraph: {}".format(type))


        elif type == "bipartite":  # bipartite (originators -> many-to-many -> beneficiaries)
            is_part_external, is_total_external, is_internal = judge_external(type)

            if is_total_external:
                sub_bank_candidates = [b for b, nbs in self.bank_to_accts_alert[type].items()
                                       if b != bank_id and len(nbs) >= num_neighbors]
                sub_bank_id = random.choice(sub_bank_candidates)
            if is_internal or is_part_external:
                sub_bank_id = bank_id

            num_orig_accts = num_accounts // 2  # The former half members are originator accounts
            num_bene_accts = num_accounts - num_orig_accts  # The latter half members are beneficiary accounts
            
            if is_part_external:
                sub_accts_candidate = [acct for acct in self.acct_to_bank_alert[type].keys()  \
                                         if self.acct_to_bank_alert[type][acct]!=sub_bank_id]
                sub_accts_candidate = random.sample(sub_accts_candidate,  \
                                                    num_accounts - len(self.bank_to_accts_alert[type][sub_bank_id])) \
                                                    + self.bank_to_accts_alert[type][sub_bank_id]

            if is_total_external or is_internal:
                sub_accts_candidates = random.sample(list(set(self.bank_to_accts_alert[type][sub_bank_id])), \
                                                     num_accounts)
            
            [orig_accts, bene_accts] = split_sampling(sub_accts_candidates,[num_orig_accts,num_bene_accts])
            main_acct = random.choice(orig_accts)            
            for n in sub_accts_candidates:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)

            for orig, bene in itertools.product(orig_accts, bene_accts):  # All-to-all transaction edges
                amount = RandomAmount(min_amount, max_amount).getAmount()
                date = random.randrange(start_date, end_date + 1)
                add_edge(orig, bene, amount, date, type)

            logger.info("successfully generate alert subgraph: {}".format(type))


        elif type == "stack":  # stacked bipartite layers
            is_part_external, is_total_external, is_internal = judge_external(type)

            if is_total_external:
                if len(self.bank_to_accts_alert[type]) >= 4:
                    [orig_bank_id, mid_bank_id, bene_bank_id] =   \
                    random.sample([b for b in self.bank_to_accts_alert[type].keys() if b!=bank_id], 3)
                elif len(self.bank_to_accts_alert[type]) == 3:
                    [orig_bank_id, mid_bank_id] = random.sample([b for b in self.bank_to_accts_alert[type].keys() if b!=bank_id], 2)
                    bene_bank_id = sub_bank_id
                else:
                    orig_bank_id = mid_bank_id = bene_bank_id =   \
                        random.sample([b for b in self.bank_to_accts_alert[type].keys() if b!=bank_id], 1)[0]

            # First and second 1/3 of members: originator and intermediate accounts
            num_orig_accts = num_mid_accts = num_accounts // 3
            # Last 1/3 of members: beneficiary accounts
            num_bene_accts = num_accounts - num_orig_accts * 2

            if is_total_external:
                orig_accts = random.sample(self.bank_to_accts_alert[type][orig_bank_id], num_orig_accts)
                mid_accts = random.sample(self.bank_to_accts_alert[type][mid_bank_id], num_mid_accts)
                bene_accts = random.sample(self.bank_to_accts_alert[type][bene_bank_id], num_bene_accts)
            if is_part_external:
                sub_acct_candidates = random.sample([a for a,b in self.bank_to_accts_alert[type].items() if b!= bank_id],
                                                     num_accounts-len(self.bank_to_accts_alert[type][bank_id])) \
                                                     + self.bank_to_accts_alert[type][bank_id]
                [orig_accts, mid_accts, bene_accts] = split_sampling(sub_acct_candidates,  \
                                                      [num_orig_accts,num_mid_accts,num_bene_accts])
            if is_internal:
                sub_acct_candidates = random.sample(self.bank_to_accts_alert[type][bank_id], num_accounts)
                [orig_accts, mid_accts, bene_accts] = split_sampling(sub_acct_candidates,  \
                                                      [num_orig_accts,num_mid_accts,num_bene_accts])                
            
            main_acct = random.choice(orig_accts)
            for n in orig_accts:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)      
            for n in mid_accts:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)
            for n in bene_accts:
                self.remove_alert_candidate(n, type)
                add_node_attr(n)
                
            for orig, bene in itertools.product(orig_accts, mid_accts):  # all-to-all transactions
                amount = RandomAmount(min_amount, max_amount).getAmount()
                date = random.randrange(start_date, end_date + 1)
                add_edge(orig, bene, amount, date, type)

            for orig, bene in itertools.product(mid_accts, bene_accts):  # all-to-all transactions
                amount = RandomAmount(min_amount, max_amount).getAmount()
                date = random.randrange(start_date, end_date + 1)
                add_edge(orig, bene, amount, date, type)

            logger.info("successfully generate alert subgraph: {}".format(type))


        elif type == "random":  # Random transactions among members
            is_part_external, is_total_external, is_internal = judge_external(type)

            amount = RandomAmount(min_amount, max_amount).getAmount()
            date = random.randrange(start_date, end_date + 1)

            if is_total_external or is_internal:
                if is_total_external:
                    all_bank_ids = [n for n in self.bank_to_accts_alert[type].keys() if n!=bank_id]
                if is_internal:
                    all_bank_ids = [n for n in self.bank_to_accts_alert[type].keys() if n==bank_id]                    
                bank_id_iter = itertools.cycle(all_bank_ids)
                prev_acct = None
                main_acct = None
                for _ in range(num_accounts):
                    sub_bank_id = next(bank_id_iter)
                    next_acct = random.sample(self.bank_to_accts_alert[type][sub_bank_id], 1)[0]
                    if prev_acct is None:
                        main_acct = next_acct
                    else:
                        add_edge(prev_acct, next_acct, amount, date, type)
                    self.remove_alert_candidate(next_acct, type)
                    add_node_attr(next_acct)
                    prev_acct = next_acct

            if is_part_external:
                main_acct, main_bank_id = get_main_acct()
                sub_accts = random.sample(self.bank_to_accts_alert[type][main_bank_id], \
                                          num_accounts-1-len(self.bank_to_accts_alert[type][main_bank_id]))  \
                                          + self.bank_to_accts_alert[type][main_bank_id]
                for n in sub_accts:
                    self.remove_alert_candidate(n, type)
                    add_node_attr(n)
                prev_acct = main_acct
                for _ in range(num_accounts - 1):
                    next_acct = random.choice([n for n in sub_accts if n != prev_acct])
                    add_edge(prev_acct, next_acct, amount, date, type)
                    prev_acct = next_acct

            logger.info("successfully generate alert subgraph: {}".format(type))

        elif type == "cycle":  # Cycle transactions
            is_part_external, is_total_external, is_internal = judge_external(type)

            amount = RandomAmount(min_amount, max_amount).getAmount()
            dates = sorted([random.randrange(start_date, end_date + 1) for _ in range(num_accounts)])

            if is_total_external:
                all_accts = list()
                #all_bank_ids = self.get_all_bank_ids()
                all_bank_ids = list(self.bank_to_accts_alert.keys())
                remain_num = num_accounts

                while all_bank_ids:
                    num_accts_per_bank = remain_num // len(all_bank_ids)
                    bank_id = all_bank_ids.pop()
                    new_members = random.sample(self.bank_to_accts_alert[type][bank_id], num_accts_per_bank)
                    all_accts.extend(new_members)

                    remain_num -= len(new_members)
                    for n in new_members:
                        self.remove_alert_candidate(n, type)
                        add_node_attr(n)
                main_acct = all_accts[0]
            else:
                main_acct, main_bank_id = get_main_acct()
                sub_accts = random.sample(self.bank_to_accts_alert[type][main_bank_id], num_accounts - 1)
                for n in sub_accts:
                    self.remove_alert_candidate(n, type)
                    add_node_attr(n)
                all_accts = [main_acct] + sub_accts

            for i in range(num_accounts):
                orig_i = i
                bene_i = (i + 1) % num_accounts
                orig_acct = all_accts[orig_i]
                bene_acct = all_accts[bene_i]
                date = dates[i]

                add_edge(orig_acct, bene_acct, amount, date, type)
                margin = amount * self.margin_ratio  # Margin the beneficiary account can gain
                amount = amount - margin  # max(amount - margin, min_amount)

            logger.info("successfully generate alert subgraph: {}".format(type))


        elif type == "scatter_gather":  # Scatter-Gather (fan-out -> fan-in)
            is_part_external, is_total_external, is_internal = judge_external(type)

            if is_total_external:
                if len(list(self.bank_to_accts_alert[type])) >= 3:
                    [sub_bank_id, mid_bank_id, bene_bank_id] = random.sample(list(self.bank_to_accts_alert[type]), 3)
                else:
                    [sub_bank_id, mid_bank_id] = random.sample(list(self.bank_to_accts_alert[type]), 2)
                    bene_bank_id = sub_bank_id
            else:
                sub_bank_id = mid_bank_id = bene_bank_id = random.sample(list(self.bank_to_accts_alert[type]), 1)[0]

            main_acct = orig_acct = random.sample(self.bank_to_accts_alert[type][sub_bank_id], 1)[0]
            self.remove_alert_candidate(orig_acct,type)
            add_node_attr(orig_acct)
            mid_accts = random.sample(self.bank_to_accts_alert[type][mid_bank_id], num_accounts - 2)
            for n in mid_accts:
                self.remove_alert_candidate(n,type)
                add_node_attr(n)
            bene_acct = random.sample(self.bank_to_accts_alert[type][bene_bank_id], 1)[0]
            self.remove_alert_candidate(bene_acct,type)
            add_node_attr(bene_acct)

            # The date of all scatter transactions must be performed before middle day
            mid_date = (start_date + end_date) // 2

            for i in range(len(mid_accts)): 
                mid_acct = mid_accts[i]
                scatter_amount = RandomAmount(min_amount, max_amount).getAmount()
                margin = scatter_amount * self.margin_ratio  # Margin of the intermediate account
                amount = scatter_amount - margin
                scatter_date = random.randrange(start_date, mid_date)
                gather_date = random.randrange(mid_date, end_date + 1)

                add_edge(orig_acct, mid_acct, scatter_amount, scatter_date, type)
                add_edge(mid_acct, bene_acct, amount, gather_date, type)

            logger.info("successfully generate alert subgraph: {}".format(type))


        elif type == "gather_scatter":  # Gather-Scatter (fan-in -> fan-out)
            is_part_external, is_total_external, is_internal = judge_external(type)

            if is_total_external:
                if len(list(self.bank_to_accts_alert[type])) >= 3:
                    [sub_bank_id, mid_bank_id, bene_bank_id] = random.sample(list(self.bank_to_accts_alert[type]), 3)
                else:
                    [sub_bank_id, mid_bank_id] = random.sample(list(self.bank_to_accts_alert[type]), 2)
                    bene_bank_id = sub_bank_id
            else:
                sub_bank_id = mid_bank_id = bene_bank_id = random.sample(list(self.bank_to_accts_alert[type]), 1)[0]

            num_orig_accts = num_bene_accts = (num_accounts - 1) // 2

            orig_accts = random.sample(self.bank_to_accts_alert[type][sub_bank_id], num_orig_accts)
            for n in orig_accts:
                self.remove_alert_candidate(n,type)
                add_node_attr(n)
            main_acct = mid_acct = random.sample(self.bank_to_accts_alert[type][mid_bank_id], 1)[0]
            self.remove_alert_candidate(mid_acct,type)
            add_node_attr(mid_acct)
            bene_accts = random.sample(self.bank_to_accts_alert[type][bene_bank_id], num_bene_accts)
            for n in bene_accts:
                self.remove_alert_candidate(n,type)
                add_node_attr(n)

            accumulated_amount = 0.0
            mid_date = (start_date + end_date) // 2
            amount = RandomAmount(min_amount, max_amount).getAmount()

            for i in range(num_orig_accts):
                orig_acct = orig_accts[i]
                date = random.randrange(start_date, mid_date)
                add_edge(orig_acct, mid_acct, amount, date, type)
                accumulated_amount += amount
                # print(orig_acct, "->", date, "->", mid_acct)

            for i in range(num_bene_accts):
                bene_acct = bene_accts[i]
                date = random.randrange(mid_date, end_date + 1)
                add_edge(mid_acct, bene_acct, amount, date, type)
                # print(mid_acct, "->", date, "->", bene_acct)
            # print(orig_accts, mid_acct, bene_accts)

            logger.info("successfully generate alert subgraph: {}".format(type))

        # TODO: Please add user-defined typology implementations here

        else:
            logger.warning("- Unknown AML typology name: %s" % type)
            return

        sub_g.graph[MAIN_ACCT_KEY] = main_acct  # Main account ID
        sub_g.graph[TYPE_KEY] = type
        sub_g.graph[IS_SAR_KEY] = is_sar
        sub_g.graph[IS_ALERT_KEY] = True
        self.alert_groups[self.alert_id] = sub_g
        self.alert_id += 1 
        logger.info("alert_id: {}".format(self.alert_id))

    def write_account_list(self):
        os.makedirs(self.output_dir, exist_ok=True)
        acct_file = os.path.join(self.output_dir, self.out_account_file)
        with open(acct_file, "w") as wf:
            writer = csv.writer(wf)
            base_attrs = ["ACCOUNT_ID", "CUSTOMER_ID", "INIT_BALANCE", "COUNTRY",
                          "ACCOUNT_TYPE", "IS_SAR", "BANK_ID"]
            writer.writerow(base_attrs + self.attr_names)
            for n in self.g.nodes(data=True):
                aid = n[0]  # Account ID
                cid = "C_" + str(aid)  # Customer ID bounded to this account
                prop = n[1]  # Account attributes
                balance = "{0:.2f}".format(prop["init_balance"])  # Initial balance
                region = prop["region"]  # Country
                business = prop["business"]  # Business type
                is_sar = "true" if IS_SAR_KEY in prop and prop[IS_SAR_KEY] else "false"  # Whether this account is involved in SAR
                bank_id = prop["bank_id"]  # Bank ID
                values = [aid, cid, balance, region, business, is_sar, bank_id]
                for attr_name in self.attr_names:
                    values.append(prop[attr_name])
                writer.writerow(values)
        logger.info("Exported %d accounts to %s" % (self.g.number_of_nodes(), acct_file))

    def write_transaction_list(self):
        tx_file = os.path.join(self.output_dir, self.out_tx_file)
        with open(tx_file, "w") as wf:
            writer = csv.writer(wf)
            writer.writerow(["id", "src", "dst"])
            for e in self.g.edges(data=True):
                src = e[0]
                dst = e[1]
                attr = e[2]
                tid = attr['edge_id']
                writer.writerow([tid, src, dst])
        with open(tx_file.split(".")[0]+"_noindex."+tx_file.split(".")[1], "w") as wf:
            writer = csv.writer(wf)
            writer.writerow(["id", "src", "dst"])
            for e in self.g.edges(data=True):
                src = e[0]
                dst = e[1]
                attr = e[2]
                tid = attr['edge_id']
                writer.writerow([src, dst])
        logger.info("Exported %d transactions to %s" % (self.g.number_of_edges(), tx_file))

    def write_alert_account_list(self):
        def get_out_edge_attrs(g, vid, name):
            return [v for k, v in nx.get_edge_attributes(g, name).items() if (k[0] == vid or k[1] == vid)]

        acct_count = 0
        alert_member_file = os.path.join(self.output_dir, self.out_alert_member_file)
        logger.info("Output alert member list to: " + alert_member_file)
        with open(alert_member_file, "w") as wf:
            writer = csv.writer(wf)
            base_attrs = ["alertID", "reason", "accountID", "isMain", "isSAR", "modelID",
                          "minAmount", "maxAmount", "startStep", "endStep", "bankID"]
            writer.writerow(base_attrs + self.attr_names)
            for gid, sub_g in self.alert_groups.items():
                main_id = sub_g.graph[MAIN_ACCT_KEY]
                model_id = sub_g.graph["type_id"]
                reason = sub_g.graph["reason"]
                start = sub_g.graph["start"]
                end = sub_g.graph["end"]
                for n in sub_g.nodes():
                    is_main = "true" if n == main_id else "false"
                    is_sar = "true" if sub_g.graph[IS_SAR_KEY] else "false"
                    min_amt = '{:.2f}'.format(min(get_out_edge_attrs(sub_g, n, "amount")))
                    max_amt = '{:.2f}'.format(max(get_out_edge_attrs(sub_g, n, "amount")))
                    min_step = start
                    max_step = end
                    bank_id = sub_g.nodes[n]["bank_id"]
                    values = [gid, reason, n, is_main, is_sar, model_id, min_amt, max_amt,
                              min_step, max_step, bank_id]
                    prop = self.g.nodes[n]
                    for attr_name in self.attr_names:
                        values.append(prop[attr_name])
                    writer.writerow(values)
                    acct_count += 1

        logger.info("Exported %d members for %d AML typologies to %s" %
                    (acct_count, len(self.alert_groups), alert_member_file))

    def write_normal_models(self):
        output_file = os.path.join(self.output_dir, self.out_normal_models_file)
        with open(output_file, "w") as wf:
            writer = csv.writer(wf)
            column_headers = ["modelID", "type", "accountID", "isMain", "isSAR"]
            writer.writerow(column_headers)
            
            acct_count = 0
            for g_id, normal_model in self.normal_groups.items():
                for account_id in normal_model.nodes:
                    #node_attr = normal_model.node[account_id]
                    graph_attr = normal_model.graph
                    values = [g_id, graph_attr["TYPE"], account_id, account_id==graph_attr[MAIN_ACCT_KEY], False]
                    writer.writerow(values)    
                    acct_count += 1
    
        logger.info("Exported %d members for %d normal typologies to %s" %
                    (acct_count, len(self.normal_groups), output_file))    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
