import networkx as nx
import random
import traceback
import copy
from utils import *

class Nominator:
    def __init__(self, g, threshold, logger):
        self.g = g
        self.hubs = dict() 
        self.threshold = threshold
        self.remaining_count_dict = dict()
        self.done_count_dict = dict()
        self.bank_to_accts = dict()
        self.acct_to_bank = dict()
        self.continuous_failure_counts = dict()
        self.edge_index = {}
        for edge in list(self.g.edges()):
            self.edge_index[edge] = 0
        
        self.fan_in_index = 0
        self.fan_out_index = 0
        self.forward_index = 0
        self.single_index = 0
        self.mutual_index = 0
        self.periodical_index = 0    
        
        self.logger = logger

    def initialize_type_counts(self, reader, hubs, acct_to_bank, bank_to_accts, special_token="#"):
        header = next(reader)

        for row in reader:
            if len(row)==0 or row[0].startswith(special_token):
                  continue
            count = int(row[header.index("count")])
            type = row[header.index("type")]
            
            self.remaining_count_dict[type] = count
            self.done_count_dict[type] = 0
            self.hubs[type] = copy.deepcopy(hubs)
            self.bank_to_accts[type] = copy.deepcopy(bank_to_accts)
            self.acct_to_bank[type] = copy.deepcopy(acct_to_bank)
            self.continuous_failure_counts[type] = 0
                    
    def remove_candidate(self, acct, type):
        self.hubs[type].discard(acct)
        bank_id = self.acct_to_bank[type][acct]
        del self.acct_to_bank[type][acct]
        self.bank_to_accts[type][bank_id].discard(acct)
        
    def add_candidate(self, acct, bank_id, type):
        self.hubs[type].add(acct)
        self.acct_to_bank[type][acct] = bank_id
        self.bank_to_accts[type][bank_id].add(acct)        
    
    def add_node_attr(self, _acct):
        attr_dict = self.g.nodes[_acct]
        self.sub_g.add_node(_acct, **attr_dict)
        
    def assign_edge(self, _orig, _bene, _amount, _date, _type):
        add_attr = {"amount":_amount, "date":_date, "type": _type, IS_NORMAL_KEY:True}
        self.g[_orig][_bene][self.edge_index[(_orig,_bene)]].update(add_attr)
        self.sub_g.add_edge(_orig, _bene, amount=_amount, date=_date, IS_NORMAL_KEY=True)
        self.edge_index[(_orig,_bene)] += 1
    
    def init_sub_g(self, sub_g):
        self.sub_g = sub_g
        
    def get_main_acct(self, num_accounts, bank_id, type):
        """Create a main account ID and a bank ID from hub accounts
        :param: sub_g is refreshed everytime when function get_main_acct() is called
        :return: main account ID and bank ID
        """

        if not self.hubs[type]:
            
            raise ValueError("No main account candidates found from hub accounts for bank_id {}".format(bank_id))
            
        #candidates = [node for node in self.hubs if self.acct_to_bank[type][node]==bank_id 
        #              and len([n for n in self.g.neigbors(node) if n in self.acct_to_bank[type]]) >= num_accounts-1]

        candidates = [n for n in self.hubs[type] if self.g.nodes[n]["bank_id"]==bank_id]   
        if candidates != []:
            _main_acct = random.sample(candidates,1)[0]
            _main_bank_id = bank_id
        else:
            _main_acct = random.sample(self.hubs[type], 1)[0]
            _main_bank_id = self.acct_to_bank[type][_main_acct]
            self.logger.warning("no hub accounts found for bank_id {}, choose from bank_id {} instead".format(bank_id, _main_bank_id))
        self.remove_candidate(_main_acct, type)
        self.add_node_attr(_main_acct)
        return _main_acct, _main_bank_id

    def assign_remaining_single(self,min_amount,max_amount,start_date,end_date,normal_types):
        sub_g_list = []
        edge_index = {}
        for (_ori, _bene) in self.g.edges():
            if (_ori, _bene) not in edge_index:
                edge_index[(_ori,_bene)] = 0
            if "amount" not in self.g[_ori][_bene][edge_index[(_ori,_bene)]]:                
                self.sub_g = nx.DiGraph(type_id = normal_types["single"], reason="single", start=start_date, end=end_date)
                amount = RoundedSampling(min_amount, max_amount).get()
                date = random.randrange(start_date, end_date + 1)   
                self.add_node_attr(_ori)
                self.add_node_attr(_bene)
                self.assign_edge(_ori, _bene, amount, date, "single")
                self.sub_g.graph[MAIN_ACCT_KEY] = _ori
                self.sub_g.graph[TYPE_KEY] = "single"
                self.sub_g.graph[IS_NORMAL_KEY] = True
                sub_g_list.append(self.sub_g)
            edge_index[(_ori,_bene)] += 1
        return sub_g_list

    def get_fan_in_subgraph(self, main_node, num_accounts, main_bank_id,  start_date, end_date, min_amount, max_amount):
        type = "fan_in"        
        
        (is_total_external, is_part_external, is_internal), sub_acct_candidates = \
                           self.get_fan_in_candidates(main_node, num_accounts, main_bank_id)
        
        if sub_acct_candidates == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        # find sub_accts
        if is_part_external:
            sub_accts = sub_acct_candidates
        else:
            sub_accts = random.sample(sub_acct_candidates, num_accounts-1)

        #for n in sub_accts:
        #    self.remove_candidate(n, type)

        for orig in sub_accts:
            self.add_node_attr(orig)
            amount = RoundedSampling(min_amount, max_amount).get()
            date = random.randrange(start_date, end_date + 1)
            self.assign_edge(orig, main_node, amount, date, type)
        
        self.fan_in_index += 1
        
        return self.sub_g
    
    def check_edge_valid(self, _orig, _bene):
        return len(self.g[_orig][_bene])>self.edge_index[(_orig,_bene)]
        
    def get_fan_in_candidates(self, main_node, num_accounts, main_bank_id):
        is_total_external, is_part_external, is_internal = False, False, False
        type = "fan_in"
        
        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type] \
            and self.check_edge_valid(n,main_node)]) < num_accounts-1:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))
            return (None, None, None), None
            
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type])>=2:
            if main_bank_id == "":
                is_total_external = True
            elif main_bank_id != "" and len([n for n in self.g.predecessors(main_node) if n  \
                in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(n,main_node)]) < num_accounts-1:            
                is_part_external = True
            else:
                is_internal = True
        else:
            is_internal = True
        
        # find sub_acct_candidates
        if is_total_external:
            sub_acct_candidates = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                   and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(n,main_node)]
        if is_part_external:
            cand_A = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                      and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(n,main_node)]
            cand_B = random.sample([n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                    and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(n,main_node)], num_accounts-1-len(cand_A))
            sub_acct_candidates = cand_A + cand_B
        else:
            sub_acct_candidates = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                    and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(n,main_node)]

        return (is_total_external, is_part_external, is_internal), sub_acct_candidates

    def get_fan_out_subgraph(self, main_node, num_accounts, main_bank_id, start_date, end_date, min_amount, max_amount):
        type = "fan_out"        
        
        (is_total_external, is_part_external, is_internal), sub_acct_candidates = \
                           self.get_fan_out_candidates(main_node, num_accounts, main_bank_id)
        
        if sub_acct_candidates == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        if is_part_external:
            sub_accts = sub_acct_candidates
        else:
            sub_accts = random.sample(sub_acct_candidates, num_accounts-1)

        #for n in sub_accts:
        #    self.remove_candidate(n, type)

        for bene in sub_accts:
            self.add_node_attr(bene)
            amount = RoundedSampling(min_amount, max_amount).get()
            date = random.randrange(start_date, end_date + 1)
            self.assign_edge(main_node, bene, amount, date, type)
        
        self.fan_out_index += 1
        
        return self.sub_g
        
    def get_fan_out_candidates(self, main_node, num_accounts, main_bank_id):
        is_total_external, is_part_external, is_internal = False, False, False
        type = "fan_out"
        
        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type] \
                and self.check_edge_valid(main_node,n)]) < num_accounts-1:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))
            return (None, None, None), None
            
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type])>=2:
            if main_bank_id == "":
                is_total_external = True
            elif main_bank_id != "" and len([n for n in self.g.successors(main_node) if n  \
                                        in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(main_node,n)]) < num_accounts-1:            
                is_part_external = True
            else:
                is_internal = True
        else:
            is_internal = True
        
        # find sub_acct_candidates
        if is_total_external:
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                   and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(main_node,n)]
        if is_part_external:
            cand_A = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                      and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(main_node,n)]
            cand_B = random.sample([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                    and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(main_node,n)], \
                                   num_accounts-1-len(cand_A))
            sub_acct_candidates = cand_A + cand_B
        else:
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                   and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(main_node,n)]

        return (is_total_external, is_part_external, is_internal), sub_acct_candidates

    def get_forward_subgraph(self, main_node, num_accounts, main_bank_id, start_date, end_date, min_amount, max_amount):
        type = "forward"
        
        if num_accounts != 3:
            raise ValueError("The number of accounts of a forward pattern should be 3, instead of {}".format(num_accounts))
        
        (is_total_external, is_pred_external, is_succ_external, is_internal), (sub_acct_candidates_pred, sub_acct_candidates_succ) = \
                      self.get_forward_candidates(main_node, num_accounts, main_bank_id)
        
        if sub_acct_candidates_pred == None or sub_acct_candidates_succ == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        pred = random.sample(sub_acct_candidates_pred, 1)[0]
        succ = random.sample(sub_acct_candidates_succ, 1)[0]
        
        #for n in [pred, succ]:
        #    self.remove_candidate(n, type)
        
        
        self.add_node_attr(pred)
        self.add_node_attr(succ)
        
        dates = random.sample(range(start_date, end_date + 1),2)
        dates.sort()        
        amount1 = RoundedSampling(min_amount, max_amount).get()
        amount2 = RoundedSampling(min_amount, max_amount).get()
        dates.sort()
        self.assign_edge(pred, main_node, amount1, dates[0], type)
        self.assign_edge(main_node, succ, amount2, dates[1], type)
        
        self.forward_index += 1
        
        return self.sub_g
        
    def get_forward_candidates(self, main_node, num_accounts, main_bank_id):
        is_total_external, is_pred_external, is_succ_external, is_internal = False, False, False, False
        type = "forward"
        
        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)])==0   \
            or len([n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type] and self.check_edge_valid(n,main_node)])==0:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))                
            return (None, None, None), (None, None)
        
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type]) > 2:
            if main_bank_id == "":
                is_total_external = True
            elif main_bank_id != "":
                if len([n for n in self.g.predecessors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(n,main_node)]) == 0  \
                   and len([n for n in self.g.successors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(main_node,n)]) != 0:
                    is_pred_external = True
                elif len([n for n in self.g.predecessors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(n,main_node)]) != 0  \
                   and len([n for n in self.g.successors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(main_node, n)]) == 0:
                    is_succ_external = True
                elif len([n for n in self.g.predecessors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(n,main_node)]) == 0  \
                   and len([n for n in self.g.successors(main_node) if n in self.bank_to_accts[type][main_bank_id] and self.check_edge_valid(main_node,n)]) == 0:
                    is_total_external = True
                else:
                    is_internal = True
        else:
            is_internal = True
                
        # find sub_acct_candidates
        if is_total_external:
            sub_acct_candidates_pred = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(n,main_node)]
            sub_acct_candidates_succ = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(main_node,n)]            
        if is_pred_external:
            sub_acct_candidates_pred = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(n,main_node)]
            sub_acct_candidates_succ = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(main_node,n)]            
        if is_succ_external:
            sub_acct_candidates_pred = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(n,main_node)]
            sub_acct_candidates_succ = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(main_node,n)]
        if is_internal:
            sub_acct_candidates_pred = [n for n in self.g.predecessors(main_node) if n in self.acct_to_bank[type] \
                                        and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(n,main_node)]
            sub_acct_candidates_succ = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                        and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(main_node,n)]

        return (is_total_external, is_pred_external, is_succ_external, is_internal), (sub_acct_candidates_pred, sub_acct_candidates_succ)

    def get_mutual_subgraph(self, main_node, num_accounts, main_bank_id, start_date, end_date, min_amount, max_amount):
        type = "mutual"
        
        if num_accounts != 2:
            raise ValueError("The number of accounts of a mutual pattern should be 2, instead of {}".format(num_accounts))

        sub_acct_candidates = self.get_mutual_candidates(main_node, num_accounts, main_bank_id)
        if sub_acct_candidates == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        # find sub_accts
        acct = random.sample(sub_acct_candidates, 1)[0]
        
        #self.remove_candidate(acct, type)
        self.add_node_attr(acct)
        amount = RoundedSampling(min_amount, max_amount).get()
        dates = random.sample(range(start_date, end_date + 1),2)
        dates.sort()
        self.assign_edge(acct, main_node, amount, dates[0], type)
        self.assign_edge(main_node, acct, amount, dates[1], type)
        
        self.mutual_index += 1
        
        return self.sub_g
        
    def get_mutual_candidates(self, main_node, num_accounts, main_bank_id):
        is_external, is_internal = False, False
        type = "mutual"
        
        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.successors(main_node) if main_node in self.g.successors(n)  \
            and n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)  \
            and self.check_edge_valid(n,main_node)]) == 0:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))
            return None        
        
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type]) >= 2:
            if main_bank_id == "":
                is_external = True
            elif main_bank_id != "":
                if [n for n in self.g.successors(main_node) if main_node in self.g.successors(n)  \
                           and n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)  \
                           and self.check_edge_valid(n,main_node) and self.acct_to_bank[type][n]==main_bank_id] == 0:
                    is_external = True
                else:
                    is_internal = True
        else:
            is_internal = True

        # find sub_acct_candidates
        if is_external:           
            sub_acct_candidates = [n for n in self.g.successors(main_node) if main_node in self.g.successors(n)  \
                                   and n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)     \
                                   and self.check_edge_valid(n,main_node) and self.acct_to_bank[type][n] != main_bank_id]

        if is_internal:
            sub_acct_candidates = [n for n in self.g.successors(main_node) if main_node in self.g.successors(n)  \
                                   and n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)     \
                                   and self.check_edge_valid(n,main_node) and self.acct_to_bank[type][n] == main_bank_id]

        return sub_acct_candidates

    def get_single_subgraph(self, main_node, num_accounts, main_bank_id, start_date, end_date, min_amount, max_amount):
        type = "single"
        if num_accounts != 2:
            raise ValueError("The number of accounts of a periodical pattern should be 2, instead of {}".format(num_accounts))

        sub_acct_candidates = self.get_single_candidates(main_node, num_accounts, main_bank_id)
        if sub_acct_candidates == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        # find sub_accts
        acct = random.sample(sub_acct_candidates, 1)[0]
        self.add_node_attr(acct)
        #self.remove_candidate(acct, type)
        
        date = start_date
        amount = RoundedSampling(min_amount, max_amount).get()        
        self.assign_edge(main_node, acct, amount, date, type)        
        self.single_index += 1
        
        return self.sub_g
        
    def get_single_candidates(self, main_node, num_accounts, main_bank_id):
        is_external, is_internal = False, False
        type = "single"

        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type] and self.check_edge_valid(main_node,n)]) == 0:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))
            return None     
        
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type]) >= 2:
            if main_bank_id == "":
                is_external = True
            elif main_bank_id != "":
                if [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                    and self.acct_to_bank[type][n]==main_bank_id and self.check_edge_valid(main_node,n)] == 0:
                    is_external = True
                else:
                    is_internal = True
        else:
            is_internal = True

        # find sub_acct_candidates
        if is_external:           
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]      
                                   and self.acct_to_bank[type][n] != main_bank_id and self.check_edge_valid(main_node,n)]

        if is_internal:
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]      
                                   and self.acct_to_bank[type][n] == main_bank_id and self.check_edge_valid(main_node,n)]
        
        return sub_acct_candidates

    def get_periodical_subgraph(self, main_node, num_accounts, main_bank_id, start_date, end_date, min_amount, max_amount):
        type = "periodical"
        
        if num_accounts != 2:
            raise ValueError("The number of accounts of a periodical pattern should be 2, instead of {}".format(num_accounts))

        sub_acct_candidates = self.get_periodical_candidates(main_node, num_accounts, main_bank_id)
        if sub_acct_candidates == None:
            #self.add_candidate(main_node, main_bank_id, type)
            return
        
        # find sub_accts
        acct = random.sample(sub_acct_candidates, 1)[0]     
        self.add_node_attr(acct)    
        #self.remove_candidate(acct, type)
        
        date = start_date
        interval = float(int((end_date-start_date))/(len(self.g.edges([main_node,acct]))-self.edge_index[(main_node,acct)]))
        for i in range(len(self.g[main_node][acct])-self.edge_index[(main_node,acct)]):
            amount = RoundedSampling(min_amount, max_amount).get()            
            self.assign_edge(main_node, acct, amount, date, type)
            date += interval

        self.periodical_index += 1        
        return self.sub_g
        
    def get_periodical_candidates(self, main_node, num_accounts, main_bank_id):
        is_external, is_internal = False, False
        type = "periodical"
        
        # 先置校验：保证所有符合条件的节点总数 大于等于 num_accounts-1
        if len([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                and (len(self.g[main_node][n])-self.edge_index[(main_node,n)]>=3)]) == 0:
            self.logger.warning("- Not enough neighbouring accounts found for type:{} main_node:{}.".format(type, main_node))
            return None   
        
        # 后置校验：check whether external
        if main_bank_id != "" and main_bank_id not in self.bank_to_accts[type]:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % main_bank_id)
        if len(self.bank_to_accts[type]) >= 2:
            if main_bank_id == "":
                is_external = True
            elif main_bank_id != "":
                if len([n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                        and self.acct_to_bank[type][n]==main_bank_id  \
                        and (len(self.g[main_node][n])-self.edge_index[(main_node,n)]>=3)]) == 0:
                    is_external = True
                else:
                    is_internal = True
        else:
            is_internal = True
        
        # find sub_acct_candidates
        if is_external:           
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]  \
                                   and self.acct_to_bank[type][n]!=main_bank_id  \
                                   and (len(self.g[main_node][n])-self.edge_index[(main_node,n)]>=3)]

        if is_internal:
            sub_acct_candidates = [n for n in self.g.successors(main_node) if n in self.acct_to_bank[type]
                                   and self.acct_to_bank[type][n]==main_bank_id  \
                                   and (len(self.g[main_node][n])-self.edge_index[(main_node,n)]>=3)]

        return sub_acct_candidates





































