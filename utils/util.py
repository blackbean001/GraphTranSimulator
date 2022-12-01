# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:23:15 2022

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:17:40 2022

@author: Administrator
"""
import pickle
import json
import pandas as pd
import networkx as nx
from texttable import Texttable
import matplotlib.pyplot as plt

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def membership_saver(membership_path, membership):
    """
    Saving the membership dictionary as a JSON.
    :param membership_path: Path to save the JSON.
    :param membership: Membership dictionary with cluster ids.
    """
    with open(membership_path, "w") as f:
        json.dump(membership, f)

def get_positive_or_none(value):
    """ Get positive value or None (used to parse simulation step parameters)
    :param value: Numerical value or None
    :return: If the value is positive, return this value. Otherwise, return None.
    """
    if value is None:
        return None
    else:
        return value if value > 0 else None

def parse_int(value):
    """ Convert string to int
    :param value: string value
    :return: int value if the parameter can be converted to str, otherwise None
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def parse_float(value):
    """ Convert string to amount (float)
    :param value: string value
    :return: float value if the parameter can be converted to float, otherwise None
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_flag(value):
    """ Convert string to boolean (True or false)
    :param value: string value
    :return: True if the value is equal to "true" (case-insensitive), otherwise False
    """
    return type(value) == str and value.lower() == "true"


def write_to_csv(T_generator):
    # write to csv
    T_generator.write_account_list()
    T_generator.write_transaction_list()
    T_generator.write_alert_account_list()
    T_generator.write_normal_models()

def write_to_pickle(T_generator, path):
    # write to pickle
    with open(path,'wb') as f:
        pickle.dump(T_generator, f)
        #nx.write_gpickle(T_generator.g, path)

def run_test(T_generator,normal_num=100,alert_num=10):
    G = nx.grid_2d_graph(3,1)
    pos = nx.spring_layout(G,iterations=100)

    # test
    g = T_generator.g
    plt.figure(1)
    plt.title("the whole graph")
    nx.draw(g)
    
    plt.figure(2)
    plt.title("normal pattern")
    normal_dict = T_generator.normal_groups
    alert_dict = T_generator.alert_groups
    G = normal_dict[normal_num]
    nx.draw(G)
    
    plt.figure(3)
    plt.title("alert pattern")
    G = alert_dict[alert_num]
    nx.draw(G)  


