# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:03:44 2022

@author: Administrator
"""

import sys

# EdMot
sys.path.append("/home/lisong/algorithms/graphtranssimulator/algo/EdMot/src")
from edmot import EdMot
from param_parser import parameter_parser
#from utils import tab_printer, graph_reader, membership_saver

# zora
sys.path.append('/home/lisong/algorithms/graphtranssimulator/algo/zora_script/')
from zora_script import *

# EvolveGCN
sys.path.append('/home/lisong/algorithms/graphtranssimulator/algo/EvolveGCN')
from generate_data import *
from run_EGCN import *