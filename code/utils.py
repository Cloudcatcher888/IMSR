from typing import DefaultDict
import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss


import os
import random


# configration
class Config():
    min_timestamp -= 100
	max_timestamp += 100
	task_num = 6
	base_task_ratio = 0.85
	item_cate_file = 	'/home/wangzhikai/recommendation_learning/dataset/amazon/electronics_item_cate.txt'
	train_file = '/home/wangzhikai/recommendation_learning/dataset/amazon/electronics_test.txt'
	current_user_group = -1
	min_timestamp = 1000000000000
	max_timestamp = 0
	cate_num = 0
	merge_num = 1


def cuda_setting():
	memory_needed = 2000
	device = torch.device('cpu')
	import pynvml
	pynvml.nvmlInit()
	if torch.cuda.is_available():
	    for i in range(4): 
	        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
	        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
	        free_mem = float(meminfo.free/1024**2)
	        print(free_mem)
	        if free_mem > memory_needed:
 	           device = torch.device(f'cuda:{i}')
	            break
	# device = torch.device('cpu')
	print(f'use {device}')
	random.seed(1) 
	np.random.seed(1) 
	torch.manual_seed(1) 
	torch.cuda.manual_seed(1) 
	torch.cuda.manual_seed_all(1)
    
