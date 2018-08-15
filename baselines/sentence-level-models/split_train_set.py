__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset, CVDataset, get_cv_dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import json
import math
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/json')
args = parser.parse_args()
ratio=0.1

with open('%s/train.json' % args.data_dir, 'r') as f:
	all_instances = json.load(f)
datasize = len(all_instances)
indices = list(range(datasize))
random.shuffle(indices)
dev_size = math.ceil(datasize * ratio)
dev_instances = [all_instances[i] for i in indices[:dev_size]]
train_instances = [all_instances[i] for i in indices[dev_size:]]
with open('%s/dev_split.json' % args.data_dir, 'w') as f:
	json.dump(dev_instances, f)
with open('%s/train_split.json' % args.data_dir, 'w') as f:
	json.dump(train_instances, f)