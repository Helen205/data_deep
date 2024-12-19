import os
import sys
import subprocess
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Safely clone repositories
def safe_git_clone(repo_url, directory):
    if not os.path.exists(directory):
        subprocess.run(['git', 'clone', repo_url, directory], check=True)
    else:
        print(f"{directory} already exists. Skipping clone.")

# Clone repositories
safe_git_clone('https://github.com/Helen205/data_deep.git', 'data_deep')

# Add Informer2020 to system path if not already added
if 'Informer2020' not in sys.path:
    sys.path.append('Informer2020')

from utils.tools import dotdict

from exp.exp_informer import Exp_Informer




# Initialize args
args = dotdict()
num_features = 14

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Data parser dictionary
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'custom':{'data':'custom.csv','T':'Close','M':[14,14,14],'S':[1,1,1],'MS':[14,14,1]}
}



args.model = 'informer'
args.data = 'custom'  # Changed from 'AKBNK_veri'
args.root_path = './data_deep/data/'
args.data_path = 'custom.csv'
args.features = 'MS'
args.target = 'Close'
args.freq = 'd'
args.checkpoints = './checkpoints/'
args.seq_len = 30
args.label_len = 15
args.pred_len = 7
args.enc_in = 14
args.dec_in = 14
args.c_out = 1
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.s_layers = '3,2,1'
args.d_ff = 2048
args.factor = 5
args.padding = 0
args.distil = True
args.dropout = 0.05
args.attn = 'prob'
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.do_predict = False
args.mix = True
args.cols = None
args.num_workers = 0
args.itr = 1
args.train_epochs = 50
args.batch_size = 32
args.patience = 10
args.learning_rate = 0.0001
args.des = 'test'
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False
args.inverse = False
args.use_gpu = torch.cuda.is_available()
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.lradj = 'type1'
args.criterion = 'mse'  # Mean Squared Error
args.optimizer = 'adam'
#args.patience = 3
args.des = 'exp'

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]