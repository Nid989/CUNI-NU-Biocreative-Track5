import pandas as pd
import numpy as np

import yaml
import argparse
from munch import Munch

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

# load csv data
def load_dataset(file_name):
    return pd.read_csv(file_name)

# drop unnecessary columns
def preprocess_data(Dataframe):
    return Dataframe.drop(['pmid',	'journal',	'title', 'doi',	'label', 'keywords', 'pub_type', 'authors'], axis=1)
    
# split data 
def Train_Test_split(Dataframe):
    train_df, val_df = train_test_split(Dataframe, test_size=0.1)
    return train_df, val_df

# configure arguments
def parse_args(args, **kwargs):
    args = Munch(**args)
    kwargs = Munch(**kwargs)
    args.data = kwargs.data
    args.device = 'cuda' if torch.cuda.is_available() and kwargs.no_cuda else 'cpu'
    if args.device == 'cuda':
        args.no_gpus = 1
    else:
        args.no_gpus = 0
    return args

def load_config_data(file_path):

    with open(file_path) as f:
        params = yaml.safe_load(f)

    return params
