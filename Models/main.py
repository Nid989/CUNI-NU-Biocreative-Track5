import pandas as pd
import numpy as np

import yaml
import argparse
from munch import Munch

from tqdm import tqdm

import torch
import torch.nn as nnx
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers.tokenization_utils_base import LARGE_INTEGER

from utils import *
from models import TopicAnnotationDataset, TopicAnnotationDataModule, TopicAnnotationTagger

def train(args):

    tokenizer = BertTokenizer.from_pretrained(args.BERT_MODEL_NAME)
    
    # load and split data 
    data = load_dataset(args.data)
    data = preprocess_data(data)
    train_df, val_df = Train_Test_split(data)
    LABEL_COLUMNS = args.LABEL_COLUMNS

    # create Data module 
    data_module = TopicAnnotationDataModule(
        train_df,
        val_df,
        tokenizer,
        batch_size=args.BATCH_SIZE,
        max_token_len=args.MAX_TOKEN_COUNT
    )

    # define total_training_steps and warmup_steps
    steps_per_epoch=len(train_df) // args.BATCH_SIZE
    total_training_steps = steps_per_epoch * args.N_EPOCHS
    warmup_steps = total_training_steps // 5
    
    # define model
    model = TopicAnnotationTagger(
        n_classes=len(LABEL_COLUMNS),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    # model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoints",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # define logger 
    logger = TensorBoardLogger("lightning_logs", name="topic-annotations")

    # early stopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    # define pl.Trainer
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.N_EPOCHS,
        gpus=args.no_gpus,
        progress_bar_refresh_rate=30
    )

    # run model
    trainer.fit(model, data_module)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default='settings\config.yml', help='path to yaml config file', type=argparse.FileType('r'))
    parser.add_argument('--data', default='dataset\data\litcovid_dataset.csv', type=str, help='path to dataset csv file')
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')

    parsed_args = parser.parse_args()

    with parsed_args.config as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(Munch(params), **vars(parsed_args))
    # start training
    train(args)
    