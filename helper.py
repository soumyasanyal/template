import numpy as np, sys, os, shutil, struct, argparse, csv, math, uuid, jsonlines, types, getpass
import time, socket, logging, itertools, json
import pickle5 as pickle

from argparse import ArgumentParser
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional
from functools import lru_cache
from collections import defaultdict as ddict
from collections import OrderedDict, Counter
from typing import Dict, List, NamedTuple, Optional
from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import NeptuneLogger
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from datasets import load_dataset
from transformers import (
	AdamW,
	AutoModelForSequenceClassification,
	AutoModelForMultipleChoice,
	AutoModel,
	AutoConfig,
	AutoTokenizer,
	get_scheduler,
	get_linear_schedule_with_warmup,
	glue_compute_metrics
)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize


def freeze_net(module):
	for p in module.parameters():
		p.requires_grad = False

def unfreeze_net(module):
	for p in module.parameters():
		p.requires_grad = True


def clear_cache():
	torch.cuda.empty_cache()

def get_username():
	return getpass.getuser()
