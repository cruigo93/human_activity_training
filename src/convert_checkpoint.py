from pathlib import Path
from collections import OrderedDict
import os
import random as rn

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageFile

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import importlib
import click
import utils


@click.command()
@click.option("--checkpoint", "-w", help="Pytorch lightning checkpoint")
@click.option("--config_file", "-c", help="Training config yaml file")
def main(checkpoint: str, config_file: str):
    config = utils.load_yaml(config_file)
    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["CLASS"])(
        **config["MODEL"]["ARGS"])
    ckpt_dict = torch.load(checkpoint, map_location="cuda:0")
    best_model_weights = utils.load_pytorch_model(ckpt_dict['state_dict'])
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    main()
