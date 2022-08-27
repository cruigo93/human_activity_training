import click
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from tqdm import tqdm


import importlib

import transforms
import datasets
import utils


@click.command()
@click.option("--config_file", "-c", help="Training config yaml file")
def train(config_file: str):
    config = utils.load_yaml(config_file)
    df = pd.read_csv(config["TEST"]["FILE"])
    items = df.to_dict("records")
    test_aug = transforms.get_train_transforms()
    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["CLASS"])(
        **config["MODEL"]["ARGS"])
    model.load_state_dict(torch.load(config["MODEL"]["CHECKPOINT"]))
    model.eval()

    test_dataset = datasets.HumanDataset(
        config["TEST"]["DIR"], items, test_aug)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    filenames = [item["filename"] for item in items]
    y_preds = []
    with torch.no_grad():
        for batch, filename in tqdm(zip(test_loader, filenames), total=len(filenames)):
            X = batch["image"]
            pred = model(X)
            probs = torch.softmax(pred, axis=1)
            y_pred = torch.argmax(
                probs, dim=1, keepdim=True).squeeze().cpu().numpy()
            y_preds.append(datasets.CLASSES[y_pred])
    sub_df = pd.DataFrame(data={"filename": filenames, "label": y_preds})
    sub_df.to_csv("submission.csv", index=None)


if __name__ == "__main__":
    train()
