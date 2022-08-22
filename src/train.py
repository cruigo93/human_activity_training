import click
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


import importlib

import utils
import datasets
import learning


@click.command()
@click.option("--config_file", "-c", help="Training config yaml file")
def train(config_file: str):
    utils.seed_everything(42)
    config = utils.load_yaml(config_file)
    df = pd.read_csv(config["DATA"]["FILE"])
    items = df.to_dict("records")
    train_items, val_items = train_test_split(
        items, test_size=config["DATA"]["VAL_SIZE"])
    logger.info(
        f"Trainig size: {len(train_items)}\t Validation size: {len(val_items)}")

    module = importlib.import_module(config["AUGMENTATION"]["PY"])
    train_aug = getattr(module, config["AUGMENTATION"]["TRAIN"])()
    val_aug = getattr(module, config["AUGMENTATION"]["VAL"])()

    train_dataset = datasets.HumanDataset(
        config["DATA"]["IMAGE_DIR"], train_items, train_aug)
    val_dataset = datasets.HumanDataset(
        config["DATA"]["IMAGE_DIR"], val_items, val_aug)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=config["DATA"]["BATCH_SIZE"], shuffle=True),
        "val": DataLoader(val_dataset, batch_size=config["DATA"]["BATCH_SIZE"], shuffle=False)
    }

    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["CLASS"])(
        **config["MODEL"]["ARGS"])

    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(
        model.parameters(), **config["OPTIMIZER"]["ARGS"])

    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(
        optimizer, **config["SCHEDULER"]["ARGS"])

    module = importlib.import_module(config["CRITERION"]["PY"])
    criterion = getattr(module, config["CRITERION"]["CLASS"])()

    learner = learning.HumanLearner(
        dataloaders, model, optimizer, scheduler, criterion)

    grad_clip = config["GRADIENT_CLIPPING"]
    grad_acum = config["GRADIENT_ACCUMULATION_STEPS"]

    callbacks = []
    early_stop_callback = EarlyStopping(
        **config["EARLY_STOPPING"]["ARGS"]
    )
    callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        **config["CHECKPOINT"]["ARGS"]
    )
    callbacks.append(checkpoint_callback)
    tensorboard_logger = TensorBoardLogger(
        "logs/"+config["EXPERIMENT_NAME"], name=config["EXPERIMENT_NAME"])

    trainer = pl.Trainer(gpus=config["GPUS"],
                         max_epochs=config["EPOCHS"],
                         num_sanity_val_steps=0,
                         #  limit_train_batches=0.01,
                         #  limit_val_batches=0.2,
                         logger=tensorboard_logger,
                         gradient_clip_val=grad_clip,
                         accumulate_grad_batches=grad_acum,
                         precision=16,
                         callbacks=callbacks)
    trainer.fit(learner)


if __name__ == "__main__":
    train()
