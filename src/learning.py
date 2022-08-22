import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import numpy as np


class HumanLearner(pl.LightningModule):

    def __init__(self,
                 dataloaders,
                 model,
                 optimizer,
                 scheduler,
                 criterion
                 ) -> None:

        super(HumanLearner, self).__init__()

        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def loss_fn(self, pred, y):
        # print(pred, y)
        return self.criterion(pred, y)

    def training_step(self, batch, batch_idx):
        X, y = batch["image"], batch["label"]
        pred = self(X)
        loss = self.loss_fn(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        X, y = batch["image"], batch["label"]
        pred = self(X)
        probs = torch.softmax(pred, axis=1)
        # y_pred = torch.argmax(probs, dim=1, keepdim=True)

        loss = self.loss_fn(pred, y)
        # y_gt = torch.
        roc_auc = roc_auc_score(
            y.cpu().numpy(), probs.cpu().numpy(), average="micro", multi_class="ovr")
        # eval_list = [[probs.cpu(), y.cpu()]]

        return {"valid_loss": loss, "valid_rocauc": roc_auc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        avg_metric = np.stack([x['valid_rocauc'] for x in outputs]).mean()
        # y_gt = np.
        self.log("avg_valid_loss", avg_loss)
        self.log("val_score", avg_metric)
