#!/usr/bin/env python3
# @brief:    Generic base class for lightning
# @author    Kaustab Pal    [kaustab21@gmail.com]

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import lightning.pytorch as pl
from mm.models.loss import Loss

class BaseModel(pl.LightningModule):
    """Pytorch Lightning base model"""

    def __init__(self, cfg):
        """Init base model

        Args:
            cfg (dict): Config parameters
        """
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.loss = Loss(self.cfg)
        self.save_dir = self.cfg["LOG_DIR"]+"/predictions/"
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        """Optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg["TRAIN"]["LR_EPOCH"],
            gamma=self.cfg["TRAIN"]["LR_DECAY"],
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step including logging

        Args:
            batch (dict): A dict with a batch of training samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        input_data = batch["input"]
        target_output = batch["target_output"][:,0]
        pred_output = self.forward(input_data)
        assert not torch.any(torch.isnan(pred_output))
        assert not torch.any(torch.isnan(target_output))
        loss = self.loss(pred_output, target_output)
        self.log("train/loss", loss["loss"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning validation step including logging

        Args:
            batch (dict): A dict with a batch of validation samples
            batch_idx (int): Index of batch in dataset

        Returns:
            None
        """
        input_data = batch["input"]
        target_output = batch["target_output"][:,0]
        pred_output = self.forward(input_data)
        loss = self.loss(pred_output, target_output, "val", self.current_epoch)

        self.log("val/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        input_data = batch["input"]
        target_output = batch["target_output"][:,0]
        idx = batch["idx"]

        batch_size, n_inputs, n_future_steps, H, W = input_data.shape

        start = time.time()
        pred_output = self.forward(input_data)
        inference_time = (time.time() - start) / batch_size
        self.log("test/inference_time", inference_time, on_epoch=True)

        loss = self.loss(pred_output, target_output, "test", self.current_epoch)

        self.log("test/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)
        for b in range(len(idx)):
            target_img = target_output[b,0].cpu()*255.
            pred_img = pred_output[b,0].cpu()*255.
            plt.subplot(121)
            plt.imshow(target_img)
            plt.title('GT')
            plt.subplot(122)
            plt.imshow(pred_img)
            plt.title('Pred')
            plt.show()
            plt.savefig(self.save_dir+str(idx[b].cpu())+'.png')
        return loss

    #def on_test_epoch_end(self, outputs):
    #    pass
