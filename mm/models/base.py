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
from lion_pytorch import Lion
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
        self.teacher_forcing_ratio = 0.99

    def forward(self, x):
        pass

    def configure_optimizers(self):
        """Optimizers"""
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        optimizer = Lion(self.parameters(), lr=self.cfg["TRAIN"]["LR"],
                weight_decay=1e-2)
        #scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer,
        #    step_size=self.cfg["TRAIN"]["LR_EPOCH"],
        #    gamma=self.cfg["TRAIN"]["LR_DECAY"],
        #)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #        optimizer, 49, eta_min=1e-5, verbose=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=3, verbose=True)
        #return [optimizer]#, [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1, #"indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step including logging

        Args:
            batch (dict): A dict with a batch of training samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        fut_seq = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        input_data = batch["input"]
        target_output = batch["target_output"][:,:fut_seq]
        pred_output = self.forward(input_data, target_output,
                self.teacher_forcing_ratio)
        self.teacher_forcing_ratio = self.teacher_forcing_ratio/1.0001
        assert not torch.any(torch.isnan(pred_output))
        assert not torch.any(torch.isnan(target_output))
        loss = self.loss(pred_output, target_output, target_output.shape[0])
        self.log("train/loss", loss["loss"], sync_dist=True,
                prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning validation step including logging

        Args:
            batch (dict): A dict with a batch of validation samples
            batch_idx (int): Index of batch in dataset

        Returns:
            None
        """
        fut_seq = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        input_data = batch["input"]
        target_output = batch["target_output"][:,:fut_seq]
        pred_output = self.forward(input_data, target_output, 0.)
        #pred_output = self.forward(input_data)
        loss = self.loss(pred_output.flatten(), target_output.flatten(),
                target_output.shape[0], "val", self.current_epoch)

        self.log("val/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        fut_seq = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        input_data = batch["input"]
        target_output = batch["target_output"][:,:fut_seq]
        idx = batch["idx"]

        batch_size, n_inputs, n_future_steps, H, W = input_data.shape

        start = time.time()
        pred_output = self.forward(input_data, target_output, 0.)
        #pred_output = self.forward(input_data)
        inference_time = (time.time() - start) / batch_size
        loss = self.loss(pred_output, target_output,
                target_output.shape[0], "test", self.current_epoch)

        self.log("test/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)
        self.log("test/inference_time", inference_time, on_epoch=True)

        for b in range(len(idx)):
            for t in range(fut_seq):
                target_img = target_output[b,t,0].cpu()*255.
                pred_img = nn.Sigmoid()(pred_output[b,t,0]).cpu()*255.
                plt.subplot(121)
                plt.imshow(target_img)
                plt.title('GT')
                plt.subplot(122)
                plt.imshow(pred_img)
                plt.title('Pred')
                plt.show()
                plt.savefig(self.save_dir+str(idx[b].cpu().numpy())+str(t)+'.png')
        return loss

    #def on_test_epoch_end(self, outputs):
    #    pass
