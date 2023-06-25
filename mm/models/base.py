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
import matplotlib as mpl
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
        pred_output, _ = self.forward(input_data, target_output,
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
        pred_output, _ = self.forward(input_data, target_output, 0.)
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

        batch_size, n_seq, chn, H, W = input_data.shape

        start = time.time()
        pred_output, attn_list = self.forward(input_data, target_output, 0.)
        #pred_output = self.forward(input_data)
        inference_time = (time.time() - start) / batch_size
        loss = self.loss(pred_output, target_output,
                target_output.shape[0], "test", self.current_epoch)

        self.log("test/loss", loss["loss"], sync_dist=True, prog_bar=True, on_epoch=True)
        self.log("test/inference_time", inference_time, on_epoch=True)
        cmap = mpl.colormaps["turbo"]
        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
        #cmap.set_under('k')

        for b in range(len(idx)):
            fig = plt.figure()
            fig.set_figheight(32)
            fig.set_figwidth(32)
            axes={}
            for i in range(26):
                # note that for some reason, add_subplot() counts from 1, hence we use i+1 here
                axes[i] = fig.add_subplot(2,13,i+1)
            t_step = 1
            for at in range(len(attn_list)):
                attn = attn_list[at]
                _, _, _, ah, aw = attn.shape
                attn = attn.view(batch_size*n_seq, chn, ah, aw)
                scaled_attn = F.interpolate(attn,(H, W), mode='bilinear', align_corners=False)
                scaled_attn = scaled_attn.view(batch_size, n_seq, chn, H, W)
                for i in range(input_data.shape[1]):
                    in_img = input_data[b,i,0].cpu()*255.
                    axes[i].imshow(in_img)
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                    axes[i].set_title('In')
                    axes[i].set_autoscale_on(False)
                    axes[i].axis('off')

                    axes[i+13].imshow(in_img)
                    axes[i+13].imshow(scaled_attn[b,i,0].cpu(),alpha=0.4,
                            cmap=cmap) #, norm=norm)
                    axes[i+13].set_xticks([])
                    axes[i+13].set_yticks([])
                    axes[i+13].set_title('In')
                    axes[i].set_autoscale_on(False)
                    axes[i].axis('off')

                for t in range(fut_seq):
                    target_img = target_output[b,t,0].cpu()*255.
                    pred_img = nn.Sigmoid()(pred_output[b,t,0]).cpu()*255.
                    #plt.subplot(2,input_data.shape[1]+fut_seq, i+t+1)
                    axes[i+1+t].imshow(target_img)
                    axes[i+1+t].set_xticks([])
                    axes[i+1+t].set_yticks([])
                    axes[i+1+t].set_title('Gt')
                    axes[i].set_autoscale_on(False)
                    axes[i].axis('off')
                    #axs.subplot(2,input_data.shape[1]+fut_seq,(i+t+1)+fut_seq)
                    axes[i+1+t+13].imshow(pred_img)
                    axes[i+1+t+13].set_xticks([])
                    axes[i+1+t+13].set_yticks([])
                    axes[i+1+t+13].set_title('pred')
                    axes[i].set_autoscale_on(False)
                    axes[i].axis('off')
                fig.subplots_adjust(hspace=0.5, wspace=0.2, left=0, bottom=0, right=1, top=1)
                r, c = 2, 13
                fig.set_figheight(fig.get_figwidth() * axes[i].get_data_ratio() * r / c )
                fig.show()
                #fig.tight_layout()
                local_dir = self.save_dir+str(idx[b].cpu().numpy())+"/"
                os.makedirs(local_dir, exist_ok=True)
                fig.savefig(local_dir+str(idx[b].cpu().numpy())+'_attn_'+str(at)+'.png',bbox_inches='tight') #, pad_inches=0)
        return loss

    #def on_test_epoch_end(self, outputs):
    #    pass
