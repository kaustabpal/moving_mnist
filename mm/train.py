#!/usr/bin/env python3
# @brief:    Generic training script
# @author    Kaustab Pal  [kaustab21@gmail.com]

import os
import time
import argparse
import yaml
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies.ddp import DDPStrategy
import torch
import subprocess
from mm.datasets.datasets import MovingMnistModule
from mm.models.clstm_models import Many2One, One2Many
from mm.models.seq2seq import Seq2Seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        "--comment", "-c", type=str, default="", help="Add a comment to the LOG ID."
    )
    parser.add_argument(
        "-res",
        "--resume",
        type=str,
        default=None,
        help="Resume training from specified model.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Init model with weights from specified model",
    )
    parser.add_argument(
        "-e",
        "--epochs", type=int, default=None,
        help="Number of training epochs."
    )

    args, unparsed = parser.parse_known_args()

    model_path = args.resume if args.resume else args.weights
    if model_path:
        ###### Load config and update parameters
        checkpoint_path = "./mm/runs/" + model_path + "/checkpoints/last.ckpt"
        config_filename = "./mm/runs/" + model_path + "/Train/hparams.yaml"
        cfg = yaml.safe_load(open(config_filename))

        if args.weights and not args.comment:
            args.comment = "_pretrained"

        cfg["LOG_DIR"] = cfg["LOG_DIR"] + args.comment
        cfg["LOG_NAME"] = cfg["LOG_NAME"] + args.comment
        print("New log dir is ", cfg["LOG_DIR"])

        if args.epochs:
            cfg["TRAIN"]["MAX_EPOCH"] = args.epochs
            print("Set max_epochs to ", args.epochs)
    else:
        ###### Create new log
        resume_from_checkpoint = None
        config_filename = "config/parameters.yml"
        cfg = yaml.safe_load(open(config_filename))
        cfg["GIT_COMMIT_VERSION"] = str(
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        ).split("'")[1]
        if args.comment:
            cfg["EXPERIMENT"]["ID"] = args.comment
        cfg["LOG_NAME"] = cfg["EXPERIMENT"]["ID"] + "_" + time.strftime("%Y%m%d_%H%M%S")
        cfg["LOG_DIR"] = os.path.join(
            "./mm/runs", cfg["GIT_COMMIT_VERSION"], cfg["LOG_NAME"]
        )
        if not os.path.exists(cfg["LOG_DIR"]):
            os.makedirs(cfg["LOG_DIR"])
        print("Starting experiment in log dir:", cfg["LOG_DIR"])
    ###### Logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg["LOG_DIR"], default_hp_metric=False, name="Train", version=""
    )

    ###### Dataset
    data = MovingMnistModule(cfg)
    data.setup()
    #print("Data setup done")

    ###### Model
    seed_everything(cfg["EXPERIMENT"]["SEED"], workers=True)
    #model = Many2One(cfg, num_channels=1, num_kernels=64, 
    #                kernel_size=(3, 3), padding=(1, 1), activation="tanh",
    #                frame_size=(64, 64), num_layers=3, peep=cfg["MODEL"]["PEEP"])

    model = One2Many(cfg, num_channels=1, num_kernels=64, 
                    kernel_size=(3, 3), padding=(1, 1), activation="tanh",
                    frame_size=(64, 64), num_layers=3, peep=cfg["MODEL"]["PEEP"])

    ###### Load checkpoint
    if args.resume:
        resume_from_checkpoint = checkpoint_path
        print("Resuming from checkpoint ", checkpoint_path)
    elif args.weights:
        model = model.load_from_checkpoint(checkpoint_path, cfg=cfg)
        resume_from_checkpoint = None
        print("Loading weigths from ", checkpoint_path)

    ###### Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(cfg["LOG_DIR"], "checkpoints"),
        filename="min_val_loss",
        mode="min",
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val/loss", mode="min",
            verbose=True, patience=30)

    ###### Trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        logger=tb_logger,
        accumulate_grad_batches=cfg["TRAIN"]["BATCH_ACC"], # times accumulate_grad_batches
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        log_every_n_steps=cfg["TRAIN"]["LOG_EVERY_N_STEPS"],
        callbacks=[lr_monitor, checkpoint, early_stop],
        strategy = DDPStrategy(find_unused_parameters=True),
        precision='16-mixed',
        check_val_every_n_epoch=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=10
    )

    ###### Training
    trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
    print("Finished Training> Starting Test.")
    #trainer.test(ckpt_path="best")
    checkpoint_path = cfg["LOG_DIR"] + "/checkpoints/min_val_loss.ckpt"
    trainer.test(model, data.test_dataloader(), ckpt_path=checkpoint_path)

    ###### Testing
   # logger = TensorBoardLogger(
   #     save_dir=cfg["LOG_DIR"], default_hp_metric=False, name="test", version=""
   # )
   # checkpoint_path = cfg["LOG_DIR"] + "/checkpoints/min_val_loss.ckpt"
   # model = TCNet.load_from_checkpoint(checkpoint_path, cfg=cfg)
    #trainer.test(model, data.test_dataloader(), ckpt_path='best')

   # if logger:
   #     filename = os.path.join(
   #         cfg["LOG_DIR"], "test", "results_" + time.strftime("%Y%m%d_%H%M%S") + ".yml"
   #     )
   #     log_to_save = {**{"results": results}, **vars(args), **cfg}
   #     with open(filename, "w") as yaml_file:
   #         yaml.dump(log_to_save, yaml_file, default_flow_style=False)



