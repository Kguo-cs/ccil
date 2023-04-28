import pytorch_lightning as pl
import os
from l5kit.configs import load_config_data
from evaluate.CloseLoop_callback import ClosedLoopEvaluate
from datetime import datetime
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_Module import module
import argparse

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--model_name', type=str,  choices=["lyft","rasterized","openloop","urbandriver","nuplan","nuplan_vector","nuplan_lanegcn","nuplan_raster"], default="lyft")
parser.add_argument('--data_root', type=str,help='path to datasets and maps',  default='/mnt/workspace/l5kit_data')#"/mnt/workspace/nuplan/dataset/")#'/mnt/workspace/l5kit_data')#/mnt/workspace/nuplan/dataset/
parser.add_argument('--vis_nuplan', action="store_true",help='if run nuplan visualizer',default=False)
parser.add_argument('--local_rank', type=int,  default=0)

args = parser.parse_args()

if "nuplan" in args.model_name:

    os.environ["NUPLAN_DATA_FOLDER"] =args.data_root

    cfg = load_config_data("./config/nuplan/" + args.model_name + ".yaml")
else:
    cfg = load_config_data("./config/lyft/" + args.model_name + ".yaml")

    os.environ["L5KIT_DATA_FOLDER"] =args.data_root

module = module(cfg,args)

logger = loggers.TensorBoardLogger(save_dir=cfg['log_dir'],
                                   name=cfg['exp_name'],
                                   version=datetime.now().strftime("%Y_%m_%d_%H_%M")
                                   )

print("log_dir:",logger.log_dir)

checkpoint_cb = ModelCheckpoint(dirpath=logger.log_dir,
                                save_top_k=3,
                                monitor='val/collision_rate',
                                filename='{epoch:02d}-{val_loss:.4f}',
                               )

trainer = pl.Trainer(fast_dev_run=False,
                     logger=logger,
                     accelerator="gpu",
                     devices=-1,
                     limit_val_batches=0.01,
                 #    val_check_interval=2,
                     strategy="ddp",
                     log_every_n_steps=1000,
                     max_epochs=1000,
                     callbacks=[ClosedLoopEvaluate(cfg),checkpoint_cb])

trainer.fit(module)
