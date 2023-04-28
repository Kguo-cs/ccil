import pytorch_lightning as pl
import os
from l5kit.configs import load_config_data
from evaluate.CloseLoop_callback import ClosedLoopEvaluate
from pl_Module import module
import argparse

parser = argparse.ArgumentParser(description='Evaling')

parser.add_argument('--model_name', type=str,  choices=["lyft","rasterized","openloop","urbandriver","nuplan","nuplan_vector","nuplan_lanegcn","nuplan_raster"], default="nuplan")
parser.add_argument('--data_root', type=str,help='path to datasets and maps',  default='/mnt/workspace/nuplan/dataset/')#'/mnt/workspace/l5kit_data'
parser.add_argument('--vis_nuplan', action="store_true",help='if run nuplan visualizer',default=False)

args = parser.parse_args()


if "nuplan" in args.model_name:

    os.environ["NUPLAN_DATA_FOLDER"] =args.data_root

    cfg = load_config_data("./config/nuplan/" + args.model_name + ".yaml")
else:
    cfg = load_config_data("./config/lyft/" + args.model_name + ".yaml")

    os.environ["L5KIT_DATA_FOLDER"] =args.data_root

module = module(cfg,args)


trainer = pl.Trainer(fast_dev_run=True,
                     accelerator="gpu",
                     devices=-1,
                     limit_val_batches=1,
                     strategy="ddp",
                     log_every_n_steps=1000,
                     max_epochs=1000,
                     callbacks=[ClosedLoopEvaluate(cfg)])

if args.model_name=="nuplan":
    trainer.test(module,ckpt_path="./pretrained_model/epoch=nuplan.ckpt")
elif args.model_name=="lyft":
    trainer.test(module,ckpt_path="./pretrained_model/epoch=lyft.ckpt")
else:
    trainer.test(module)


