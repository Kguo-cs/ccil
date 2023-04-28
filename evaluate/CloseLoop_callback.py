import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from .evaluator import MyClosedLoopEvaluator

class ClosedLoopEvaluate(Callback):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def setup(self, trainer: pl.Trainer, pl_module, stage=None):

        self.num_scenes_per_device = self.cfg["eval"]["num_scenes_to_unroll"] // trainer.num_devices

        self.rollout_batchsize = self.cfg['eval']['batch_size']

        self.data_type=self.cfg["data_type"]

        visualizer=None

        if self.data_type=="lyft":
            from l5kit.simulation.dataset import SimulationConfig
            from .Lyft_simulator import ClosedLoopSimulator
            from .Lyft_visualizer import Visualizer

            eval_dataset = pl_module.val_dataset
            eval_size = np.ceil(len(eval_dataset.cumulative_sizes) / trainer.num_devices).astype(int)

            sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                                            distance_th_far=500, distance_th_close=50, num_simulation_steps=None,
                                            start_frame_index=0, show_info=False)

            self.sim_loop = ClosedLoopSimulator(
                sim_cfg,
                eval_dataset,
                pl_module.device,
                model_ego=pl_module.model,
                model_agents=None)

            #visualizer=Visualizer(eval_dataset)
        else:
            from .nuPlan_simulator import Nuplan_simulator

            if trainer.training:
                eval_dataset = pl_module.val_dataset
            else:
                eval_dataset =pl_module.test_dataset

            eval_size=pl_module.val_dataset.scene_num

            self.sim_loop = Nuplan_simulator(pl_module.model,eval_dataset,pl_module.model_name)

        self.scene_ids = list(range(pl_module.global_rank, eval_size, trainer.num_devices))

        self.my_evaluator = MyClosedLoopEvaluator(self.data_type,visualizer,verbose=False)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        with torch.no_grad():
            if trainer.sanity_checking:
                scene_ids = np.random.choice(self.scene_ids, 1, replace=False)
            else:
                scene_ids = np.random.choice(self.scene_ids, min(self.num_scenes_per_device,len(self.scene_ids)), replace=False)

            results = self.roll_sim(pl_module, scene_ids)

            pl_module.log_dict(pl_module.val_metrics(results), batch_size=len(scene_ids))

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            scene_ids=self.scene_ids
            #scene_ids = np.random.choice(self.scene_ids, 2, replace=False)

            results = self.roll_sim(pl_module,  scene_ids)

        pl_module.log_dict(pl_module.val_metrics(results), batch_size=len(scene_ids))

    def roll_sim(self,pl_module, scene_ids):
        scene_num = len(scene_ids)

        batch_num = np.ceil(scene_num / self.rollout_batchsize).astype(int)

        for i in range(batch_num):
            scenes_to_unroll = list(
                scene_ids[i * self.rollout_batchsize:min((i + 1) * self.rollout_batchsize, scene_num)])


            sim_outs = self.sim_loop.unroll(scenes_to_unroll,device=pl_module.device, model_name=pl_module.model_name)

            self.my_evaluator.evaluate(sim_outs,data_type=self.data_type)

        res_agg = self.my_evaluator.validate()

        return res_agg