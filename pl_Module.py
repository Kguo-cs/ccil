import pytorch_lightning as pl
from torch import nn
from evaluate.metric_utils import DictMetric,val_metrics
from torch.utils.data import DataLoader
import torch
import pathlib
import os


class module(pl.LightningModule):
    def __init__(self, cfg,args):
        super().__init__()

        self.cfg=cfg
        self.l1loss = nn.L1Loss(reduction="mean")
        self.data_type=self.cfg["data_type"]
        self.model_name=args.model_name
        self.mseloss= nn.MSELoss(reduction="mean")

        if self.model_name=="nuplan_lanegcn":
            from nuplan.planning.training.modeling.models.lanegcn_model import LaneGCN
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            self.model = LaneGCN(map_net_scales=4,
                            num_res_blocks=3,
                            num_attention_layers=4,
                            a2a_dist_threshold=20.0,
                            l2a_dist_threshold=30.0,
                            num_output_features=36,
                            feature_dim=128,
                            vector_map_feature_radius=50,
                            vector_map_connection_scales=[1, 2, 3, 4],
                            past_trajectory_sampling=TrajectorySampling(num_poses=4, time_horizon=1.5),
                            future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6.0),
                            )

        elif self.model_name=="nuplan_vector":
            from nuplan.planning.training.modeling.models.simple_vector_map_model import VectorMapSimpleMLP
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            self.model = VectorMapSimpleMLP(
                            hidden_size=128,
                            num_output_features=36,
                            vector_map_feature_radius=20,
                            past_trajectory_sampling=TrajectorySampling(num_poses=4, time_horizon=1.5),
                            future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6.0),
                            )


        elif self.model_name=="nuplan_raster":
            from nuplan.planning.training.modeling.models.raster_model import RasterModel
            from nuplan.planning.training.modeling.models.lanegcn_model import LaneGCN
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
            from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

            feature_builders=[RasterFeatureBuilder(map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5},
                                                   num_input_channels=4,
                                                   target_width=224,
                                                   target_height=224,
                                                   target_pixel_size=0.5,
                                                   ego_width=2.297,
                                                   ego_front_length=4.049,
                                                   ego_rear_length=1.127,
                                                   ego_longitudinal_offset=0.0,
                                                   baseline_path_thickness=1
                                                   )]

            future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6.0)

            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)]

            self.model=RasterModel(
                    feature_builders=feature_builders,
                    target_builders=target_builders,
                    model_name="resnet50",
                    pretrained=True,
                    num_input_channels=4,
                    num_features_per_pose=3,
                    future_trajectory_sampling=future_trajectory_sampling
            )

        elif self.model_name == "urbandriver":
            self.model = torch.load('./pretrained_model/BPTT.pt')
        elif self.model_name == "openloop":
            self.model = torch.load('./pretrained_model/OL_HS.pt')
        elif self.model_name == "openloop_nohist":
            self.model = torch.load('./pretrained_model/OL.pt')
        elif self.model_name == "multistep":
            self.model = torch.load('./pretrained_model/MS.pt')
        elif self.model_name == 'rasterized':
            use_preturb=True
            early_stop=True
            if use_preturb:
                if early_stop:
                    self.model = torch.load("./pretrained_model/planning_model_20201208_early.pt")
                else:
                    self.model = torch.load("./pretrained_model/planning_model_20201208.pt")
            else:
                if early_stop:
                    self.model = torch.load("./pretrained_model/planning_model_20201208_nopt_early.pt")
                else:
                    self.model = torch.load("./pretrained_model/planning_model_20201208_nopt.pt")
        else:
            from model.model import Model

            model_cfg=cfg['model_params']

            self.read_into_memory= model_cfg["read_into_memory"]

            self.model = Model(model_cfg)

            self.prev_weight=model_cfg["prev_weight"]

        if args.vis_nuplan:
            self.nuplan_vis()
        else:
            self.dataset_setup()

            self.save_hyperparameters(ignore=['model'])

            self.val_metrics=DictMetric(val_metrics,prefix="val/")


    def training_step(self, batch, batch_idx):

        if self.data_type=="nuplan_feature":

            features, targets = batch

            preds = self.model(features)

            predicted_trajectory = preds["trajectory"]

            targets_trajectory = targets["trajectory"]

            loss=  self.mseloss(predicted_trajectory.xy, targets_trajectory.xy)+ self.l1loss(predicted_trajectory.heading, targets_trajectory.heading)

        else:
            preds = self.model(batch)

            pred_positions = preds["positions"]

            pred_yaws = preds["yaws"]

            target_positions = batch["target_positions"]

            target_yaws = batch["target_yaws"]

            target_availabilities = batch["target_availabilities"]

            target_action=torch.cat([target_positions,target_yaws],dim=-1)

            pred_actions=torch.cat([pred_positions,pred_yaws],dim=-1)

            l1_norm=torch.norm(target_action-pred_actions,p=1,dim=-1)

            weighted_l1=torch.cat([l1_norm[:,:-1]*self.prev_weight,l1_norm[:,-1:]],dim=1)

            loss = torch.mean(weighted_l1[target_availabilities])
        return loss

    def validation_step(self, batch, batch_idx):

        if self.data_type=="nuplan_feature":
            features, targets = batch

            predictions = self.model(features)

            predicted_trajectory = predictions["trajectory"]
            targets_trajectory = targets["trajectory"]

            x_diff_mean=self.l1loss(predicted_trajectory.position_x, targets_trajectory.position_x)
            y_diff_mean=self.l1loss(predicted_trajectory.position_y, targets_trajectory.position_y)
            yaw_diff_mean=self.l1loss(predicted_trajectory.heading, targets_trajectory.heading)

        else:
            preds = self.model(batch)

            pred_positions = preds["positions"]
            pred_yaws = preds["yaws"]

            pred_length=pred_positions.shape[1]

            target_mask = batch["target_availabilities"].to(bool)[:,-pred_length:]
            xy = batch["target_positions"][:,-pred_length:]
            yaw = batch["target_yaws"][:,-pred_length:]

            x_diff_mean = self.l1loss(pred_positions[...,0][target_mask], xy[...,0][target_mask])
            y_diff_mean= self.l1loss(pred_positions[...,1][target_mask], xy[...,1][target_mask])
            yaw_diff_mean = self.l1loss(pred_yaws[target_mask], yaw[target_mask])

        output={"x_diff_mean":x_diff_mean,
                "y_diff_mean":y_diff_mean,
                "yaw_diff_mean":yaw_diff_mean,
                }

        return output

    def validation_step_end(self,output):

       self.log_dict(self.val_metrics(output), batch_size=1)

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch,batch_idx)

    def test_step_end(self, output):

        self.log_dict(self.val_metrics(output),batch_size=1)

    def dataset_setup(self):
        self.num_workers = len(os.sched_getaffinity(0)) // torch.cuda.device_count()

        if self.data_type=="lyft":

            from l5kit.data import LocalDataManager
            from data.Lyft_load import ChunkedDataset

            dm = LocalDataManager(None)
            train_zarr = ChunkedDataset(dm.require(self.cfg["train_data_loader"]["key"])).open(read_to_memory=self.read_into_memory)

            val_zarr = ChunkedDataset(dm.require(self.cfg["val_data_loader"]["key"])).open()


            if self.model_name=='lyft':
                from data.Lyft_dataset import LyftDataset
                from data.Lyft_manager import LyftManager

                meta_manager = LyftManager(self.cfg, dm)

                self.train_dataset = LyftDataset(self.cfg, 'train', meta_manager, train_zarr)
                self.val_dataset = LyftDataset(self.cfg, 'val', meta_manager, val_zarr)
            elif self.model_name=="rasterized":
                from l5kit.dataset import EgoDataset
                from l5kit.rasterization import build_rasterizer

                rasterizer = build_rasterizer(self.cfg, dm)

                self.train_dataset = EgoDataset(self.cfg, train_zarr, rasterizer)
                self.val_dataset = EgoDataset(self.cfg, val_zarr, rasterizer)

            else:
                from l5kit.dataset import EgoDatasetVectorized
                from l5kit.vectorization.vectorizer_builder import build_vectorizer

                vectorizer = build_vectorizer(self.cfg, dm)

                self.train_dataset = EgoDatasetVectorized(self.cfg, train_zarr, vectorizer)
                self.val_dataset = EgoDatasetVectorized(self.cfg, val_zarr, vectorizer)

        elif self.data_type == "nuplan":

            from data.nuPlan_dataset import nuPlanDataset
            from data.nuPlan_manager import nuPlanMapManager

            meta_manager =nuPlanMapManager(self.cfg)


            self.train_dataset = nuPlanDataset(self.cfg, "train",meta_manager)
            self.val_dataset = nuPlanDataset(self.cfg, 'val',meta_manager)
            self.test_dataset = nuPlanDataset(self.cfg, 'test',meta_manager)

        else:

            from data.nuPlan_feature_dataset import nuPlanFeatureDataset
            from nuplan.planning.training.data_augmentation.kinematic_agent_augmentation import \
                KinematicAgentAugmentor

            feature_builders = self.model.get_list_of_required_feature()

            target_builders = self.model.get_list_of_computed_target()

            kine_aug=KinematicAgentAugmentor(augment_prob=0.5,
                                                        mean=[1.0, 0.0, 0.0],
                                                        std=[1.0, 1.0, 0.5],
                                                        low=[0.0, -1.0, -0.5],
                                                        high=[1.0, 1.0, 0.5],
                                                        use_uniform_noise=False,
                                                        trajectory_length=12,
                                                        dt=0.5
                                                        )

            if self.model_name=="nuplan_lanegcn":
                from nuplan.planning.training.data_augmentation.agent_dropout_augmentation import AgentDropoutAugmentor

                augmentors = [kine_aug,  AgentDropoutAugmentor(augment_prob=0.5, dropout_rate=0.5)]
            elif self.model_name=="nuplan_vector":
                augmentors = [kine_aug]
            else:
                augmentors=None

            #cache_dir="./data/nuplan_meta/"+self.data_type

            self.train_dataset = nuPlanFeatureDataset('train',feature_builders,target_builders,augmentors)
            self.val_dataset = nuPlanFeatureDataset('val',feature_builders,target_builders)
            self.test_dataset = nuPlanFeatureDataset('test',feature_builders,target_builders)

    def build_dataloader(self,dataset,data_loader_cfg):

        if self.data_type=="nuplan_feature":
            from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate

            train_loader = DataLoader(dataset,
                                      shuffle=data_loader_cfg["shuffle"],
                                      batch_size=data_loader_cfg["batch_size"] // torch.cuda.device_count(),
                                      num_workers=self.num_workers,
                                      prefetch_factor=2,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=FeatureCollate()
                                      )

        else:
            train_loader = DataLoader(dataset,
                                      shuffle=data_loader_cfg["shuffle"],
                                      batch_size=data_loader_cfg["batch_size"] // torch.cuda.device_count(),
                                      num_workers=self.num_workers,
                                      prefetch_factor=2,
                                      pin_memory=True,
                                      drop_last=True
                                      )
        return train_loader

    def train_dataloader(self):
        train_cfg = self.cfg['train_data_loader']

        return self.build_dataloader(self.train_dataset,train_cfg)


    def val_dataloader(self):
        val_cfg = self.cfg['val_data_loader']

        return self.build_dataloader(self.val_dataset,val_cfg)

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):

        policy_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg["optimizer"]['learning_rate']),
            weight_decay=float(self.cfg["optimizer"]['weight_decay'])
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            policy_optimizer,
            lambda steps: min((steps + 1) / int(self.cfg["optimizer"]['warmup_steps']), 1)
        )


        return {'optimizer': policy_optimizer,  'lr_scheduler': { 'scheduler': warmup_scheduler,  'interval': 'step' } }

    def nuplan_vis(self):

        from nuplan.planning.nuboard.base.data_class import NuBoardFile

        from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
        from nuplan.planning.nuboard.nuboard import NuBoard

        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

        test_data_root = os.environ["NUPLAN_DATA_FOLDER"]+"/nuplan-v1.0/test"
        map_root = os.environ["NUPLAN_DATA_FOLDER"]+"/maps"
        map_version = 'nuplan-maps-v1.0'

        scenario_builder = NuPlanScenarioBuilder(
            data_root=test_data_root,
            map_root=map_root,
            db_files=None,
            map_version=map_version,
        )

        vehicle_parameters = VehicleParameters(width=2.297, front_length=4.049, rear_length=1.127,
                                               cog_position_from_rear_axle=1.67, height=1.777, wheel_base=3.089,
                                               vehicle_name='pacifica', vehicle_type='gen1')

        main_exp_folder = './vis/nuplan'

        nuboard_filename = pathlib.Path(main_exp_folder) / (f'nuboard' + NuBoardFile.extension())

        if not os.path.exists(nuboard_filename):

            nuboard_file = NuBoardFile(
                simulation_main_path=main_exp_folder,
                simulation_folder='simulation_log',
                metric_main_path=main_exp_folder,
                metric_folder='metric',
                aggregator_metric_folder='aggregator_metric',
            )

            nuboard_file.save_nuboard_file(nuboard_filename)

        metric_path=main_exp_folder+'/metric'

        if not os.path.exists(metric_path):
            os.mkdir(metric_path)


        nuboard = NuBoard(
            nuboard_paths=[str(nuboard_filename)],
            scenario_builder=scenario_builder,
            vehicle_parameters=vehicle_parameters,
        )

        nuboard.run()
