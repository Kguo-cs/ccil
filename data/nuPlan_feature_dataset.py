from torch.utils.data import Dataset
import numpy as np
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from .nuPlan_load import db_process
import os

class nuPlanFeatureDataset(Dataset):

    def __init__(self,type,feature_builders,target_builders,augmentors=None,cache_dir=None):

        try:
            data=np.load('./data/nuplan_meta/'+type+".npz",allow_pickle=True)

            scenarios = data["scenarios"]

        except:
            data_root = os.environ["NUPLAN_DATA_FOLDER"]+'/nuplan-v1.0/' +type

            ego_array,agent_array,tl_array,route_array,scenarios,scenarios_info=db_process(type,data_root)

        if type=="test":
            self.scenarios=scenarios[::250]
        else:
            self.scenarios=scenarios

        self.scene_num=len(self.scenarios)

        self.cache_dir=cache_dir

        self._radius=50

        self.connection_scales=[1,2,3,4]

        self._num_future_poses=12

        self._time_horizon=6.0

        self.num_past_poses=4

        self.past_time_horizon=1.5

        self._feature_preprocessor = FeaturePreprocessor(
            cache_path=cache_dir,
            force_feature_computation=False,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        self.augmentors=augmentors

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):

        scenario = self.scenarios[idx]

        features, targets, _ = self._feature_preprocessor.compute_features(scenario)

        if self.augmentors is not None:
            for augmentor in self.augmentors:
                augmentor.validate(features, targets)
                features, targets = augmentor.augment(features, targets, scenario)

        features = {key: value.to_feature_tensor() for key, value in features.items()}
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}

        return features, targets

