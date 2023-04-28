from torch import nn
import torch
from torch.nn import functional as F

class MLP(nn.Module):
    r"""
    Construct a MLP, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, hidden_size):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, ..., input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class LinearWithNorm(nn.Module):
    r"""
    Construct a MLP, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, output_size):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size: the size of output layer.
        """
        super(LinearWithNorm, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.norm = torch.nn.LayerNorm(output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, ..., input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        return x

class MapTransformer(nn.Module):

    def __init__(self,input_dim,d_model,layer_num,encoder_layer):
        super(MapTransformer, self).__init__()

        self.input_embed=LinearWithNorm(input_dim, d_model)

        self.map_net = nn.TransformerEncoder( encoder_layer, num_layers=layer_num )

    def forward(self,raw_polylines):

        mask= raw_polylines[:,:,0,-1].to(bool)

        polylines=raw_polylines[mask]

        polylines_mask = ~polylines[..., -1].to(bool)

        polylines_features = self.input_embed(polylines[...,:-1])

        polylines_features = self.map_net(polylines_features,src_key_padding_mask=polylines_mask)

        features=torch.zeros([raw_polylines.shape[0],raw_polylines.shape[1],polylines_features.shape[-1]],device=raw_polylines.device)

        features[mask]=torch.amax(polylines_features,dim=1)

        return features,~mask


class ObsEncoder(nn.Module):

    def __init__(self,cfg,encoder_layer):
        super(ObsEncoder, self).__init__()

        d_model = cfg["d_model"]

        local_num_layers = cfg["local_num_layers"]
        global_layers=cfg["global_num_layers"]

        lane_feat_dim = cfg["lane_feat_dim"]
        cross_feat_dim = cfg["cross_feat_dim"]

        self.ignore_ego=cfg["ignore_ego"]

        if self.ignore_ego:
            ego_feat_dim=2
        else:
            ego_feat_dim=11

        agent_feat_dim =cfg["agent_feat_dim"]*(cfg["history_num_frames_agents"]+1)-1
        # if self.use_pos_enc==True:
        #     self.lane_pose_enc   = nn.Embedding(150, d_model)
        #     self.lane_embed = LinearWithNorm(lane_feat_dim-1, d_model)
        #     self.cross_embed = LinearWithNorm(cross_feat_dim-1, d_model)
        # else:

        self.lane_net=MapTransformer(lane_feat_dim,d_model,local_num_layers,encoder_layer)

        self.cross_net=MapTransformer(cross_feat_dim,d_model,local_num_layers,encoder_layer)

        self.ego_embed=MLP(ego_feat_dim,d_model)

        self.agent_embed=MLP(agent_feat_dim,d_model)

        self.global_net=nn.TransformerEncoder( encoder_layer, num_layers=global_layers )

    def forward(self,batch) :

        lane_polylines=batch["lane_polylines"]
        crosswalk_polylines=batch["crosswalk_polylines"]
        agent_polylines=batch["agent_polylines"]
        ego_polyline_ = batch["ego_polyline"]

        lane_polylines=lane_polylines.view(-1,lane_polylines.shape[-3],lane_polylines.shape[-2],lane_polylines.shape[-1])
        crosswalk_polylines=crosswalk_polylines.view(-1,crosswalk_polylines.shape[-3],crosswalk_polylines.shape[-2],crosswalk_polylines.shape[-1])
        agent_polylines=agent_polylines.view(-1,agent_polylines.shape[-2],agent_polylines.shape[-1])
        ego_polylines=ego_polyline_.view(-1,ego_polyline_.shape[-2],ego_polyline_.shape[-1])

        if self.ignore_ego:
            ego_features=self.ego_embed(ego_polylines[...,-3:-1])
        else:
            ego_features=self.ego_embed(ego_polylines[...,:-1])

        agent_features = self.agent_embed(agent_polylines[...,:-1])

        agent_mask=~agent_polylines[...,-1].to(bool)

        ego_mask=~ego_polylines[...,-1].to(bool)

        lane_features,lane_mask=self.lane_net(lane_polylines)

        cross_features,cross_mask=self.cross_net(crosswalk_polylines)

        all_features=torch.cat([ego_features,agent_features,lane_features,cross_features],dim=1)

        all_mask=torch.cat([ego_mask,agent_mask,lane_mask,cross_mask],dim=1)

        features = self.global_net(all_features,src_key_padding_mask= all_mask)

        #ego_features=torch.amax(features,dim=1).view(ego_polyline_.shape[0],-1,ego_features.shape[-1])

        ego_features=features[:,0].view(ego_polyline_.shape[0],-1,ego_features.shape[-1])

        return ego_features,ego_mask.view(ego_polyline_.shape[0],-1)