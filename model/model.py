import torch
import torch.nn as nn
from .obsEncoder import ObsEncoder
from  .lqr_smoother import LQR

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        dropout = cfg["dropout"]
        d_model = cfg['d_model']
        causal_num_layers=cfg['causal_num_layers']

        self.nhead = cfg['head_num']

        self.future_num_frames = cfg['future_num_frames']
        self.act_dim = cfg['act_dim']
        self.causal_len = cfg['causal_len']
        self.causal_interval = cfg['causal_interval']

        attention_mask = torch.ones((self.causal_len, self.causal_len)).bool()
        causal_mask = ~torch.tril(attention_mask, diagonal=0)  # diagonal=0, keep the diagonal

        self.register_buffer("causal_mask", causal_mask)

        self.embed_timestep = nn.Embedding(self.causal_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=d_model,
            nhead=self.nhead,
            dropout=dropout,
            batch_first=True)

        self.embed_state = ObsEncoder(cfg,encoder_layer)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=causal_num_layers)

        self.predict_action=nn.Linear(d_model,   self.act_dim *  self.future_num_frames)

        self.lqr=LQR(cfg)

    def causal_transformer(self, state_embeddings_,key_mask=None):

        batch_size,seq_length =state_embeddings_.shape[0], state_embeddings_.shape[1]

        timestep=torch.arange(self.causal_len-seq_length,self.causal_len,device=state_embeddings_.device)

        time_embeddings_= self.embed_timestep( timestep)

        stacked_inputs=state_embeddings_+time_embeddings_[None]

        mask=self.causal_mask[:seq_length,:seq_length]

        if key_mask is not None:

            diag_mask=~torch.eye(seq_length,device=stacked_inputs.device).to(bool)

            key_src_mask=key_mask[:,None]  |  mask[None]

            mask= (key_src_mask & diag_mask[None]).repeat(self.nhead,1,1)#(bsz * num_heads, tgt_len, src_len)

        transformer_outputs = self.transformer(stacked_inputs,   mask=mask)

        if not self.training:
            transformer_outputs=transformer_outputs[:,-1]

        action_preds= self.predict_action(transformer_outputs).view( batch_size, -1, self.future_num_frames, self.act_dim)

        return action_preds

    def forward(self,batch):

        state_embeddings,ego_mask=self.embed_state(batch)

        action_preds=self.causal_transformer(state_embeddings,ego_mask)

        return {"positions":action_preds[...,:2],
                "yaws":action_preds[...,2:3] }

    def get_action(self, batch, state_embeddings):

        state_embedding,_ = self.embed_state(batch)

        state_embeddings = torch.cat([state_embeddings, state_embedding], dim=1)

        start_index = self.causal_interval - 1 - min(self.causal_len, (
                state_embeddings.shape[1] + self.causal_interval - 1) // self.causal_interval) * self.causal_interval

        state_embeddings_ = state_embeddings[:, start_index::self.causal_interval]

        action_preds = self.causal_transformer(state_embeddings_)

        ego_polyline=batch["ego_polyline"][:, 0]

        last_action_preds=self.lqr(ego_polyline,action_preds)

        return {"positions": last_action_preds[..., :2],
            "yaws": last_action_preds[..., 2:3]}, state_embeddings[:,-(self.causal_len-1)*self.causal_interval:]


