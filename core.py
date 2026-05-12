import torch 
from transformer_heads import FeedForward, SingleHead
import gymnasium as gym
from tqdm import tqdm 
import numpy as np 
torch.set_default_dtype(torch.float64)
import os 
import gc 
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


class DecisionTransformer(torch.nn.Module):

    def __init__(self, params, batch_size, sampled_ep, lr, dataset_id):

        super().__init__()

        self.horizon_length = params['block_size']
        self.n_embed = int(params['n_embed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_n = params['state_n']
        action_n = params['action_n']

        self.state_embedding = torch.nn.Linear(state_n, self.n_embed).to(self.device)
        self.action_embedding = torch.nn.Linear(action_n, self.n_embed).to(self.device)
        self.reward_embedding = torch.nn.Linear(1, self.n_embed).to(self.device)

        self.pos_embedding = torch.nn.Embedding(self.horizon_length, self.n_embed).to(self.device)
        self.embed_ln = torch.nn.LayerNorm(self.n_embed).to(self.device)

        self.predict_state = torch.nn.Linear(self.n_embed, state_n).to(self.device)
        self.predict_action = torch.nn.Sequential(
            torch.nn.Linear(self.n_embed, action_n),
            torch.nn.Tanh()
        ).to(self.device)
        self.predict_return = torch.nn.Linear(self.n_embed, 1).to(self.device)

        self.attn_head = SingleHead(self.n_embed, self.n_embed, 3*self.horizon_length).to(self.device)
        self.ffn       = FeedForward(self.n_embed).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

