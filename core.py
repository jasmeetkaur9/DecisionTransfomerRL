import torch 
from transformer_heads import FeedForward, SingleHead
import gymnasium as gym
from tqdm import tqdm 
import numpy as np 
from dataset import DatasetGenerator
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

        self.dg = DatasetGenerator(batch_size, self.horizon_lenght, dataset_id)
        self.dg.setup_episodes(sampled_ep)

    
    def forward(self, states, actions, returns_to_go, horizon_length):

        pos_em = self.pos_embedding(torch.arange(0, horizon_length, device=self.device))
        state_em = self.state_embedding(states) + pos_em
        action_em = self.action_embedding(actions) + pos_em 
        reward_em = self.reward_embedding(returns_to_go) + pos_em 

        batch_size, horizon_length, n_embed = state_em.shape
        stack_input = torch.stack([state_em, action_em, reward_em], dim=2)
        stack_input = stack_input.reshape(batch_size,3*horizon_length,n_embed)

        output = self.attn_head(stack_input)
        output = self.ffn(output)

        out = output.reshape(batch_size, horizon_length, 3, self.n_embed).permute(0, 2, 1, 3)

        state_preds = self.predict_state(out[:, 0])
        action_preds = self.predict_action(out[:, 1])
        return_preds = self.predict_return(out[:, 2])

        return output, state_preds, action_preds, return_preds

