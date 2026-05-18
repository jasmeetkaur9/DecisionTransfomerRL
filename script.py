import sys
import os 
import gc
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import wandb 
from omegaconf import OmegaConf
import matplotlib.pyplot as plt 
import mujoco
import gymnasium as gym
import numpy as np
from core import DecisionTransformer
from tqdm import tqdm 
import cProfile, pstats
import argparse


def train():

    run = wandb.init()
    wandb_config = wandb.config 

    seed = wandb_config.seed
    batch_size = wandb_config.batch_size
    env_name = str(wandb_config.env_name)
    dataset_id = str(wandb_config.env_id)
    lr = float(wandb_config.lr)
    n_epochs = wandb_config.training_epochs
    eval_epochs = wandb_config.target_return 
    sampled_ep = int(wandb_config.sampled_eps)

    env = gym.make(env_name)

    state_n  = env.observation_space.shape[0]
    action_n = env.action_space.shape[0]