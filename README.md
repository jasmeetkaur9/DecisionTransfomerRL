Implementation for paper: Decision Transformer: Reinforcement Learning via Sequence Modeling
[Paper Link](https://arxiv.org/pdf/2106.01345)

### Tranformer Arch
1. Single Head with Embedding Size 128
2. FeedForward Head with 3 Layers.

### Experiments

1. Hopper Medium from D4RL Averaged over three random seeds.
<img src="(imgs/hopper.png)" width="45%">
<img src="(imgs/plot_hp_dt.png)" width="45%">
 ![Hopper Task](imgs/hopper.png) ![Hopper Medium](imgs/plot_hp_dt.png) 

2. Ant Medium from D4RL Averaged over three random seeds.
 ![Ant Task](imgs/ant.png) ![Ant Medium](imgs/plot_ant_dt.png) 

3. Half-Cheetah Medium from D4RL Averaged over three random seeds.
 ![HalfCheetah Task](imgs/hc.png) ![HalfCheetah Medium](imgs/plot_hc_dt.png) 
