import minari 
import torch 
import random 
from torch import vmap 


def reward_to_return(rewards, length):
    mask = torch.tril(torch.ones(length, length, dtype=torch.double))
    rewards = rewards @ mask
    return rewards 


class DatasetGenerator:

    def __init__(self, batch_size, horizon, dataset_id):
        
        self.batch_size = batch_size
        self.horizon = horizon 
        self.dataset_id = dataset_id 
        self.dataset = minari.load_dataset(self.dataset_id, download=True)
        self.tn_ep = self.dataset.total_episodes

    def setup_episodes(self): 

        self.sampled_episodes = self.dataset.sample_episodes(1000)
        self.ep_length = self.sampled_episodes[0].observations.shape[0]
        self.n_ep = self.tn_ep

    def get_dataset(self):

        random_episode = random.randint(0, self.n_ep-1)
        self.ep_length = self.sampled_episodes[random_episode].observations.shape[0]
        idx = torch.randint(low = 0, high = self.ep_length-self.horizon-1, size = (self.batch_size,))

        states = self.sampled_episodes[random_episode].observations 
        actions = self.sampled_episodes[random_episode].actions
        rewards = self.sampled_episodes[random_episode].rewards 

        data_states = torch.stack([torch.tensor(states[x:x+self.horizon], dtype=torch.float64) for x in idx])
        data_actions = torch.stack([torch.tensor(actions[x:x+self.horizon], dtype=torch.float64) for x in idx])
        data_rewards = torch.stack([torch.tensor(rewards[x:x+self.horizon], dtype=torch.float64) for x in idx])

        data_returns = reward_to_return(data_rewards, self.horizon).unsqueeze(-1)

        data_next_states = torch.stack([torch.tensor(states[x+1:x+1+self.horizon], dtype=torch.float64) for x in idx])
        data_next_actions = torch.stack([torch.tensor(actions[x+1:x+1+self.horizon], dtype=torch.float64) for x in idx])
        data_next_rewards = torch.stack([torch.tensor(rewards[x+1:x+1+self.horizon], dtype=torch.float64) for x in idx])

        data_next_returns = reward_to_return(data_next_rewards, self.horizon).unsqueeze(-1)

        return data_states, data_actions, data_returns, data_next_states, data_next_actions, data_next_returns




