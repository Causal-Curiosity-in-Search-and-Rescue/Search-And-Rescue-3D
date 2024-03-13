import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, make_proba_distribution
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import pdb 

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, use_sde=False, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs)

        # Define your custom layers and logic here
        features_dim = 256
        num_m = 9  # Movable
        num_i = 11  # Immovable
        num_s = 1  # Start Positions
        n_texture_classes = 2
        n_objects = num_m + 1 + num_i
        AGENT_ACTION_LEN = 1000  # Your actual length

        # Calculate the total dimension after flattening all parts of the observation space
        total_dim = np.prod([3]) + np.prod([n_objects, 3]) + n_objects + n_objects + np.prod([4, 3]) + 1 + np.prod([1000])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(total_dim, features_dim)
        self.fc1 = nn.Linear(features_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # self.action_out = nn.Linear(64, action_space.n)
        # self.value_out = nn.Linear(64, 1)  # Value output layer
        
        self.action_out = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )
        self.value_out = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, deterministic=False):
        # Custom feature extraction logic
        goal_pos = self.flatten(torch.tensor(obs['goal_position'], dtype=torch.float32))
        obj_pos = self.flatten(torch.tensor(obs['object_positions'], dtype=torch.float32))
        obj_tex = self.flatten(torch.tensor(obs['object_textures'], dtype=torch.float32))
        obj_mov = self.flatten(torch.tensor(obs['object_movables'], dtype=torch.float32))
        walls_info = self.flatten(torch.tensor(obs['walls_info'], dtype=torch.float32))
        collision_info = torch.tensor(obs['collision_info'], dtype=torch.float32).unsqueeze(-1)  # This is already a single value, so we make it a batched single value
        prev_actions = self.flatten(torch.tensor(obs['previous_actions'], dtype=torch.float32))

        # Combine all the tensors into one large tensor
        combined = torch.cat([goal_pos, obj_pos, obj_tex, obj_mov, walls_info, collision_info, prev_actions], dim=1)

        # Pass through MLP layers
        x = F.relu(self.fc(combined))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

         # Generate action logits and value
        action_logits = self.action_net(x)
        value = self.value_net(x)  # Calculate value

        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            # Use softmax for action distribution for exploration
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()

        return actions, action_logits, value  # Return raw logits for training