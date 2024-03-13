import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, make_proba_distribution
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # Here you should add any specific layers you need to process your observations
        # For example, for simplicity, let's concatenate everything into one big vector
        # Note: You should replace 'n_objects' and 'AGENT_ACTION_LEN' with their actual values
        num_m = 9 # Movable
        num_i = 11 # Immovable
        num_s = 1 # Start Positions
        n_texture_classes = 2
        n_objects = num_m + 1 + num_i
        AGENT_ACTION_LEN = 1000# Your actual length

        total_dim = np.prod(observation_space['goal_position'].shape) + \
                    np.prod(observation_space['object_positions'].shape) + \
                    np.prod(observation_space['object_textures'].shape) + \
                    np.prod(observation_space['object_movables'].shape) + \
                    np.prod(observation_space['walls_info'].shape) + \
                    np.prod([1]) + \
                    np.prod(observation_space['previous_actions'].shape)
                    
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(total_dim, features_dim)

    def forward(self, observations):
        # Flatten and concatenate all parts of the observations
        goal_pos = self.flatten(observations['goal_position'])
        obj_pos = self.flatten(observations['object_positions'])
        obj_tex = self.flatten(observations['object_textures'])
        obj_mov = self.flatten(observations['object_movables'])
        walls_info = self.flatten(observations['walls_info'])
        collision_info = observations['collision_info'].float().unsqueeze(-1)  # Add an extra dimension to match
        prev_actions = self.flatten(observations['previous_actions'])

        combined = torch.cat([goal_pos, obj_pos, obj_tex, obj_mov, walls_info, collision_info, prev_actions], dim=1)
        return self.fc(combined)


class CustomPolicy(nn.Module):
    def __init__(self, features_dim, action_space):
        super(CustomPolicy, self).__init__()
        # Adjust these layers according to the output of your feature extractor
        self.fc1 = nn.Linear(features_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_out = nn.Linear(64, action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_out(x)
        return action_logits

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_class=CustomFeatureExtractor, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, features_extractor_class, *args, **kwargs)
        
        # Instead of directly using CustomPolicy here, it will now receive features from the feature extractor
        total_features_dim = kwargs.get('features_dim', 256)  # This should match what's defined in your feature extractor
        self.features_dim = total_features_dim
        self.net = CustomPolicy(total_features_dim, action_space.n)

    def _build(self, lr_schedule):
        # This is where we build the model
        # Note: No need to call super()._build() as we redefine everything here
        self.mlp_extractor = self.features_extractor_class(self.observation_space, self.features_dim)

    def forward(self, obs, deterministic=False):
        # Here, obs is a dict from your environment's observation space
        features = self.extract_features(obs)
        return self.net(features)
