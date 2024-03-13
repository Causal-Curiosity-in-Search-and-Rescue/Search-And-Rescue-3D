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

lass CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        features_dim = 256
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
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

    def forward(self, observations):
        # Convert each part of the observation from NumPy to PyTorch tensor and ensure it's flattened correctly
        goal_pos = self.flatten(torch.tensor(observations['goal_position'], dtype=torch.float32))
        obj_pos = self.flatten(torch.tensor(observations['object_positions'], dtype=torch.float32))
        obj_tex = self.flatten(torch.tensor(observations['object_textures'], dtype=torch.float32))
        obj_mov = self.flatten(torch.tensor(observations['object_movables'], dtype=torch.float32))
        walls_info = self.flatten(torch.tensor(observations['walls_info'], dtype=torch.float32))
        collision_info = torch.tensor(observations['collision_info'], dtype=torch.float32).unsqueeze(-1)  # This is already a single value, so we make it a batched single value
        prev_actions = self.flatten(torch.tensor(observations['previous_actions'], dtype=torch.float32))

        # Combine all the tensors into one large tensor
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
        self.net = CustomPolicy(total_features_dim, action_space)

    def _build(self, lr_schedule):
        # This is where we build the model
        # Note: No need to call super()._build() as we redefine everything here
        self.mlp_extractor = self.features_extractor_class(self.observation_space)

    def forward(self, obs, deterministic=False):
        # Here, obs is a dict from your environment's observation space
        features = self.mlp_extractor(obs)
        return self.net(features)