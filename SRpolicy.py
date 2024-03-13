import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, make_proba_distribution


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Assuming num_m, num_i, and other dimensions based on your setup
        num_m = 9  # Movable
        num_i = 11  # Immovable
        num_s = 1  # Start Positions
        self.n_objects = num_m + 1 + num_i
        self.agent_action_len = 1000  # Placeholder, adjust according to your needs
        self.action_dist = make_proba_distribution(action_space)

        # Define the separate branches based on your provided structure
        self.goal_position_branch = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.object_positions_branch = nn.Sequential(nn.Linear(self.n_objects * 3, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.object_attributes_branch = nn.Sequential(nn.Linear(self.n_objects * 2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.walls_info_branch = nn.Sequential(nn.Linear(4 * 3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.previous_actions_branch = nn.Sequential(nn.Linear(self.agent_action_len, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.collision_info_branch = nn.Sequential(nn.Embedding(5, 10), nn.Flatten(), nn.Linear(10, 32), nn.ReLU())

        # Combine the branches
        total_features = 128 + 128 + 64 + 64 + 64 + 32  # Sum of all branch outputs
        self.combined_network = nn.Sequential(nn.Linear(total_features, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())

        # Additional actor and critic specific layers if necessary
        # Adjust these dimensions according to your environment's action space
        self.actor = nn.Linear(128, action_space.n)  # For discrete action spaces
        self.critic = nn.Linear(128, 1)  # Typically outputs a single value for value estimation

    def forward(self, obs):
        goal_features = self.goal_position_branch(obs['goal_position'])
        object_pos_features = self.object_positions_branch(obs['object_positions'].reshape(-1, self.n_objects * 3))
        object_attr_features = self.object_attributes_branch(th.cat([obs['object_textures'].float(), obs['object_movables'].float()], dim=1))
        wall_features = self.walls_info_branch(obs['walls_info'].reshape(-1, 4 * 3))
        prev_action_features = self.previous_actions_branch(obs['previous_actions'].float())
        collision_features = self.collision_info_branch(obs['collision_info'])

        combined_features = th.cat([goal_features, object_pos_features, object_attr_features, wall_features, prev_action_features, collision_features], dim=1)
        combined_output = self.combined_network(combined_features)

        # Actor and critic networks
        action_logits = self.actor(combined_output)
        value_estimate = self.critic(combined_output)
        return action_logits, value_estimate

    def _predict(self, observation, deterministic=False):
        # This method is used for predictions
        action_logits, _ = self.forward(observation)
        # Create the distribution and sample actions
        action_distribution = self.action_dist.proba_distribution(action_logits)
        actions = action_distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions):
        # This method is used during learning to evaluate the actions taken
        action_logits, values = self.forward(obs)
        dist = self.action_dist.proba_distribution(action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy
