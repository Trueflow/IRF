import numpy as np
import torch as th
import torch.nn as nn


class SI_Weight(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_agents = args.num_agents
        self.n_actions = args.action_size
        self.state_dim = int(np.prod(args.state_size))
        self.action_dim = self.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        for _ in range(self.num_kernel):
            self.key_extractors.append(nn.Linear(self.state_dim, 1))
            self.agents_extractors.append(
                nn.Linear(self.state_dim, self.n_agents)
            )
            self.action_extractors.append(
                nn.Linear(self.state_action_dim, self.n_agents)
            )

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_action = th.cat([states, actions], dim=1)

        weights = []
        for key_net, agent_net, action_net in zip(
            self.key_extractors,
            self.agents_extractors,
            self.action_extractors,
        ):
            key = th.sigmoid(key_net(states)).repeat(1, self.n_agents)
            agent = th.sigmoid(agent_net(states))
            action = th.sigmoid(action_net(state_action))
            weights.append((key + 1e-10) * agent * action)

        return th.stack(weights, dim=1).sum(dim=1)
