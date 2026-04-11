import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Qatten_Weight(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.num_agents
        self.state_dim = int(np.prod(args.state_size))
        self.unit_dim = args.unit_dim
        # EnvInfo.yaml uses the plural spelling.
        self.unit_state_offset = getattr(
            args,
            "unit_states_offset",
            getattr(args, "unit_state_offset", 0),
        )
        self.n_actions = args.action_size
        self.n_head = args.num_heads
        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        required_state_dim = (
            self.unit_state_offset + self.unit_dim * self.n_agents
        )
        if required_state_dim > self.state_dim:
            raise ValueError(
                "Unit-state slice exceeds state_size: "
                f"offset={self.unit_state_offset}, "
                f"unit_dim={self.unit_dim}, n_agents={self.n_agents}, "
                f"state_dim={self.state_dim}."
            )

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        for _ in range(self.n_head):
            self.selector_extractors.append(
                nn.Sequential(
                    nn.Linear(self.state_dim, args.hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(
                        args.hypernet_embed,
                        self.embed_dim,
                        bias=False,
                    ),
                )
            )
            self.key_extractors.append(
                nn.Linear(self.unit_dim, self.embed_dim, bias=False)
            )

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states, actions=None):
        states = states.reshape(-1, self.n_agents, self.state_dim)
        states = states[:, 0, :]

        start = self.unit_state_offset
        end = start + self.unit_dim * self.n_agents
        unit_states = states[:, start:end].reshape(
            -1, self.n_agents, self.unit_dim
        )
        unit_states = unit_states.permute(1, 0, 2)

        selectors = [
            extractor(states) for extractor in self.selector_extractors
        ]
        keys = [
            [extractor(unit) for unit in unit_states]
            for extractor in self.key_extractors
        ]

        head_logits = []
        head_weights = []
        for current_keys, selector in zip(keys, selectors):
            logits = th.matmul(
                selector.view(-1, 1, self.embed_dim),
                th.stack(current_keys).permute(1, 2, 0),
            )
            weights = F.softmax(
                logits / np.sqrt(self.embed_dim), dim=2
            )
            head_logits.append(logits)
            head_weights.append(weights)

        attention = th.stack(head_weights, dim=1).view(
            -1, self.n_head, self.n_agents
        )
        attention = attention.sum(dim=1)
        value = self.V(states).view(-1, 1)

        magnitude_regularizer = self.attend_reg_coef * sum(
            (logit ** 2).mean() for logit in head_logits
        )
        entropies = [
            -(
                (prob.squeeze(1) + 1e-8).log()
                * prob.squeeze(1)
            )
            .sum(dim=-1)
            .mean()
            for prob in head_weights
        ]
        return attention, value, magnitude_regularizer, entropies
