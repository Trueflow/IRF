import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_agents = args.num_agents
        self.state_dim = int(np.prod(args.state_size))
        self.embed_dim = args.mixing_embed_dim
        self.normalize_state = getattr(
            args, "normalize_mixer_state", True
        )
        self.state_norm = nn.LayerNorm(
            self.state_dim, elementwise_affine=False
        )

        hypernet_layers = getattr(args, "hypernet_layers", 1)
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(
                self.state_dim, self.embed_dim * self.n_agents
            )
            self.hyper_w_final = nn.Linear(
                self.state_dim, self.embed_dim
            )
        elif hypernet_layers == 2:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(
                    hypernet_embed,
                    self.embed_dim * self.n_agents,
                ),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise ValueError(
                "QMIX supports one or two hypernetwork layers."
            )

        self.hyper_b_1 = nn.Linear(
            self.state_dim, self.embed_dim
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        if not th.isfinite(states).all():
            raise FloatingPointError(
                "Non-finite value found in QMIX global state."
            )
        if self.normalize_state:
            states = self.state_norm(states)

        agent_qs = agent_qs.reshape(
            -1, 1, self.n_agents
        )
        w1 = th.abs(self.hyper_w_1(states)).view(
            -1, self.n_agents, self.embed_dim
        )
        b1 = self.hyper_b_1(states).view(
            -1, 1, self.embed_dim
        )
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = th.abs(
            self.hyper_w_final(states)
        ).view(-1, self.embed_dim, 1)
        value = self.V(states).view(-1, 1, 1)
        q_total = th.bmm(hidden, w_final) + value
        return q_total.view(batch_size, -1, 1)
