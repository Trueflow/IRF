# Adapted from: https://github.com/yalidu/liir
# Original project: LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning
# Upstream license: MIT
import torch
import torch.nn
import torch.nn.functional as F
class IRFcritic(torch.nn.Module):
    def __init__(self, args):
        super(IRFcritic, self).__init__()
        self.args = args
        self.state_size = args.state_size
        self.obs_size = args.obs_size
        self.n_actions = args.action_size
        self.n_agents = args.num_agents
        self.device = args.device
        self.embadding_dim = args.embadding_dim

        # self.max_seq_length = n_step
        self.output_type = "q"
        #input_shape : state + obs + actions_onehot * n_agents + n_agents
        self.input_shape = self.create_input_size(onehot=True)
        # 655 + 4*3 + 3
        self.fc1 = torch.nn.Linear(self.input_shape, self.embadding_dim)
        self.fc2 = torch.nn.Linear(self.embadding_dim, self.embadding_dim)
        self.r_in = torch.nn.Linear(self.embadding_dim, self.n_actions) # r_in layer     
        self.v_mix = torch.nn.Linear(self.embadding_dim, 1) # v_mix layer (v?)  
        self.v_ex = torch.nn.Linear(self.embadding_dim*self.n_agents,1) # v_ex layer
    
    def forward(self, states, obs, actions, t=None):
        max_t = actions.shape[0] if t is None else 1
        inputs = self.build_input_Critic(states, obs, actions, t=t, onehot=True) # (bs, n_agents, input_shape)
        #x = self.trans(inputs)
        x_1 = F.relu(self.fc1(inputs), inplace=False) # (bs, n_agents, 128)
        # x_1 = F.relu(self.fc1(inputs), inplace=False) # (bs, n_agents, 128)
        x_2 = F.relu(self.fc2(x_1), inplace=False) # (bs, n_agents, 128)
        v_mix = self.v_mix(x_2) # Calculate v_mix (coma - v)
        # Calculate v_ex, r_explore
        x_v_ex = x_2.reshape(max_t, -1) #(bs, n_agents*128) 
        v_ex = self.v_ex(x_v_ex)
        x_r_in = self.r_in(x_2)
        r_in = 10 * F.tanh(x_r_in/10)
        return v_mix, v_ex, r_in

    def build_input_Critic(self, states, obs, actions, t=None, onehot=True):
        torch.autograd.set_detect_anomaly(True)
        # bs = self.batch_size # batch_size 
        max_t = actions.shape[0] if t is None else 1
        inputs=[]
        state_input = states if t is None else states[slice(t,t+1)]
        inputs.append(state_input) # state+obs: [bs, sequence, n_agents, state_size] 
        obs_input = obs if t is None else obs[slice(t,t+1)]
        inputs.append(obs_input)
        onehot_actions = F.one_hot(actions.long(),num_classes=self.n_actions)
        action_input = onehot_actions if t is None else onehot_actions[slice(t,t+1)]
        agent_mask = (1 - torch.eye(self.n_agents, device=self.device))
        agent_mask = agent_mask.view(-1,1).repeat(1,self.n_actions).view(self.n_agents,-1)
        action_input = action_input.view(max_t, 1, -1).repeat(1,self.n_agents,1)
        
        inputs.append(action_input * agent_mask.unsqueeze(0))
        if t == 0:
            inputs.append(torch.zeros_like(onehot_actions[0:1]).view(max_t, 1, -1).repeat(1,self.n_agents,1))
        elif isinstance(t, int):
            inputs.append(onehot_actions[slice(t-1,t)].view(max_t,1,-1).repeat(1,self.n_agents,1))
        else:
            last_actions = torch.cat([torch.zeros_like(onehot_actions[0:1]), onehot_actions[:-1]], dim=0)
            last_actions = last_actions.view(max_t, 1, -1).repeat(1,self.n_agents,1)
            inputs.append(last_actions)
        inputs.append(torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(max_t, -1, -1))
        critic_input = torch.cat([x.reshape(max_t, self.n_agents, -1) for x in inputs], dim=-1)
        del action_input, onehot_actions, agent_mask
        return critic_input
        
    def create_input_size(self, onehot=True):
        act_size = self.n_actions * self.n_agents * 2 if onehot else 1
        input_size = self.state_size + self.obs_size + act_size + self.n_agents # + self.n_agents
        return input_size