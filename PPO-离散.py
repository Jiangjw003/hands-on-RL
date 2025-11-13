from rl_utils import *
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = F.softmax(self.fc_out(x), dim=-1)
        return y


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' 处理离散动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = Actor(state_dim, hidden_dim,
                           action_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.state_dim = state_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = state.to(self.device)
        action_probs = self.actor(state)
        # 这个Categorical是用于多项式分布的，创建的时候需要输入每一项的概率
        # 不使用argmax可以保留一定的探索性
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.cpu().numpy().item()

    def update(self, transition_dict):
        states = torch.stack(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 优势函数的计算可以看着是td-error除了最后一项，GAE在一个trajectory中，使用每一项的优势函数递推计算
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        # 是指定action下的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).squeeze(1).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions)).squeeze(1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 100000
hidden_dim = 128
gamma = 0.999
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
# env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
            lmbda, epochs, eps, gamma, device)

return_list = train_on_policy_agent(env, agent, num_episodes, state_dim=state_dim)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

test_agent(env, agent, state_dim)
