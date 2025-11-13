import torch
from torch import nn
import gymnasium as gym
from rl_utils import *


# 决策模型，用于判断数据是专家还是智能体生成的
class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device).unsqueeze(1)
        agent_states = torch.stack(agent_s).float().to(self.device)
        agent_actions = torch.tensor(agent_a).float().to(self.device).unsqueeze(1)

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
            expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)


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
        self.alpha = 0.001

    def take_action(self, state):
        state = state.to(self.device)
        action_probs = self.actor(state)
        # 这个Categorical是用于多项式分布的，创建的时候需要输入每一项的概率
        # 不使用argmax可以保留一定的探索性
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.cpu().numpy().item()

    def update(self, transition_dict):
        states = torch.stack(transition_dict['states']).to(self.device).float()
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(transition_dict['next_states']).to(self.device).float()
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 优势函数的计算可以看着是td-error除了最后一项，GAE在一个trajectory中，使用每一项的优势函数递推计算
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        # 是指定action下的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).squeeze(1).detach()

        for _ in range(self.epochs):
            action_probs = self.actor(states)
            log_action_probs = torch.log(action_probs)
            entropy = - torch.sum(action_probs * log_action_probs, dim=1)  # 熵，用于加大探索，求sum就是当前动作的熵
            entropy = torch.mean(entropy)  # 取mean是为了标量才能进行反向传播梯度
            log_probs = log_action_probs.gather(1, actions).squeeze(1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.alpha * entropy
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=True)

state_dim, action_dim = env.observation_space.n, env.action_space.n
hidden_dim = 128
actor_lr = 1e-4
critic_lr = 1e-3
lmbda = 0.95
epochs = 10
gamma = 0.99999
eps = 0.2
lr_d = 1e-3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
gail = GAIL(agent, state_dim, 1, hidden_dim, lr_d, device)
n_episode = 500
return_list = []

best_policy = [2, 3, 0, 3, 0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 0]


def sample_expert_data(n_episode):
    expert_datas = []
    for episode in range(n_episode):
        state, _ = env.reset()
        done = False
        while not done:
            action = best_policy[state]
            state_tensor = F.one_hot(torch.tensor(state), num_classes=state_dim)
            action_tensor = torch.tensor(action)

            expert_datas.append((state_tensor, action_tensor))
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
    return expert_datas


# 采样专家数据
expert_data = sample_expert_data(100000)

expert_s = torch.stack([item[0].float() for item in expert_data]).to(device)
expert_a = torch.stack([item[1] for item in expert_data]).to(device)

with tqdm(total=n_episode, desc="进度条") as pbar:
    for i in range(n_episode):
        episode_return = 0
        state, _ = env.reset()
        done = False
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        state_tensor = F.one_hot(torch.tensor(state), num_classes=state_dim).float().to(device)
        while not done:
            action = agent.take_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or truncated
            next_state_tensor = F.one_hot(torch.tensor(next_state), num_classes=state_dim).float().to(device)
            state_list.append(state_tensor)
            action_list.append(action)
            next_state_list.append(next_state_tensor)
            done_list.append(done)
            state = next_state
            state_tensor = next_state_tensor
            episode_return += reward
        return_list.append(state == state_dim - 1)
        gail.learn(expert_s, expert_a, state_list, action_list,
                   next_state_list, done_list)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

# 进度条: 100%|██████████| 500/500 [04:08<00:00,  2.01it/s, return=200.000]
