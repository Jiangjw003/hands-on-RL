"""
对于冰冻湖任务中，ppo在线学习不能很好拟合，还是探索性不够，还有初始的策略太拉
没法得到好的经验，策略没法提升。
这里可以看地图后给出一定的(s,a)对预训练ppo的策略网络
然后在这个基础上在进行最后一步的强化学习训练

 "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
按照地图，专家经验就直接给出一个从s到g的最短路线策略
"""

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


actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.99999
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.manual_seed(0)
env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=True)
state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
            lmbda, epochs, eps, gamma, device)

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

            expert_datas.append((state_tensor,action_tensor))
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
    return expert_datas

# 采样专家数据
expert_data = sample_expert_data(10000)

pre_train(agent.actor, agent.actor_optimizer, expert_data, device, 5000)

return_list = train_on_policy_agent_discrete(env, agent, num_episodes, state_dim)

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=True, render_mode='human')
# 可以看出，在很少的训练情况下能达到不错的效果
test_agent(env, agent, state_dim)

for step in range(100):
    done = False
    s,_ = env.reset()
    s = F.one_hot(torch.tensor(s, dtype=torch.long), num_classes=state_dim).float()
    rewards = []
    while not done:
        action = agent.take_action(s)
        ns, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        s = ns
        s = F.one_hot(torch.tensor(s, dtype=torch.long), num_classes=state_dim).float()
        done = terminated or truncated
    print(f'step:{step},avg_reward:{np.mean(np.array(rewards))}')