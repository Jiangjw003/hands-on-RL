import time
import torch
from torch import nn
from torch.distributions import Normal
from rl_utils import *
import gymnasium as gym
from torch.utils.data import TensorDataset, DataLoader


# 连续状态动作空间，首先动作的均值和标准差，然后构建分布后采样和计算log_probs
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.l1(state))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-8
        dist = Normal(mu, sigma)
        normal_sample = dist.rsample()  # 这里的rsample()是重参数化采样，注意不是sample
        log_prob = dist.log_prob(normal_sample)
        action = F.tanh(normal_sample)
        # 最终的action是经过tanh激活的，需要重新计算对应的概率
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, num_random, beta):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 策略网络
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.num_random = num_random  # 正则项中每个state采样action个数
        self.beta = beta  # 正则项强度

    def take_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action.squeeze().cpu().detach().numpy()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -torch.sum(log_prob, dim=1, keepdim=True)  # 多维动作要求log的sum，每个维度的概率乘积才是这个动作的概率
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 两个Q网络的TD-error项损失
        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        # 这里还需要计算正则项

        # 这里还需要
        # next_actions, log_prob = self.actor(next_states)
        # 以上与SAC相同,以下Q网络更新是CQL的额外部分
        batch_size = states.shape[0]
        # 第一项是均匀分布的action，使用torch.rand.uniform_生成
        random_unif_actions = torch.rand(
            [batch_size * self.num_random, actions.shape[-1]],
            dtype=torch.float).uniform_(-1, 1).to(device)
        # 由于是均匀概率，其log 概率分布如下，
        random_unif_log_pi = np.log(0.5 ** actions.shape[-1])
        # 扩充维度，和上面均匀概率分布一样，一个state采样num_random个动作
        # 不过这里的状态都是采样得到的状态，然后使用当前策略生成动作
        tmp_states = states.unsqueeze(1).repeat(1, self.num_random,
                                                1).view(-1, states.shape[-1])
        tmp_next_states = next_states.unsqueeze(1).repeat(
            1, self.num_random, 1).view(-1, next_states.shape[-1])
        # 获取当前策略生成当前状态和下一状态的动作概率分布log值
        random_curr_actions, random_curr_log_pi = self.actor(tmp_states)
        random_next_actions, random_next_log_pi = self.actor(tmp_next_states)
        # 计算整体mu的三个构成情况下的动作Q值
        q1_unif = self.critic_1(tmp_states, random_unif_actions)
        q2_unif = self.critic_2(tmp_states, random_unif_actions)
        q1_curr = self.critic_1(tmp_states, random_curr_actions)
        q2_curr = self.critic_2(tmp_states, random_curr_actions)
        q1_next = self.critic_1(tmp_states, random_next_actions)
        q2_next = self.critic_2(tmp_states, random_next_actions)
        # 减去概率分布的log值是修改采样带来的误差
        q1_cat = torch.cat([
            q1_unif - random_unif_log_pi,
            q1_curr - random_curr_log_pi.detach(),
            q1_next - random_next_log_pi.detach()
        ],
            dim=1)
        q2_cat = torch.cat([
            q2_unif - random_unif_log_pi,
            q2_curr - random_curr_log_pi.detach(),
            q2_next - random_next_log_pi.detach()
        ],
            dim=1)
        # 取log sum exp是为正则化中第一项对OOD数据的Q值下压
        qf1_loss_1 = torch.logsumexp(q1_cat, dim=1).mean()
        qf2_loss_1 = torch.logsumexp(q2_cat, dim=1).mean()
        # 正则化第二项是对数据集中action的修正，防止估计数据集内的action被下压
        qf1_loss_2 = self.critic_1(states, actions).mean()
        qf2_loss_2 = self.critic_2(states, actions).mean()
        # q的整体损失为TD-error项+正则项
        qf1_loss = critic_1_loss + self.beta * (qf1_loss_1 - qf1_loss_2)
        qf2_loss = critic_2_loss + self.beta * (qf2_loss_1 - qf2_loss_2)
        # 使用正则化后的q_loss更新参数，由于存在相同的计算tensor，有相同的计算图，需要报错计算图，防止重复计算
        self.critic_1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


env_name = 'HalfCheetah-v5'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4
critic_lr = 1e-3
alpha_lr = 1e-4
num_episodes = 1000
hidden_dim = 512
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 2000000
minimal_size = 1000
batch_size = 1024
beta = 1.0
num_random = 5
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = ReplayBuffer(buffer_size)
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device, num_random, beta)

# 首先加载预训练模型
dataset_id = "mujoco/halfcheetah/expert-v0"
have_trained_model = True
if have_trained_model:
    agent.actor.load_state_dict(torch.load('trained_actors/bl-halfcheetah-CQL.pth'))
else:
    s, a = load_expert_data_sa(dataset_id)
    dataset = TensorDataset(s, a)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    num_epochs = 1
    for epoch in range(num_epochs):
        step = 0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            preds, _ = agent.actor(batch_states)
            loss = F.mse_loss(preds, batch_actions)
            agent.actor_optimizer.zero_grad()
            loss.backward()
            agent.actor_optimizer.step()
            print(f'epoch{epoch},step:{step},loss:{loss.item()}')
    torch.save(agent.actor.state_dict(), 'trained_actors/bl-halfcheetah-CQL.pth')
# 尝试与环境交互
# test_env = gym.make(env_name, render_mode='human')
#
# s, _ = test_env.reset()
# done = False
# while not done:
#     action = agent.take_action(s)
#     ns, reward, terminated, truncated, info = test_env.step(action)
#     done = terminated or truncated
#     s = ns
#     time.sleep(0.02)
# test_env.close()
load_expert_data_sarsa(dataset_id, replay_buffer)

return_list = train_off_policy_agent(env, agent, num_episodes,
                                     replay_buffer, minimal_size,
                                     batch_size)
torch.save(agent.actor.state_dict(), 'trained_actors/CQL-机器狗.pth')
# env = gym.make(env_name, render_mode='human')
#
# test_agent(env, agent)
