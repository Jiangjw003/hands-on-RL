import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from pettingzoo.butterfly import pistonball_v6
from tqdm import tqdm

from rl_utils import ReplayBufferMa


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        """
        Ornstein-Uhlenbeck过程噪声

        Args:
            action_dimension: 动作空间的维度
            mu: 均值
            theta: 回归速度参数
            sigma: 波动率参数
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        """生成噪声样本"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=256):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，使任意图像的形状转换为(1,1)，通道不变，就是取平均
        )

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 转换形状 b,channels,1,1 -> b,channels
        x = self.fc(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, device):
        super().__init__()
        self.img_process = SimpleCNN(input_channels=3, output_dim=obs_dim).to(device)
        self.f1 = TwoLayerFC(obs_dim, hidden_dim, hidden_dim).to(device)
        self.f2 = TwoLayerFC(hidden_dim, action_dim, hidden_dim).to(device)

    def forward(self, obs):
        """
        :param obs:  智能体的观测，观测是一个图像数据，b,c,h,w
        :return:
        """
        x = self.img_process(obs)
        x = F.relu(self.f1(x))
        return F.tanh(self.f2(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agent, hidden_dim, device):
        super().__init__()
        # 每次输入是所有的观测在通道数堆叠
        self.img_process = SimpleCNN(input_channels=3 * num_agent, output_dim=state_dim).to(device)
        self.f1 = TwoLayerFC(state_dim + num_agent * action_dim, hidden_dim, hidden_dim).to(
            device)  # 全局state + 全部action
        self.f2 = TwoLayerFC(hidden_dim, 1, hidden_dim).to(device)

    def forward(self, state, actions):
        x = self.img_process(state)
        x = torch.cat((x, actions), 1)
        x = F.relu(self.f1(x))
        return self.f2(x)


ou_noise = OUNoise(1)
device_dis = [0, 1]


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, obs_dim, state_dim, action_dim, num_agent, hidden_dim,
                 actor_lr, critic_lr, tau, device):
        self.actor = Actor(obs_dim, action_dim, hidden_dim, device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim, device).to(device)
        self.critic = Critic(state_dim, action_dim, num_agent, hidden_dim, device)
        self.target_critic = Critic(state_dim, action_dim, num_agent, hidden_dim, device).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.tau = tau

    def take_action(self, state, explore=True):
        action = self.actor(state)
        if explore:
            noise = torch.from_numpy(ou_noise.sample()).to(action.device, dtype=action.dtype)  # 先统一为tensor
            action = torch.clip(action + noise, min=-1, max=1)
        return action.detach().cpu().numpy()[0]

    def soft_update(self):
        for param_target, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
        for param_target, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)


# 修改 MADDPG 类，添加缺失的属性和方法
# 模型较大，使用多卡训练
class MADDPG(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, num_agents, max_action, actor_lr, critic_lr, hidden_dim,
                 gamma, tau, device):
        super().__init__()
        self.agents = [
            DDPG(obs_dim, state_dim, action_dim, num_agents, hidden_dim, actor_lr, critic_lr, tau, device) for _
            in range(num_agents)]

        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = 0.5

        # 为优化器创建列表
        self.actor_optimizers = [agent.actor_optimizer for agent in self.agents]
        self.critic_optimizers = [agent.critic_optimizer for agent in self.agents]

        # 为网络创建列表以便访问
        self.actors = [agent.actor for agent in self.agents]
        self.target_actors = [agent.target_actor for agent in self.agents]

    def select_action(self, obs, explore=False):
        actions = []
        for i in range(self.num_agents):
            # 确保obs[i]是tensor格式
            if isinstance(obs[i], np.ndarray):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            else:
                obs_tensor = obs[i]
            action = self.agents[i].take_action(obs_tensor, explore)
            actions.append(action)
        return actions

    def update(self, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones):
        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)  # [B, N, C, H, W]
        actions = torch.tensor(np.array(batch_actions), dtype=torch.float32, device=device)  # [B, N, A]
        rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device)  # [B, N, 1]
        next_obs = torch.tensor(np.array(batch_next_obs), dtype=torch.float32, device=device)  # [B, N, C, H, W]
        dones = torch.tensor(np.array(batch_dones), dtype=torch.float32, device=device)  # [B, N, 1]

        B, N = obs.size(0), obs.size(1)

        obs_list = [obs[:, i] for i in range(N)]
        next_obs_list = [next_obs[:, i] for i in range(N)]
        act_list = [actions[:, i] for i in range(N)]
        rew_list = [rewards[:, i].squeeze(-1) for i in range(N)]  # [B]
        done_list = [dones[:, i].squeeze(-1) for i in range(N)]  # [B]

        with torch.no_grad():
            next_act_list = []
            for i, agent in enumerate(self.agents):
                next_act_i = agent.target_actor(next_obs_list[i])  # [B, act_dim]
                next_act_list.append(next_act_i)
            # 拼成 [B, N*act_dim]
            next_actions_cat = torch.cat(next_act_list, dim=1)

        # 3. 并行更新每个 agent
        for i, agent in enumerate(self.agents):
            # 3.1 Critic 更新
            # 当前 Q_i(s,a)
            # torch.cat(obs_list, dim=1) 在通道数维度上堆叠 B N*C W H
            actions_cat = torch.cat(act_list, dim=1)  # [B, N*act_dim]
            q_current = agent.critic(torch.cat(obs_list, dim=1), actions_cat)  # [B, 1]

            # 目标 Q_i(s',a')
            next_obs_cat = torch.cat(next_obs_list, dim=1)  # [B, N*C,W,H]
            q_next = agent.target_critic(next_obs_cat, next_actions_cat)  # [B, 1]
            q_target = rew_list[i].unsqueeze(1) + gamma * q_next * (1 - done_list[i].unsqueeze(1))

            critic_loss = F.mse_loss(q_current, q_target.detach())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            # 3.2 Actor 更新
            # 重新采样当前策略动作（需要反向传播）
            act_i_new = agent.actor(obs_list[i])  # [B, act_dim]
            act_list_new = act_list.copy()
            act_list_new[i] = act_i_new
            actions_cat_new = torch.cat(act_list_new, dim=1)

            actor_loss = -agent.critic(torch.cat(obs_list, dim=1), actions_cat_new).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()


# 环境初始化
env = pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True,
                        random_drop=False, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
                        ball_elasticity=1.5, max_cycles=500)
env.reset(seed=42)
# 参数设置
obs_dim = 512
state_dim = 1024
action_dim = 1
hidden_dim = 512
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.95
tau = 1e-2
batch_size = 512
num_agents = env.num_agents
max_action = 1.0
capacity = 100000
num_episode = 1000
minimal_size = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化MADDPG智能体和经验回放缓冲区
maddpg = MADDPG(obs_dim, state_dim, action_dim, num_agents, max_action, actor_lr, critic_lr, hidden_dim, gamma, tau,
                device)
replay_buffer = ReplayBufferMa(capacity)


# 训练循环
def preprocess_observation(obs):
    """预处理观测数据"""
    # 将观测转换为适合网络输入的格式
    if isinstance(obs, dict):
        # 如果有字典格式的观测，提取图像部分
        obs = obs['observation'] if 'observation' in obs else obs
    return obs


agent_names = env.agents

for i in range(10):
    with tqdm(total=int(num_episode / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episode / 10)):
            env.reset(seed=42)
            done = False
            total_reward_list = []
            while not done:
                # 每一次遍历一遍全部的智能体,记录对应的观测信息,还有action
                obs_list = []
                for agent_name in agent_names:
                    obs = env.observe(agent_name)
                    obs = obs.transpose(2, 0, 1)
                    obs_list.append(obs)
                action_list = maddpg.select_action(obs_list, explore=True)
                # 转换为每个智能体对应action，用于环境执行
                action_dict = {agent_name: action for agent_name, action in zip(agent_names, action_list)}
                # print('env.agents :', env.agents)
                # print('dict keys  :', list(action_dict.keys()))
                # print(type(list(action_dict.values())[0]))
                #  env.step(action_dict) 这个得用并行环境接口，比较麻烦，没法调用查看某个智能体的观测
                # 环境默认的活动智能体循环严格按照顺序的
                # 也可以调用current_agent = env.agent_selection获取当前环境活动的agent
                # 这里只需要一个agent出现两次退出执行即可
                used_agent = set()
                while True:
                    current_agent = env.agent_selection
                    if current_agent in used_agent:
                        break
                    env.step(action_dict[current_agent])
                    used_agent.add(current_agent)
                #
                reward_dict = env.rewards
                reward_list = list(reward_dict.values())
                total_reward_list.extend(list(reward_dict.values()))
                next_obs_list = []
                for agent_name in agent_names:
                    next_obs = env.observe(agent_name)
                    next_obs = next_obs.transpose(2, 0, 1)
                    next_obs_list.append(next_obs)
                terminated, truncations = env.terminations, env.truncations
                done_dict = {}
                for agent_name in agent_names:
                    done_dict[agent_name] = terminated[agent_name] or truncations[agent_name]
                done_list = list(done_dict.values())
                # 经验放入经验池
                replay_buffer.add(obs_list, action_list, reward_list, next_obs_list, done_list)
                # 结束标志，当所有智能体达到终点条件或者达到截断条件
                done = all(terminated.values()) or all(truncations.values())
            if replay_buffer.size() >= minimal_size:
                # 返回的都是字典类型的列表，有每个智能体对应的值
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                maddpg.update(states, actions, rewards, next_states, dones)
                # 记录胜负
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episode / 10 * i + i_episode + 1),
                    'avg_reward': '%.3f' % np.mean(np.array(total_reward_list))
                })
            pbar.update(1)
