import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from pettingzoo.butterfly import pistonball_v6
from tqdm import tqdm
from rl_utils import ReplayBufferMa  # 假设存在且接口与原来相同

# --------------------------
# Ornstein-Uhlenbeck 噪声
# --------------------------
class OUNoise:
    def __init__(self, action_dimension, mu=0., theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension, dtype=np.float32) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)).astype(np.float32)
        self.state = x + dx
        return self.state

# --------------------------
# 简单 MLP / CNN 模块
# --------------------------
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.fc(x)
        return x

# --------------------------
# Actor / Critic 结构
# --------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, device):
        super().__init__()
        # img_process 输出 obs_dim 维特征
        self.img_process = SimpleCNN(input_channels=3, output_dim=obs_dim).to(device)
        self.f1 = TwoLayerFC(obs_dim, hidden_dim, hidden_dim).to(device)
        self.f2 = TwoLayerFC(hidden_dim, action_dim, hidden_dim).to(device)

    def forward(self, obs):
        # obs: [B, C, H, W]
        x = self.img_process(obs)
        x = F.relu(self.f1(x))
        return torch.tanh(self.f2(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agent, hidden_dim, device):
        super().__init__()
        # 将所有 agent 的图像通道按通道维堆叠（3 * num_agent）
        self.img_process = SimpleCNN(input_channels=3 * num_agent, output_dim=state_dim).to(device)
        # critic 接收 state 特征 + 全体 actions
        self.f1 = TwoLayerFC(state_dim + num_agent * action_dim, hidden_dim, hidden_dim).to(device)
        self.f2 = TwoLayerFC(hidden_dim, 1, hidden_dim).to(device)

    def forward(self, state, actions):
        # state: [B, 3*N, H, W] -> img_process -> [B, state_dim]
        x = self.img_process(state)
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.f1(x))
        return self.f2(x)

# --------------------------
# DDPG（单 agent）- 支持 DataParallel
# --------------------------
class DDPG:
    def __init__(self, obs_dim, state_dim, action_dim, num_agent, hidden_dim,
                 actor_lr, critic_lr, tau, device, device_ids=None):
        """
        device: torch.device 主设备 (e.g., torch.device("cuda:0"))
        device_ids: 用于 DataParallel 的 device id 列表 或 None
        """
        self.device = device
        self.device_ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
        # 创建网络（先在主设备上创建）
        self.actor = Actor(obs_dim, action_dim, hidden_dim, device).to(device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim, device).to(device)
        self.critic = Critic(state_dim, action_dim, num_agent, hidden_dim, device).to(device)
        self.target_critic = Critic(state_dim, action_dim, num_agent, hidden_dim, device).to(device)

        # 如果多卡则封装 DataParallel（确保模型先 .to(device)）
        if self.device_ids and len(self.device_ids) > 1:
            print(f"⚡ Wrapping actor/critic with DataParallel on devices {self.device_ids}")
            # 注意：DataParallel 要放在主设备（device）上
            self.actor = nn.DataParallel(self.actor, device_ids=self.device_ids)
            self.target_actor = nn.DataParallel(self.target_actor, device_ids=self.device_ids)
            self.critic = nn.DataParallel(self.critic, device_ids=self.device_ids)
            self.target_critic = nn.DataParallel(self.target_critic, device_ids=self.device_ids)

        # 将 target 初始参数同步
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器：注意 DataParallel 后 model.parameters() 仍然可用
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau

    def take_action(self, state_tensor, explore=True, ou_noise=None):
        """
        state_tensor: tensor [1, C, H, W] 且已在正确 device 上
        返回 numpy 数组，形状 (action_dim, ) 或 (1, action_dim)
        """
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)  # [1, action_dim]
        self.actor.train()
        if explore and (ou_noise is not None):
            noise = torch.from_numpy(ou_noise.sample()).to(action.device, dtype=action.dtype)
            # 如果 noise 是 (action_dim,), reshape 为 (1, action_dim)
            if noise.dim() == 1:
                noise = noise.unsqueeze(0)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def soft_update(self):
        """
        将 actor/critic 的参数软更新到 target_* 上。
        兼容 DataParallel（如果是 DataParallel，就用 .module.parameters()）
        """
        # 处理 actor
        actor_params = self.actor.module.parameters() if isinstance(self.actor, nn.DataParallel) else self.actor.parameters()
        target_actor_params = self.target_actor.module.parameters() if isinstance(self.target_actor, nn.DataParallel) else self.target_actor.parameters()
        for target_param, param in zip(target_actor_params, actor_params):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # 处理 critic
        critic_params = self.critic.module.parameters() if isinstance(self.critic, nn.DataParallel) else self.critic.parameters()
        target_critic_params = self.target_critic.module.parameters() if isinstance(self.target_critic, nn.DataParallel) else self.target_critic.parameters()
        for target_param, param in zip(target_critic_params, critic_params):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

# --------------------------
# MADDPG：多 agent 管理（使用多个 DDPG 实例）
# --------------------------
class MADDPG:
    def __init__(self, obs_dim, state_dim, action_dim, num_agents, max_action,
                 actor_lr, critic_lr, hidden_dim, gamma, tau, device, device_ids=None):
        """
        device: 主设备 (torch.device)
        device_ids: 列表或 None，用于 DataParallel
        """
        self.device = device
        self.device_ids = device_ids
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = 0.5

        # 为每个 agent 创建一个 DDPG 实例
        self.agents = [
            DDPG(obs_dim, state_dim, action_dim, num_agents, hidden_dim,
                 actor_lr, critic_lr, tau, device, device_ids)
            for _ in range(num_agents)
        ]

    def select_action(self, obs_list, explore=False, ou_noise_list=None):
        """
        obs_list: python list, len = num_agents, 每项为 numpy array 或 tensor
        ou_noise_list: optional list of OUNoise per agent (长度 num_agents)
        返回 actions: list, 每个元素为 numpy 数组 (action_dim,)
        """
        actions = []
        for i in range(self.num_agents):
            obs = obs_list[i]
            # 将 obs 转为 tensor 并放到主 device
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, C, H, W]
            else:
                obs_tensor = obs.to(self.device)
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
            ou_noise = (ou_noise_list[i] if ou_noise_list is not None else None)
            action = self.agents[i].take_action(obs_tensor, explore=explore, ou_noise=ou_noise)
            actions.append(action)
        return actions

    def update(self, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones):
        """
        batch_*: numpy arrays / nested lists from ReplayBufferMa.sample
        组织后转换成 tensor 并放到 self.device
        期望 shapes:
            batch_obs: [B, N, C, H, W]
            batch_actions: [B, N, action_dim]
            batch_rewards: [B, N, 1]
            batch_next_obs: [B, N, C, H, W]
            batch_dones: [B, N, 1]
        """
        device = self.device
        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)  # [B, N, C, H, W]
        actions = torch.tensor(np.array(batch_actions), dtype=torch.float32, device=device)  # [B, N, A]
        rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device)  # [B, N, 1]
        next_obs = torch.tensor(np.array(batch_next_obs), dtype=torch.float32, device=device)  # [B, N, C, H, W]
        dones = torch.tensor(np.array(batch_dones), dtype=torch.float32, device=device)  # [B, N, 1]

        B, N = obs.size(0), obs.size(1)

        # 拆分为每个 agent 的 list（每项 shape: [B, C, H, W] 或 [B, A]）
        obs_list = [obs[:, i] for i in range(N)]
        next_obs_list = [next_obs[:, i] for i in range(N)]
        act_list = [actions[:, i] for i in range(N)]  # [B, A]
        rew_list = [rewards[:, i].squeeze(-1) for i in range(N)]  # [B]
        done_list = [dones[:, i].squeeze(-1) for i in range(N)]  # [B]

        # -----------------------------
        # 计算 next actions（target actor）
        # -----------------------------
        with torch.no_grad():
            next_act_list = []
            for i, agent in enumerate(self.agents):
                # 需要把 next_obs_list[i] 给 target_actor，target_actor 是 DataParallel 或 module
                target_actor = agent.target_actor
                next_act_i = target_actor(next_obs_list[i])  # [B, A]
                next_act_list.append(next_act_i)
            next_actions_cat = torch.cat(next_act_list, dim=1)  # [B, N*A]

        # -----------------------------
        # 对每个 agent 并行更新
        # -----------------------------
        for i, agent in enumerate(self.agents):
            # 准备 critic 输入：将所有 obs 按通道堆叠 -> [B, 3*N, H, W]
            # 注意 obs_list 每项为 [B, C, H, W]，直接在 channel 维度拼接
            obs_cat = torch.cat(obs_list, dim=1)  # [B, 3*N, H, W] (因为每 obs 的 channel = 3)
            actions_cat = torch.cat(act_list, dim=1)  # [B, N*A]

            # 当前 Q 值
            q_current = agent.critic(obs_cat, actions_cat)  # [B, 1]

            # 目标 Q 值
            next_obs_cat = torch.cat(next_obs_list, dim=1)
            q_next = agent.target_critic(next_obs_cat, next_actions_cat)  # [B, 1]
            q_target = rew_list[i].unsqueeze(1) + self.gamma * q_next * (1 - done_list[i].unsqueeze(1))

            # Critic loss & update
            critic_loss = F.mse_loss(q_current, q_target.detach())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.max_grad_norm)
            agent.critic_optimizer.step()

            # Actor 更新：用 actor 重新生成该 agent 的动作，其它 agent 的动作使用旧的动作（act_list）。
            act_i_new = agent.actor(obs_list[i])  # [B, A]
            act_list_new = act_list.copy()
            act_list_new[i] = act_i_new
            actions_cat_new = torch.cat(act_list_new, dim=1)  # [B, N*A]

            actor_loss = -agent.critic(obs_cat, actions_cat_new).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
            agent.actor_optimizer.step()

            # soft update target nets (在内部处理 DataParallel)
            agent.soft_update()

# --------------------------
# 主训练脚本
# --------------------------
def preprocess_observation(obs):
    """
    原始 obs 是 numpy 的 (H, W, C)，环境中可能提供 dict 等格式
    返回 (C, H, W) 的 numpy float32
    """
    if isinstance(obs, dict):
        # 若存在 'observation' 这一 key
        obs = obs.get('observation', obs)
    # 有些 obs 可能是 (H,W,C) 或者 (C,H,W)，确保 (H,W,C)
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 3 and obs.shape[2] == 3:
        obs = obs.transpose(2, 0, 1)  # -> (C, H, W)
    elif obs.ndim == 3 and obs.shape[0] == 3:
        # 已经是 (C,H,W)
        pass
    else:
        # 处理其他情况（保守处理）
        obs = obs.transpose(2, 0, 1)
    return obs

if __name__ == "__main__":
    # --------------------------
    # 环境与参数
    # --------------------------
    env = pistonball_v6.env(n_pistons=20, time_penalty=-0.1, continuous=True,
                            random_drop=False, random_rotate=True, ball_mass=0.75,
                            ball_friction=0.3, ball_elasticity=1.5, max_cycles=500)
    env.reset(seed=42)

    obs_dim = 64
    state_dim = 128
    action_dim = 1
    hidden_dim = 32
    actor_lr = 1e-4
    critic_lr = 1e-3
    gamma = 0.95
    tau = 1e-2
    batch_size = 32
    num_agents = env.num_agents
    max_action = 1.0
    capacity = 10000
    num_episode = 1000
    minimal_size = 1000

    # 设备与多卡配置
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_ids = list(range(n_gpus)) if n_gpus > 1 else None
        main_device = torch.device("cuda:0")
        print("Using device:", main_device, "GPU count:", n_gpus)
    else:
        device_ids = None
        main_device = torch.device("cpu")
        print("Using CPU")

    maddpg = MADDPG(obs_dim, state_dim, action_dim, num_agents, max_action,
                    actor_lr, critic_lr, hidden_dim, gamma, tau, main_device, device_ids=device_ids)

    replay_buffer = ReplayBufferMa(capacity)

    # 为每个 agent 创建 OU noise（可选）
    ou_noises = [OUNoise(action_dim) for _ in range(num_agents)]

    agent_names = env.agents

    # 训练循环（原版结构保留，分为若干个迭代块以显示 tqdm）
    outer_iters = 10
    episodes_per_iter = int(num_episode / outer_iters)

    for outer in range(outer_iters):
        with tqdm(total=episodes_per_iter, desc=f'Iteration {outer}') as pbar:
            for i_episode in range(episodes_per_iter):
                env.reset(seed=42)
                # 将环境回合变量清理
                total_reward_list = []
                done = False

                # 重置 OU 噪声
                for n in ou_noises:
                    n.reset()

                # 由于 PettingZoo 的 step 是轮流 agent，使用 while not done 来推进
                while True:
                    obs_list = []
                    for agent_name in agent_names:
                        obs = env.observe(agent_name)
                        obs = preprocess_observation(obs)  # -> (C,H,W)
                        obs_list.append(obs)

                    # 选择动作（所有 agent）
                    actions = maddpg.select_action(obs_list, explore=True, ou_noise_list=ou_noises)

                    # 构建 action dict 并 step 环境（使用 env.agent_selection 循环）
                    action_dict = {agent_name: action for agent_name, action in zip(agent_names, actions)}

                    used_agent = set()
                    # 迭代步：依照环境活动智能体逐一 step，直到回到已处理的 agent
                    while True:
                        current_agent = env.agent_selection
                        if current_agent in used_agent:
                            break
                        # PettingZoo 期望 action 的形状与 action_space 一致
                        env.step(action_dict[current_agent])
                        used_agent.add(current_agent)

                    # 收集 reward / next_obs / done
                    reward_dict = env.rewards
                    total_reward_list.extend(list(reward_dict.values()))

                    next_obs_list = []
                    for agent_name in agent_names:
                        next_obs = env.observe(agent_name)
                        next_obs = preprocess_observation(next_obs)
                        next_obs_list.append(next_obs)

                    terminated, truncations = env.terminations, env.truncations
                    done_dict = {}
                    for agent_name in agent_names:
                        done_dict[agent_name] = terminated[agent_name] or truncations[agent_name]
                    done_list = list(done_dict.values())

                    # 经验放入回放池
                    # 注意：ReplayBufferMa.add 的签名应与原先一致；这里传入 numpy list
                    replay_buffer.add(obs_list, actions, list(reward_dict.values()), next_obs_list, done_list)

                    # 终止条件：当所有智能体都终止或截断时结束回合
                    if all(terminated.values()) or all(truncations.values()):
                        break

                # 在每个回合结束后尝试学习
                if replay_buffer.size() >= minimal_size:
                    states, actions_b, rewards_b, next_states, dones_b = replay_buffer.sample(batch_size)
                    # 期望 sample 返回形状与 update 中注释相匹配：
                    # states: [B, N, C, H, W]
                    # actions_b: [B, N, A]
                    # rewards_b: [B, N, 1]
                    # next_states: [B, N, C, H, W]
                    # dones_b: [B, N, 1]
                    maddpg.update(states, actions_b, rewards_b, next_states, dones_b)

                # 日志输出
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode': f'{outer * episodes_per_iter + i_episode + 1}',
                        'avg_reward': f'{np.mean(np.array(total_reward_list)):.3f}'
                    })
                pbar.update(1)

    print("Training finished.")
