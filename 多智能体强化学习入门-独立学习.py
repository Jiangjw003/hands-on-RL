import time
import torch
import torch.nn.functional as F
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm
import numpy as np
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, action_mask=None):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)

        if action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
            probs = probs * action_mask.float()
            # 避免除零错误
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = action_mask.float() / action_mask.float().sum()

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


# 超参数设置
actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 10000
hidden_dim = 128
gamma = 0.9
lmbda = 0.97
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化环境
env = tictactoe_v3.env()

# 获取状态和动作维度
state_dim = 3 * 3 * 2  # 棋盘展平
action_dim = 9  # 9个位置

# 初始化PPO智能体
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

# 写一个不共享参数的版本
agent1 = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)
agent2 = PPO(state_dim, hidden_dim, action_dim, actor_lr * 5, critic_lr * 5, lmbda, eps, gamma, device)
agents = {'player_1': agent1, 'player_2': agent2}
# 训练循环
win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_dicts = {
                'player_1': {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []},
                'player_2': {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            }

            env.reset(seed=42)
            current_states = {'player_1': None, 'player_2': None}

            for agent_name in env.agent_iter():
                """
                注意这里返回值很巧妙
                observation是上一个智能体执行结束后，当前智能体的观测
                reward是上一个智能体执行动作后，使当前智能体得到的奖励，这个和单智能体环境不同
                termination, truncation是当前智能体是否结束信息
                info是当前智能体的调试信息
                都打包为当前智能体的信息返回给用户
                
                游戏流程是：
                首先轮流枚举可以行动的agent，然后使用last获取该agent的信息，这里在游戏结束的时候将所有智能体都枚举一遍。
                如果游戏已经结束了，action设为None，其上一次游戏没有结束时候的reward因为设置为当前获取的reward
                如果没有结束，获取agent选择的action，记录state,action,reward,done信息
                环境中执行当前智能体的action
                
                由于游戏结束后还会再枚举一遍智能体，那么这个state就是智能体从开始到结束的所有状态
                那么获取next_state的时候只需要偏移即可，对于最后一个状态，实际TD-target为(1-done)*critic(next_state)
                所以并不会实际学习最后一个状态，使用最后一个state重复即可
                """
                observation, reward, termination, truncation, info = env.last()
                state = observation['observation'].flatten()  # 展平3x3棋盘
                action_mask = observation['action_mask']

                if termination or truncation:
                    action = None
                    # 给上一个状态添加最终奖励，
                    if current_states[agent_name] is not None:
                        if agent_name == 'player_2':
                            if reward == 1:
                                reward = 10
                        transition_dicts[agent_name]['rewards'][-1] += reward
                else:
                    current_states[agent_name] = state
                    action = agents[agent_name].take_action(state, action_mask)
                    # 存储当前状态和动作
                    transition_dicts[agent_name]['states'].append(state)
                    transition_dicts[agent_name]['actions'].append(action)
                    transition_dicts[agent_name]['rewards'].append(-0.1)  # 初始奖励为0，这里防止智能体绕路，奖励为-0.1，卧槽先手胜率直接到1
                    transition_dicts[agent_name]['dones'].append(False)
                env.step(action)

            # 更新网络
            for player in ['player_1', 'player_2']:
                if len(transition_dicts[player]['states']) > 0:
                    states = transition_dicts[player]['states']
                    transition_dicts[player]['next_states'] = states[1:] + [states[-1]]
                    agents[player].update(transition_dicts[player])

            # 记录胜负
            win_list.append(1 if any(transition_dicts['player_1']['rewards']) > 0 else 0)
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    '先手win_rate': '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)

for _ in range(10):
    env = tictactoe_v3.env(render_mode='human')
    env.reset(seed=42)
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if reward == 1:
            print('agent:', agent_name, 'win!!!')
        if reward == -1:
            print('agent:', agent_name, 'lose!!!')
        state = observation['observation'].flatten()  # 展平3x3棋盘
        action_mask = observation['action_mask']
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = agents[agent_name].take_action(state, action_mask)
        env.step(action)
        time.sleep(0.1)
    env.close()
