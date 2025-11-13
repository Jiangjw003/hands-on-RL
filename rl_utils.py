import minari
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import keyboard
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class ReplayBufferMa:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_on_policy_agent_discrete(env, agent, num_episodes, state_dim):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                state = F.one_hot(torch.tensor(state), state_dim).float()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    if terminated:
                        if next_state != state_dim - 1:  # 掉进洞的reward不能为0，而是要惩罚
                            reward = -1
                    next_state = F.one_hot(torch.tensor(next_state), state_dim).float()
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=0)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s,
                                           'actions': b_a,
                                           'next_states': b_ns,
                                           'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def test_agent(env, agent, state_dim=None):
    step = 0
    try:
        while True:
            rewards = []
            s, _ = env.reset()
            if state_dim is not None:
                s = F.one_hot(torch.tensor(s, dtype=torch.long), num_classes=state_dim).float()
            done = False

            while not done:
                # 检测ESC键
                if keyboard.is_pressed('esc'):
                    print("ESC pressed, exiting...")
                    env.close()
                    return

                action = agent.take_action(s)
                ns, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                s = ns
                if state_dim is not None:
                    s = F.one_hot(torch.tensor(s, dtype=torch.long), num_classes=state_dim).float()
                done = terminated or truncated
            print(f'step:{step},avg_reward:{np.mean(np.array(rewards))}')
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()


def pre_train(actor_model, actor_optim, expert_data, device, num_epoch):
    x_data = torch.stack([item[0].float() for item in expert_data]).to(device)
    y_data = torch.stack([item[1] for item in expert_data]).to(device)
    for epoch in range(num_epoch):
        pred_y = actor_model(x_data)
        l = F.cross_entropy(pred_y, y_data)
        actor_optim.zero_grad()
        l.backward()
        actor_optim.step()
        if (epoch + 1) % 100 == 0:
            print(f"epoch:{epoch + 1},loss:{l.item()}")


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }


def load_expert_data_sa(dataset_id):
    if dataset_id not in minari.list_local_datasets():
        print(f"\n下载数据集 {dataset_id}...")
        minari.download_dataset(dataset_id)
    else:
        print(f"\n数据集 {dataset_id} 已存在")

    # 加载数据集
    state_list = []
    action_list = []
    dataset = minari.load_dataset(dataset_id)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        states = batch['observations']
        actions = batch['actions']
        tra_num = states.shape[0]
        for tra in range(tra_num):
            tra_length = actions[tra].shape[0]  # 注意，state是要比action多一个的
            for step in range(tra_length):
                state_list.append(states[tra][step].float())
                action_list.append(actions[tra][step].float())
    states = torch.stack(state_list)
    actions = torch.stack(action_list)
    return states, actions


def load_expert_data_sarsa(dataset_id, replay_buffer_expert):
    if dataset_id not in minari.list_local_datasets():
        print(f"\n下载数据集 {dataset_id}...")
        minari.download_dataset(dataset_id)
    else:
        print(f"\n数据集 {dataset_id} 已存在")
    dataset = minari.load_dataset(dataset_id)
    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
    for episode in dataset.iterate_episodes():
        # 3. 从回合对象中提取五元组数据
        observations = episode.observations[:-1]  # 状态序列 (s0, s1, s2, ..., sT)
        actions = episode.actions  # 动作序列 (a0, a1, a2, ..., aT-1)
        rewards = episode.rewards  # 奖励序列 (r1, r2, r3, ..., rT)
        terminations = episode.terminations  # 终止标志序列
        truncations = episode.truncations  # 截断标志序列
        next_observations = episode.observations[1:]
        done_flags = []
        for i in range(len(actions)):  # 对于每个时间步
            # 如果当前步终止或截断，则done为True
            done = terminations[i] or truncations[i]
            done_flags.append(done)
        all_states.extend(observations)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_dones.extend(done_flags)
        all_next_states.extend(next_observations)

    # 最后经验全部进入经验池
    for s, a, r, ns, d in zip(all_states, all_actions, all_rewards, all_next_states, all_dones):
        replay_buffer_expert.add(s, a, r, s, d)
