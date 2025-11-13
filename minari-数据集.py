import time
import minari
from torch.utils.data import DataLoader
import torch
from rl_utils import load_expert_data_sa
# 下载机器狗数据集
dataset_id = "mujoco/halfcheetah/expert-v0"

s,a = load_expert_data_sa(dataset_id)

if dataset_id not in minari.list_local_datasets():
    print(f"\n下载数据集 {dataset_id}...")
    minari.download_dataset(dataset_id)
else:
    print(f"\n数据集 {dataset_id} 已存在")

# 加载数据集
dataset = minari.load_dataset(dataset_id)
episode_ids = dataset.episode_indices

import minari

# 加载数据集
dataset = minari.load_dataset(dataset_id)

# 使用 iterate_episodes() 遍历所有回合
for episode in dataset.iterate_episodes():
    # episode 包含了一个完整回合的所有数据
    observations = episode.observations
    actions = episode.actions
    rewards = episode.rewards
    terminations = episode.terminations
    truncations = episode.truncations

    print(f"回合长度: {len(actions)}")
    print(f"状态形状: {observations[0].shape}")
    print(f"动作形状: {actions[0].shape}")
    break  # 只看第一个回合

# 查看数据集基本信息
print(f"\n数据集信息：")
print(f"  轨迹数量：{len(dataset)}")
print(f"  总步数：{dataset.total_steps}")
print(f"  观测空间：{dataset.observation_space}")
print(f"  动作空间：{dataset.action_space}")


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


dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

env = dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space

for batch in dataloader:
    print('id:', batch['id'])
    print('observations:', batch['observations'])
    print('actions', batch['actions'])
    print('rewards', batch['rewards'])
    print('terminations:', batch['terminations'])
    print('truncations', batch['truncations'])
    break


