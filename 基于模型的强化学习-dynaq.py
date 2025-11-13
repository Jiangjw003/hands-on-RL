import time
import gymnasium as gym
import numpy as np
from collections import defaultdict

ENV_ID = "FrozenLake-v1"  # 4×4 冰湖
MAP_NAME = "4x4"  # 也可换 "8x8"
SLIPPERY = True  # 先关闭随机滑动，便于观察模型学习
TOTAL_STEPS = 3000
N_PLANNING = 5  # 每步额外做 5 次模型模拟（Dyna-Q 核心）
ALPHA = 0.1  # Q 学习率
GAMMA = 0.99
EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.995

env = gym.make(ENV_ID, map_name=MAP_NAME, is_slippery=SLIPPERY)

nS, nA = env.observation_space.n, env.action_space.n

Q = np.zeros((nS, nA))

# 1) 记 (s,a)→s' 的频次  ->  估计 p(s'|s,a)
trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # trans[s][a][s'] = cnt

# 2) 记 (s,a,s')→r 的频次  ->  估计 r(s,a,s') 的期望
reward_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "cnt": 0})))


def record(s, a, r, s_next):
    """把一条真实经验写进模型"""
    trans_counts[s][a][s_next] += 1
    reward_counts[s][a][s_next]["sum"] += r
    reward_counts[s][a][s_next]["cnt"] += 1


def sample_model(s, a):
    """根据已学模型返回 (r, s') 样本，贴合真实 p(s'|s,a) 与 r(s,a,s')"""
    # 1) 先算 p(s'|s,a)
    cnt_dict = trans_counts[s][a]  # {s': count}
    total = sum(cnt_dict.values())
    if total == 0:  # 没见过这条 (s,a)，随便返回一个无效值
        return 0.0, s  # 外部调用者需判断并跳过
    s_choices = list(cnt_dict.keys())
    p_vals = [cnt_dict[s_] / total for s_ in s_choices]
    s_next = np.random.choice(s_choices, p=p_vals)

    # 2) 再取 r(s,a,s') 的期望
    rc = reward_counts[s][a][s_next]
    r_mean = rc["sum"] / rc["cnt"] if rc["cnt"] else 0.0
    return r_mean, s_next


def epsilon_greedy(s, eps):
    if np.random.rand() < eps:
        return env.action_space.sample()
    return int(Q[s].argmax())


def update_model(s, a, r, s_):
    record(s, a, r, s_)


def plan_once():
    if not trans_counts:
        return
    # 随机挑一条见过的 (s,a)
    s = np.random.choice(list(trans_counts.keys()))
    a = np.random.choice(list(trans_counts[s].keys()))
    r_model, s_next = sample_model(s, a)
    # 用模型样本做 Q-learning 更新
    Q[s, a] += ALPHA * (r_model + GAMMA * Q[s_next].max() - Q[s, a])


eps = EPS_START
for step in range(1, TOTAL_STEPS + 1):
    s, _ = env.reset()
    done = False
    while not done:
        a = epsilon_greedy(s, eps)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # 真实经验 Q-learning
        Q[s, a] += ALPHA * (r + GAMMA * Q[s_].max() * (not done) - Q[s, a])

        # 更新模型
        update_model(s, a, r, s_)

        # Dyna-Q planning
        for _ in range(N_PLANNING):
            plan_once()

        s = s_

    eps = max(EPS_MIN, eps * EPS_DECAY)

    # 每 500 步测一次平均成功率
    if step % 500 == 0:
        wins = 0
        for _ in range(200):
            s, _ = env.reset()
            while True:
                a = int(Q[s].argmax())
                s, r, ter, tru, _ = env.step(a)
                if ter or tru:
                    wins += r > 0
                    break
        print(f"Step {step:5d} | 200 场评估成功率 {wins / 200:.2%}")

env.close()

policy = np.argmax(Q, axis=1)

test_env = gym.make(ENV_ID, map_name=MAP_NAME, is_slippery=SLIPPERY, render_mode='human')

step = 0

while step <= 10:
    s, _ = test_env.reset()
    while True:
        a = policy[s]
        s, r, ter, tru, _ = test_env.step(a)
        time.sleep(0.5)
        if ter or tru:
            if r == 1:
                print('yes')
            break
    step += 1
test_env.close()
